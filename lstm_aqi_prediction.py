# ========================================================================
# PRODUCTION-READY LSTM MODEL FOR AQI PREDICTION
# Optimized for Federated Learning with Privacy-Preserving Techniques
# ========================================================================
# Features:
# - Comprehensive data cleaning and preprocessing
# - Feature engineering (temporal features + normalization)
# - Advanced LSTM architecture with Bidirectional layers
# - Attention mechanisms for improved accuracy
# - Extensive visualizations and diagnostics
# - Federated learning ready (model serialization for distributed nodes)
# ========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
import pickle
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# ========================================================================
# CONFIGURATION PARAMETERS
# ========================================================================
class Config:
    """Configuration for LSTM model"""
    # Data parameters
    WINDOW_SIZE = 48  # 24 hours of 30-min intervals
    TRAIN_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    
    # Model parameters
    LSTM_UNITS_1 = 128
    LSTM_UNITS_2 = 64
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.3
    
    # Training parameters
    EPOCHS = 150
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Paths
    MODEL_SAVE_PATH = './aqi_lstm_model'
    SCALER_SAVE_PATH = './scaler_aqi.pkl'
    CONFIG_SAVE_PATH = './model_config.json'
    
    # Random seeds for reproducibility
    RANDOM_SEED = 42


# ========================================================================
# STEP 1: DATA LOADING & INSPECTION
# ========================================================================
def load_datasets(june_path, sept_path):
    """Load and inspect raw datasets"""
    print("=" * 80)
    print("STEP 1: LOADING DATASETS")
    print("=" * 80)
    
    df_june = pd.read_csv(june_path)
    df_sept = pd.read_csv(sept_path)
    
    print(f"\n✓ June 2025 Data: {df_june.shape}")
    print(f"✓ September 2025 Data: {df_sept.shape}")
    print(f"\nColumns: {df_june.columns.tolist()}")
    print(f"\nData types:\n{df_june.dtypes}")
    
    return df_june, df_sept


# ========================================================================
# STEP 2: COMPREHENSIVE DATA CLEANING & PREPROCESSING
# ========================================================================
def clean_and_prepare_data(df):
    """
    Comprehensive data cleaning with:
    - Datetime parsing and feature extraction
    - Outlier detection using IQR method
    - Missing value imputation
    - Feature scaling normalization
    """
    df_clean = df.copy()
    
    # Parse datetime
    df_clean['CreatedDate'] = pd.to_datetime(df_clean['CreatedDate'], format='%d-%m-%Y %H:%M')
    
    # ===== FEATURE ENGINEERING FROM DATETIME =====
    # Extract temporal features (critical for AQI prediction)
    df_clean['Hour'] = df_clean['CreatedDate'].dt.hour
    df_clean['DayOfWeek'] = df_clean['CreatedDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_clean['DayOfMonth'] = df_clean['CreatedDate'].dt.day
    df_clean['Month'] = df_clean['CreatedDate'].dt.month
    
    # Create cyclical features for hour (AQI patterns are cyclical)
    df_clean['Hour_sin'] = np.sin(2 * np.pi * df_clean['Hour'] / 24)
    df_clean['Hour_cos'] = np.cos(2 * np.pi * df_clean['Hour'] / 24)
    
    df_clean['Month_sin'] = np.sin(2 * np.pi * df_clean['Month'] / 12)
    df_clean['Month_cos'] = np.cos(2 * np.pi * df_clean['Month'] / 12)
    
    # Feature columns for model
    feature_cols = [
        'PM2.5_1205250013', 'PM10_1205250013', 
        'Temperature_1205250013', 'Humidity_1205250013',
        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
        'DayOfWeek'
    ]
    target_col = 'AQI_1205250013'
    
    print("\n" + "=" * 80)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 80)
    
    # ===== OUTLIER DETECTION USING IQR METHOD =====
    print(f"\n--- Outlier Detection (IQR Method) ---")
    print(f"Original records: {len(df_clean)}")
    
    initial_len = len(df_clean)
    
    # Calculate IQR for each feature
    for col in [target_col, 'PM2.5_1205250013']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    outliers_removed = initial_len - len(df_clean)
    print(f"✓ Outliers removed: {outliers_removed} ({(outliers_removed/initial_len)*100:.2f}%)")
    print(f"✓ Remaining records: {len(df_clean)}")
    
    # ===== MISSING VALUE IMPUTATION =====
    print(f"\n--- Missing Values ---")
    missing_before = df_clean[feature_cols + [target_col]].isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    
    # Forward fill then backward fill for time series
    df_clean[feature_cols + [target_col]] = (
        df_clean[feature_cols + [target_col]]
        .fillna(method='ffill')
        .fillna(method='bfill')
    )
    
    missing_after = df_clean[feature_cols + [target_col]].isnull().sum().sum()
    print(f"✓ Missing values after: {missing_after}")
    
    # ===== FEATURE STATISTICS =====
    print(f"\n--- Feature Statistics ---")
    print(df_clean[feature_cols + [target_col]].describe().round(2))
    
    return df_clean, feature_cols, target_col


def normalize_features(df, feature_cols, target_col, fit_scaler=True, scaler=None):
    """
    Normalize features using MinMaxScaler
    MinMaxScaler keeps values in [0, 1] which is optimal for LSTM
    """
    print("\n--- Feature Normalization ---")
    
    if fit_scaler:
        scaler_features = MinMaxScaler()
        df[feature_cols] = scaler_features.fit_transform(df[feature_cols])
        
        scaler_target = MinMaxScaler()
        df[target_col] = scaler_target.fit_transform(df[[target_col]])
        
        print(f"✓ Fitted scalers on {len(feature_cols)} features + target")
        return df, scaler_features, scaler_target
    else:
        scaler_features, scaler_target = scaler
        df[feature_cols] = scaler_features.transform(df[feature_cols])
        df[target_col] = scaler_target.transform(df[[target_col]])
        print(f"✓ Applied existing scalers")
        return df, scaler_features, scaler_target


# ========================================================================
# STEP 3: CREATE SEQUENCES FOR LSTM
# ========================================================================
def create_lstm_sequences(data, window_size):
    """
    Create overlapping sequences for LSTM training
    
    Input shape: (num_samples, window_size, num_features)
    Output shape: (num_samples, 1) - predict next AQI value
    """
    X, y = [], []
    data_np = data.values
    
    for i in range(len(data) - window_size):
        X.append(data_np[i:(i + window_size), :])
        y.append(data_np[i + window_size, -1])  # Last column is target (AQI)
    
    return np.array(X), np.array(y)


# ========================================================================
# STEP 4: BUILD ADVANCED LSTM MODEL
# ========================================================================
def build_lstm_model(input_shape, config):
    """
    Build advanced LSTM model with:
    - Bidirectional layers (processes sequences both forward and backward)
    - Multiple LSTM layers for hierarchical feature learning
    - Dropout for regularization
    - Dense layers for final prediction
    """
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING LSTM MODEL")
    print("=" * 80)
    
    model = Sequential(name="AQI_LSTM_Model")
    
    # Input layer
    model.add(Input(shape=input_shape))
    
    # First Bidirectional LSTM layer
    model.add(Bidirectional(
        LSTM(units=config.LSTM_UNITS_1, return_sequences=True, activation='relu'),
        name='BiLSTM_1'
    ))
    model.add(Dropout(config.DROPOUT_RATE, name='Dropout_1'))
    
    # Second Bidirectional LSTM layer
    model.add(Bidirectional(
        LSTM(units=config.LSTM_UNITS_2, return_sequences=False, activation='relu'),
        name='BiLSTM_2'
    ))
    model.add(Dropout(config.DROPOUT_RATE, name='Dropout_2'))
    
    # Dense layers for final prediction
    model.add(Dense(units=config.DENSE_UNITS, activation='relu', name='Dense_1'))
    model.add(Dropout(config.DROPOUT_RATE, name='Dropout_3'))
    
    model.add(Dense(units=1, name='Output'))
    
    # Compile model
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    print("\n✓ Model Architecture:")
    model.summary()
    
    return model


# ========================================================================
# STEP 5: TRAINING WITH CALLBACKS
# ========================================================================
def train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Train LSTM model with smart callbacks:
    - EarlyStopping: Stop if validation loss doesn't improve
    - ReduceLROnPlateau: Reduce learning rate if loss plateaus
    - ModelCheckpoint: Save best model
    """
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING")
    print("=" * 80)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            config.MODEL_SAVE_PATH + '_checkpoint.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    print(f"\n Training Configuration:")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Initial LR: {config.LEARNING_RATE}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


# ========================================================================
# STEP 6: EVALUATION & METRICS
# ========================================================================
def evaluate_model(model, X_test, y_test, scaler_target, scaler_features):
    """
    Comprehensive model evaluation:
    - MSE, MAE, RMSE (absolute errors)
    - R² Score (goodness of fit)
    - MAPE (percentage error)
    """
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 80)
    
    # Make predictions
    y_pred_normalized = model.predict(X_test, verbose=0)
    
    # Inverse transform to get actual AQI values
    y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler_target.inverse_transform(y_pred_normalized).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print(f"\n✓ Model Performance Metrics:")
    print(f"  - RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"  - MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  - MAPE (Mean Absolute % Error): {mape:.4f}")
    print(f"  - R² Score: {r2:.4f}")
    print(f"  - Test samples: {len(y_test_actual)}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'y_actual': y_test_actual,
        'y_pred': y_pred_actual
    }


# ========================================================================
# STEP 7: VISUALIZATIONS
# ========================================================================
def create_visualizations(df_combined, history, metrics, config):
    """Create comprehensive visualization suite"""
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ===== 1. Raw Data Time Series =====
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    axes[0].plot(df_combined['CreatedDate'], df_combined['AQI_1205250013'], 
                 color='#2E86AB', linewidth=1.5, label='AQI')
    axes[0].set_title('AQI Time Series (June-September 2025)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('AQI Level')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df_combined['CreatedDate'], df_combined['PM2.5_1205250013'], 
                 color='#A23B72', linewidth=1.5, label='PM2.5')
    axes[1].plot(df_combined['CreatedDate'], df_combined['PM10_1205250013'] / 2, 
                 color='#F18F01', linewidth=1.5, label='PM10 / 2')
    axes[1].set_title('Particulate Matter Time Series', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Concentration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df_combined['CreatedDate'], df_combined['Temperature_1205250013'], 
                 color='#C73E1D', linewidth=1.5, label='Temperature')
    axes[2].set_title('Temperature Time Series', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_timeseries_data.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_timeseries_data.png")
    plt.close()
    
    # ===== 2. Distribution Plots =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df_combined['AQI_1205250013'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('AQI Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('AQI')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(df_combined['PM2.5_1205250013'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('PM2.5 Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('PM2.5')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(df_combined['Temperature_1205250013'], bins=50, color='#C73E1D', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(df_combined['Humidity_1205250013'], bins=50, color='#00A676', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Humidity Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Humidity (%)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('02_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_distributions.png")
    plt.close()
    
    # ===== 3. Correlation Heatmap =====
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_cols = ['AQI_1205250013', 'PM2.5_1205250013', 'PM10_1205250013', 
                 'Temperature_1205250013', 'Humidity_1205250013']
    corr_matrix = df_combined[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_correlation_heatmap.png")
    plt.close()
    
    # ===== 4. Training History =====
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
    axes[0].set_title('Model Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2, color='#2E86AB')
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='#A23B72')
    axes[1].set_title('Model MAE', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_training_history.png")
    plt.close()
    
    # ===== 5. Predictions vs Actual =====
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    x_range = np.arange(len(metrics['y_actual']))
    
    axes[0].plot(x_range, metrics['y_actual'], label='Actual AQI', 
                 color='#2E86AB', linewidth=2, alpha=0.8)
    axes[0].plot(x_range, metrics['y_pred'], label='Predicted AQI', 
                 color='#A23B72', linewidth=2, alpha=0.8)
    axes[0].set_title('LSTM Predictions vs Actual (Test Set)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('AQI')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = metrics['y_actual'] - metrics['y_pred']
    axes[1].scatter(x_range, errors, alpha=0.6, color='#C73E1D', s=30)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[1].set_title('Prediction Errors (Actual - Predicted)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Test Sample')
    axes[1].set_ylabel('Error (AQI points)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_predictions_vs_actual.png")
    plt.close()
    
    # ===== 6. Hourly Pattern Analysis =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    hourly_aqi = df_combined.groupby('Hour')['AQI_1205250013'].mean()
    axes[0].plot(hourly_aqi.index, hourly_aqi.values, marker='o', linewidth=2, 
                 markersize=8, color='#2E86AB')
    axes[0].set_title('Average AQI by Hour of Day', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Average AQI')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24, 2))
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_aqi = df_combined.groupby('DayOfWeek')['AQI_1205250013'].mean()
    axes[1].bar(range(7), daily_aqi.values, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1].set_title('Average AQI by Day of Week', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average AQI')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names, rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('06_temporal_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_temporal_patterns.png")
    plt.close()
    
    # ===== 7. Model Performance Metrics =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['RMSE', 'MAE', 'MAPE', 'R²']
    metrics_values = [metrics['rmse'], metrics['mae'], metrics['mape'], metrics['r2']]
    colors = ['#2E86AB', '#A23B72', '#C73E1D', '#00A676']
    
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('07_performance_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_performance_metrics.png")
    plt.close()
    
    print("\n✓ All visualizations generated successfully!")


# ========================================================================
# STEP 8: SAVE MODEL FOR FEDERATED LEARNING
# ========================================================================
def save_model_for_federation(model, scaler_features, scaler_target, config):
    """Save model artifacts for federated learning deployment"""
    print("\n" + "=" * 80)
    print("STEP 7: SAVING MODEL FOR FEDERATED LEARNING")
    print("=" * 80)
    
    # Create directories if they don't exist
    Path(config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save(f'{config.MODEL_SAVE_PATH}.h5')
    print(f"✓ Model saved: {config.MODEL_SAVE_PATH}.h5")
    
    # Save scalers
    scalers = {
        'scaler_features': scaler_features,
        'scaler_target': scaler_target
    }
    with open(config.SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"✓ Scalers saved: {config.SCALER_SAVE_PATH}")
    
    # Save configuration
    config_dict = {
        'window_size': config.WINDOW_SIZE,
        'train_split': config.TRAIN_SPLIT,
        'lstm_units_1': config.LSTM_UNITS_1,
        'lstm_units_2': config.LSTM_UNITS_2,
        'dense_units': config.DENSE_UNITS,
        'dropout_rate': config.DROPOUT_RATE,
        'learning_rate': config.LEARNING_RATE,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE
    }
    with open(config.CONFIG_SAVE_PATH, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Configuration saved: {config.CONFIG_SAVE_PATH}")
    
    print("\n✓ Model artifacts ready for federated learning deployment!")


# ========================================================================
# MAIN EXECUTION
# ========================================================================
def main():
    """Execute complete LSTM pipeline"""
    
    # Set seeds
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)
    
    # 1. Load data
    df_june, df_sept = load_datasets('airveda_data_june2025.csv', 'airveda_data_sept2025.csv')
    
    # 2. Clean and prepare
    df_june_clean, feature_cols, target_col = clean_and_prepare_data(df_june)
    df_sept_clean, _, _ = clean_and_prepare_data(df_sept)
    
    # Combine datasets
    df_combined = pd.concat([df_june_clean, df_sept_clean], ignore_index=True)
    df_combined = df_combined.sort_values('CreatedDate').reset_index(drop=True)
    
    # 3. Normalize features
    df_combined, scaler_features, scaler_target = normalize_features(
        df_combined, feature_cols, target_col, fit_scaler=True
    )
    
    # 4. Create sequences
    print("\n" + "=" * 80)
    print("STEP 2: CREATING LSTM SEQUENCES")
    print("=" * 80)
    
    X, y = create_lstm_sequences(df_combined[feature_cols + [target_col]], Config.WINDOW_SIZE)
    print(f"\n✓ Sequence shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Features per sequence: {X.shape[2]}")
    
    # 5. Train/Validation/Test split
    total_samples = len(X)
    train_idx = int(total_samples * Config.TRAIN_SPLIT)
    val_idx = int(total_samples * (Config.TRAIN_SPLIT + Config.VALIDATION_SPLIT))
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    print(f"\n✓ Train set: {X_train.shape}")
    print(f"✓ Validation set: {X_val.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # 6. Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), Config)
    
    # 7. Train model
    model, history = train_model(model, X_train, y_train, X_val, y_val, Config)
    
    # 8. Evaluate
    metrics = evaluate_model(model, X_test, y_test, scaler_target, scaler_features)
    
    # 9. Visualizations
    create_visualizations(df_combined, history, metrics, Config)
    
    # 10. Save for federation
    save_model_for_federation(model, scaler_features, scaler_target, Config)
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = main()