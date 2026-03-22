"""
SIMPLE ENCRYPTION DEMO - 10 PARAMETERS
======================================
Shows only Paillier homomorphic encryption of model parameters
No differential privacy - just encryption/decryption
"""

import numpy as np
import torch
import torch.nn as nn
from phe import paillier
import time

print("="*60)
print("PARAMETER ENCRYPTION DEMO (10 Parameters)")
print("="*60)

# ═══════════════════════════════════════════════════════════════
# STEP 1: CREATE SIMPLE 10-PARAMETER MODEL
# ═══════════════════════════════════════════════════════════════

class TenParamModel(nn.Module):
    """Simple model with exactly 10 parameters"""
    def __init__(self):
        super().__init__()
        # 2 inputs × 5 outputs = 10 weights (no bias)
        self.fc = nn.Linear(2, 5, bias=False)
    
    def forward(self, x):
        return self.fc(x)

# Create model and set some weights
torch.manual_seed(42)
model = TenParamModel()

# Extract parameters as numpy array
params = list(model.parameters())[0].detach().cpu().numpy().flatten()

print("\n" + "="*60)
print("STEP 1: ORIGINAL MODEL PARAMETERS (FROM NODE)")
print("="*60)
print(f"\nTotal parameters: {len(params)}")
print(f"\nParameter values:")
for i, p in enumerate(params):
    print(f"  Param {i+1}: {p:.6f}")

# ═══════════════════════════════════════════════════════════════
# STEP 2: GENERATE PAILLIER KEYS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 2: GENERATE ENCRYPTION KEYS")
print("="*60)

print("\nGenerating Paillier keypair (2048-bit)...")
start_time = time.time()
public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
elapsed = time.time() - start_time

print(f"✓ Keypair generated in {elapsed:.2f} seconds")
print(f"✓ Public key ready for encryption")
print(f"✓ Private key kept on server for decryption")

# ═══════════════════════════════════════════════════════════════
# STEP 3: ENCRYPT PARAMETERS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3: ENCRYPT PARAMETERS")
print("="*60)

# Quantization scale (convert float to int for faster encryption)
SCALE = 1_000_000

print(f"\nQuantization scale: {SCALE:,}")
print(f"(Multiply by {SCALE:,} to convert float → integer)")

print(f"\nEncrypting {len(params)} parameters...")
start_time = time.time()

encrypted_params = []
for i, param in enumerate(params):
    # Convert float to integer (quantization)
    quantized = int(round(param * SCALE))
    
    # Encrypt
    encrypted = public_key.encrypt(quantized)
    encrypted_params.append(encrypted)
    
    # Show progress for first 3 and last parameter
    if i < 3 or i == len(params) - 1:
        ciphertext_preview = str(encrypted.ciphertext())[:20] + "..."
        print(f"  Param {i+1}: {param:.6f} → {quantized:>10} → {ciphertext_preview}")

elapsed = time.time() - start_time

print(f"\n✓ Encrypted {len(params)} parameters in {elapsed:.2f} seconds")
print(f"✓ Speed: {len(params)/elapsed:.1f} parameters/second")

# ═══════════════════════════════════════════════════════════════
# STEP 4: SHOW ENCRYPTED DATA (UNREADABLE)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 4: ENCRYPTED DATA (TRANSMITTED TO SERVER)")
print("="*60)

print("\nEncrypted ciphertexts (first 3):")
for i in range(3):
    ciphertext = str(encrypted_params[i].ciphertext())
    print(f"  Param {i+1}: {ciphertext[:50]}... ({len(ciphertext)} digits)")

print(f"\n✓ These values are unreadable without private key")
print(f"✓ Can be transmitted securely over network")

# ═══════════════════════════════════════════════════════════════
# STEP 5: DECRYPT PARAMETERS (ON SERVER)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 5: DECRYPT PARAMETERS (ON SERVER)")
print("="*60)

print(f"\nDecrypting {len(encrypted_params)} parameters...")
start_time = time.time()

decrypted_params = []
for encrypted in encrypted_params:
    # Decrypt (returns integer)
    decrypted_int = private_key.decrypt(encrypted)
    
    # Dequantize (convert back to float)
    decrypted_float = decrypted_int / SCALE
    decrypted_params.append(decrypted_float)

decrypted_params = np.array(decrypted_params)
elapsed = time.time() - start_time

print(f"✓ Decrypted in {elapsed:.2f} seconds")

# ═══════════════════════════════════════════════════════════════
# STEP 6: VERIFY ACCURACY
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 6: VERIFY ENCRYPTION/DECRYPTION ACCURACY")
print("="*60)

print(f"\n{'Param':<8} {'Original':<15} {'Decrypted':<15} {'Difference':<15}")
print("-"*60)

for i in range(len(params)):
    diff = decrypted_params[i] - params[i]
    print(f"{i+1:<8} {params[i]:<15.6f} {decrypted_params[i]:<15.6f} {diff:<15.9f}")

max_error = np.max(np.abs(decrypted_params - params))
avg_error = np.mean(np.abs(decrypted_params - params))

print("\n" + "="*60)
print("ACCURACY STATISTICS")
print("="*60)
print(f"Maximum error: {max_error:.9f}")
print(f"Average error: {avg_error:.9f}")
print(f"Relative error: {avg_error / (np.abs(params).mean() + 1e-8) * 100:.6f}%")

if max_error < 1e-6:
    print("\n✓ PERFECT RECONSTRUCTION (errors due to rounding only)")
else:
    print(f"\n⚠ Some quantization error present")

# ═══════════════════════════════════════════════════════════════
# STEP 7: DEMONSTRATE PREDICTION
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 7: TEST PREDICTIONS")
print("="*60)

# Test input
test_input = torch.FloatTensor([[1.5, 2.5]])

# Original model prediction
original_output = model(test_input).detach().numpy()[0]

# Create new model with decrypted parameters
decrypted_model = TenParamModel()
with torch.no_grad():
    decrypted_model.fc.weight[:] = torch.FloatTensor(
        decrypted_params.reshape(5, 2)
    )

# Decrypted model prediction
decrypted_output = decrypted_model(test_input).detach().numpy()[0]

print(f"\nTest input: {test_input.numpy()[0]}")
print(f"\n{'Output':<10} {'Original':<15} {'After Enc/Dec':<15} {'Difference':<15}")
print("-"*60)

for i in range(len(original_output)):
    diff = decrypted_output[i] - original_output[i]
    print(f"{i+1:<10} {original_output[i]:<15.6f} {decrypted_output[i]:<15.6f} {diff:<15.9f}")

pred_error = np.mean(np.abs(decrypted_output - original_output))
print(f"\nAverage prediction difference: {pred_error:.9f}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\n✓ 10 parameters encrypted with Paillier")
print(f"✓ Encryption time: ~{elapsed:.2f}s for {len(params)} params")
print(f"✓ Decryption accuracy: {avg_error:.9f} average error")
print(f"✓ Prediction impact: {pred_error:.9f} average difference")
print("\nKEY POINTS:")
print("  • Parameters encrypted → unreadable ciphertext")
print("  • Server decrypts using private key")
print("  • Minimal loss due to quantization (negligible)")
print("  • Model predictions nearly identical after encryption/decryption")

print("\n" + "="*60)
