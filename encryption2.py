"""
MULTI-NODE FEDERATED AGGREGATION DEMO WITH PAILLIER ENCRYPTION
==============================================================
- 3 simulated nodes, each with its own 10-parameter model
- Each node encrypts its parameters with Paillier
- Central server receives all, aggregates (averages) on encrypted data
- Server decrypts only the final aggregated model (never sees individual models)
- FIXED: Proper tensor handling for predictions

Dependencies: numpy, torch, phe (pip install numpy torch phe)
"""

import numpy as np
import torch
import torch.nn as nn
from phe import paillier
import time

# print("="*70)
# print("MULTI-NODE FEDERATED LEARNING WITH PAILLIER ENCRYPTION")
# print("="*70)

# ═════════════════════════════════════════════════════════════════
# STEP 1: Model definition (10 parameters)
# ═════════════════════════════════════════════════════════════════

class TenParamModel(nn.Module):
    """Simple model with exactly 10 parameters"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 5, bias=False)  # 2*5=10 parameters

    def forward(self, x):
        return self.fc(x)

# ═════════════════════════════════════════════════════════════════
# STEP 2: Create 3 nodes with different models
# ═════════════════════════════════════════════════════════════════

# print("\n" + "="*70)
# print("STEP 1: CREATE 3 NODES WITH THEIR OWN MODELS")
# print("="*70)

nodes = []
num_nodes = 3

for i in range(num_nodes):
    torch.manual_seed(42 + i)
    model = TenParamModel()
    nodes.append(model)
    params = list(model.parameters())[0].detach().cpu().numpy().flatten()
    print(f"\nNode {i+1} Parameters:")
    for j, p in enumerate(params):
        print(f"  Param {j+1}: {p:.6f}")

# ═════════════════════════════════════════════════════════════════
# STEP 3: Generate Paillier keys (once, shared public key)
# ═════════════════════════════════════════════════════════════════

# print("\n" + "="*70)
# print("STEP 2: GENERATE PAILLIER ENCRYPTION KEYS")
# print("="*70)

print("\nGenerating 2048-bit Paillier keypair...")
start_time = time.time()
public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
elapsed = time.time() - start_time

print(f"✓ Keypair generated in {elapsed:.2f} seconds")
# print(f"✓ Public key shared with all nodes")
# print(f"✓ Private key kept only on server")

SCALE = 1_000_000

# ═════════════════════════════════════════════════════════════════
# STEP 4: Each node encrypts and sends parameters
# ═════════════════════════════════════════════════════════════════

# print("\n" + "="*70)
# print("STEP 3: NODES ENCRYPT AND SEND PARAMETERS TO SERVER")
# print("="*70)

all_encrypted_params = []
all_original_params = []

for node_idx, model in enumerate(nodes):
    print(f"\n--- Node {node_idx+1} ---")
    
    # Extract parameters
    params = list(model.parameters())[0].detach().cpu().numpy().flatten()
    all_original_params.append(params)
    
    print(f"Original parameters (first 3): {params[:3].round(6)}")
    
    # Quantize (float → integer for faster encryption)
    quantized = np.round(params * SCALE).astype(int)
    print(f"Quantized (×{SCALE}): {quantized[:3]}")
    
    # Encrypt with Paillier
    print(f"Encrypting {len(params)} parameters...")
    start_time = time.time()
    encrypted = [public_key.encrypt(int(x)) for x in quantized]
    elapsed = time.time() - start_time
    
    print(f"✓ Encrypted in {elapsed:.3f} seconds")
    print(f"Ciphertext preview (first 3): {[str(e.ciphertext())[:20]+'...' for e in encrypted[:3]]}")
    
    all_encrypted_params.append(encrypted)

# ═════════════════════════════════════════════════════════════════
# STEP 5: Server aggregates encrypted parameters
# ═════════════════════════════════════════════════════════════════

# print("\n" + "="*70)
# print("STEP 4: SERVER AGGREGATES ON ENCRYPTED DATA")
# print("="*70)

num_params = len(all_encrypted_params[0])
print(f"\nServer received {num_nodes} encrypted models")
print(f"Each with {num_params} parameters")

print(f"\nAggregating (weighted average on encrypted data)...")
print(f"Using equal weights: 1/{num_nodes} = {1/num_nodes:.4f} each\n")

# Compute weights
weights = np.ones(num_nodes) / num_nodes
print(f"Weights: {weights}")

# Homomorphic aggregation
enc_agg = []
for p in range(num_params):
    # Start with first node's encrypted parameter
    enc_sum = all_encrypted_params[0][p] * weights[0]
    
    # Add other nodes' encrypted parameters
    for n in range(1, num_nodes):
        enc_sum = enc_sum + (all_encrypted_params[n][p] * weights[n])
    
    enc_agg.append(enc_sum)

print(f"\n✓ Aggregation complete (on encrypted data)")
# print(f"✓ Server holding {num_params} encrypted aggregate parameters")
# print(f"✓ Server CANNOT read these values (encrypted!)")

# ═════════════════════════════════════════════════════════════════
# STEP 6: Server decrypts only final aggregated model
# ═════════════════════════════════════════════════════════════════

# print("\n" + "="*70)
print("STEP 5: SERVER DECRYPTS FINAL AGGREGATED MODEL")
# print("="*70)

print(f"\nDecrypting aggregated parameters...")
start_time = time.time()
decrypted_ints = [private_key.decrypt(x) for x in enc_agg]
elapsed = time.time() - start_time

# Dequantize (integer → float)
params_global = np.array(decrypted_ints) / SCALE

print(f"✓ Decrypted in {elapsed:.3f} seconds")
print(f"\nGlobal Aggregated Model Parameters:")
for i, p in enumerate(params_global):
    print(f"  Param {i+1}: {p:.6f}")

# ═════════════════════════════════════════════════════════════════
# STEP 7: Verify accuracy (compare with manual average)
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 6: VERIFY AGGREGATION CORRECTNESS")
print("="*70)

# Manual average (for verification only)
manual_avg = np.mean(all_original_params, axis=0)

print(f"\nManual average (for verification):")
for i in range(3):
    print(f"  Param {i+1}: {manual_avg[i]:.6f}")

print(f"\nEncrypted aggregation result:")
for i in range(3):
    print(f"  Param {i+1}: {params_global[i]:.6f}")

print(f"\nDifference (due to rounding):")
diff = np.abs(params_global - manual_avg)
for i in range(3):
    print(f"  Param {i+1}: {diff[i]:.9f}")

max_error = np.max(diff)
# print(f"\nMax error: {max_error:.9f} (negligible, due to quantization)")

# ═════════════════════════════════════════════════════════════════
# STEP 8: Test global model predictions
# ═════════════════════════════════════════════════════════════════

# print("\n" + "="*70)
# print("STEP 7: TEST GLOBAL MODEL PREDICTIONS")
# print("="*70)

# Test input
test_input = torch.FloatTensor([[1.5, 2.5]])

# print(f"\nTest input: {test_input.numpy()[0]}")

# Original model predictions (for comparison)
# print(f"\nOriginal Node Predictions:")
for node_idx, model in enumerate(nodes):
    model.eval()
    with torch.no_grad():
        pred = model(test_input)
    # print(f"  Node {node_idx+1}: {pred.numpy().flatten()}")

# Global model prediction
# print(f"\nGlobal Aggregated Model Prediction:")
model_global = TenParamModel()
with torch.no_grad():
    model_global.fc.weight[:] = torch.FloatTensor(params_global.reshape(5, 2))
    pred_global = model_global(test_input)

# print(f"  Output: {pred_global.numpy().flatten()}")

# ═════════════════════════════════════════════════════════════════
# STEP 9: Summary
# ═════════════════════════════════════════════════════════════════

# print("\n" + "="*70)
# print("PRIVACY & SECURITY SUMMARY")
# print("="*70)

# print("="*70)
# print("✓ MULTI-NODE FEDERATED LEARNING WITH ENCRYPTION: SUCCESS!")
# print("="*70)
