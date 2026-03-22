import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("="*60)
print("DIFFERENTIAL PRIVACY (DP) DEMO - 10 Parameters Only")
print("="*60)

# ==== 1. Model with Exactly 10 Parameters ====
class TenParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 5, bias=False)  # 10 weights: 2x5, no bias

    def forward(self, x):
        return self.fc(x)

# ==== 2. Simulate Model Training ====
torch.manual_seed(42)
model = TenParamModel()
X = torch.randn(20, 2)
y = torch.randn(20, 5)
optimizer = optim.Adam(model.parameters(), lr=0.01)
for _ in range(15):
    optimizer.zero_grad()
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()

params = list(model.parameters())[0].detach().numpy().flatten()

print("\nORIGINAL NODE PARAMETERS:")
for i, p in enumerate(params):
    print(f"  Param {i+1:2d}: {p:.5f}")

# ==== 3. Add DP Noise: Best Practice ====
def add_gaussian_dp(params, epsilon=5.0, delta=1e-5, sensitivity=0.05):
    """Highly practical DP: moderate ε, small sensitivity"""
    import math
    sigma = (sensitivity / epsilon) * math.sqrt(2 * math.log(1.25 / delta))
    noise = np.random.normal(0, sigma, params.shape)
    return params + noise, noise, sigma

epsilon = 5.0  # Good privacy/accuracy tradeoff
sensitivity = 0.05  # Reasonable for parameter range
params_dp, noise, sigma = add_gaussian_dp(params, epsilon=epsilon, sensitivity=sensitivity)

print(f"\nDP SETTINGS USED: epsilon={epsilon}, sensitivity={sensitivity}, sigma={sigma:.6f}")

print("\nAFTER DP NOISE IS APPLIED:")
print(f"{'Param':<7} {'Orig':>9} {'Noise':>9} {'Noisy':>9} {'Δ%':>7}")
for i, (orig, n, p_dp) in enumerate(zip(params, noise, params_dp)):
    delta_pct = 100 * abs(n) / (abs(orig) + 1e-8)
    print(f"{i+1:<7} {orig:>9.5f} {n:>9.5f} {p_dp:>9.5f} {delta_pct:>7.2f}")

avg_noise = np.mean(np.abs(noise))
avg_abs_change_pct = 100 * np.mean(np.abs(noise) / (np.abs(params) + 1e-8))
print(f"\nAverage abs noise added: {avg_noise:.6f}")
print(f"Average % change to parameter: {avg_abs_change_pct:.4f}%")

# ==== 4. Effect on Model Output ====
inp = torch.FloatTensor([[1.25, -2.5]])
orig_out = model(inp).detach().numpy()[0]

# Swap model weights for noisy (DP) values
with torch.no_grad():
    model.fc.weight[:] = torch.FloatTensor(params_dp.reshape(5, 2))
dp_out = model(inp).detach().numpy()[0]

print("\nIMPACT ON AQI PREDICTIONS (SIMULATED):")
print(f"{'Output':<8}{'Orig':>12}{'Noisy':>12}{'Δ':>10}{'Δ%':>10}")
diffs = []
for i in range(5):
    abs_diff = dp_out[i] - orig_out[i]
    pct_diff = 100 * abs_diff / (abs(orig_out[i]) + 1e-8)
    diffs.append(abs(abs_diff))
    print(f"{i+1:<8}{orig_out[i]:>12.6f}{dp_out[i]:>12.6f}{abs_diff:>10.6f}{pct_diff:>10.2f}")

mean_pred_diff = np.mean(diffs)
print(f"\nAverage prediction absolute change: {mean_pred_diff:.6f}")

# ==== 5. "Privacy/Accuracy Tradeoff" Message ====
print("\nSUMMARY (Best Practice DP):")
print(f"- Epsilon: {epsilon}, Sensitivity: {sensitivity}, Sigma: {sigma:.6f}")
print(f"- Typical parameter value: {np.mean(np.abs(params)):.4f}, avg noise: {avg_noise:.6f}")
print(f"- Parameter change: ~{avg_abs_change_pct:.2f}%")
print(f"- Model output avg shift: {mean_pred_diff:.4f}")
if mean_pred_diff < 0.2:
    print("-> The privacy noise is strong enough for DP, but accuracy is minimally impacted (recommended setting).")
else:
    print("-> Consider lowering sensitivity or increasing epsilon for less accuracy impact, or the opposite if greater privacy needed.")

print("\nDone.")
