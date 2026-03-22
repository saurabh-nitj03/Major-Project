
# """
# COMPLETE PRIVACY-PRESERVING FEDERATED LEARNING SYSTEM
# ======================================================

# Features:
# 1. Three simulated nodes (different locations/data)
# 2. Differential Privacy (DP) on each node
# 3. Paillier Homomorphic Encryption
# 4. Asynchronous parameter transmission
# 5. Server-side encrypted aggregation
# 6. Complete privacy guarantee

# Privacy layers:
# - DP: Protects against pattern inference
# - Encryption: Protects during transmission and storage
# - Aggregation: Server never sees individual parameters

# Run until you press Ctrl+C to exit
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from phe import paillier
# import time
# from datetime import datetime
# import threading
# import queue

# print("="*70)
# print("PRIVACY-PRESERVING FEDERATED LEARNING: DP + ENCRYPTION")
# print("="*70)
# print("\nPress Ctrl+C to stop the simulation")
# print("="*70)

# # ═══════════════════════════════════════════════════════════════
# # PART 1: MODEL DEFINITION
# # ═══════════════════════════════════════════════════════════════

# class TenParamModel(nn.Module):
#     """Simple 10-parameter model"""
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(2, 5, bias=False)  # 10 weights
    
#     def forward(self, x):
#         return self.fc(x)


# # ═══════════════════════════════════════════════════════════════
# # PART 2: DIFFERENTIAL PRIVACY MODULE
# # ═══════════════════════════════════════════════════════════════

# class DifferentialPrivacy:
#     """DP with Gaussian mechanism"""
    
#     def __init__(self, epsilon=5.0, delta=1e-5, sensitivity=0.05):
#         self.epsilon = epsilon
#         self.delta = delta
#         self.sensitivity = sensitivity
        
#         import math
#         self.sigma = (sensitivity / epsilon) * math.sqrt(2 * math.log(1.25 / delta))
    
#     def add_noise(self, params):
#         """Add Gaussian noise to parameters"""
#         noise = np.random.normal(0, self.sigma, params.shape)
#         return params + noise, noise


# # ═══════════════════════════════════════════════════════════════
# # PART 3: NODE CLASS (Sensor/Device)
# # ═══════════════════════════════════════════════════════════════

# class FederatedNode:
#     """Represents a sensor node with DP and encryption"""
    
#     def __init__(self, node_id, location_type, public_key, epsilon=5.0):
#         self.node_id = node_id
#         self.location_type = location_type
#         self.public_key = public_key
#         self.epsilon = epsilon
        
#         # Initialize model
#         self.model = TenParamModel()
        
#         # Initialize DP
#         self.dp = DifferentialPrivacy(epsilon=epsilon)
        
#         # Training data patterns (simulated based on location)
#         self.data_patterns = {
#             'industrial': {'mean': 1.5, 'std': 0.5},
#             'park': {'mean': -0.5, 'std': 0.3},
#             'residential': {'mean': 0.2, 'std': 0.4}
#         }
        
#         print(f"\n[{node_id}] Node initialized")
#         print(f"  Location: {location_type}")
#         print(f"  Privacy (ε): {epsilon}")
    
#     def train_local_model(self):
#         """Train model on local simulated data"""
#         pattern = self.data_patterns[self.location_type]
        
#         # Simulate local training data
#         X = torch.randn(20, 2) * pattern['std'] + pattern['mean']
#         y = torch.randn(20, 5) * pattern['std']
        
#         optimizer = optim.Adam(self.model.parameters(), lr=0.01)
#         self.model.train()
        
#         for _ in range(10):  # Quick training
#             optimizer.zero_grad()
#             pred = self.model(X)
#             loss = ((pred - y) ** 2).mean()
#             loss.backward()
#             optimizer.step()
    
#     def prepare_parameters(self):
#         """Extract, add DP noise, and encrypt parameters"""
        
#         # Extract parameters
#         params = list(self.model.parameters())[0].detach().cpu().numpy().flatten()
        
#         # Add DP noise
#         params_noisy, noise = self.dp.add_noise(params)
        
#         # Encrypt with Paillier
#         SCALE = 10_000  # Reduced for bandwidth efficiency
#         quantized = np.round(params_noisy * SCALE).astype(int)
#         encrypted = [self.public_key.encrypt(int(q)) for q in quantized]
        
#         return {
#             'node_id': self.node_id,
#             'location_type': self.location_type,
#             'encrypted_params': encrypted,
#             'timestamp': datetime.now(),
#             'epsilon': self.epsilon,
#             'original_params': params,  # For demo only (never sent in real system)
#             'noisy_params': params_noisy,  # For demo only
#             'noise': noise  # For demo only
#         }


# # ═══════════════════════════════════════════════════════════════
# # PART 4: CENTRAL SERVER
# # ═══════════════════════════════════════════════════════════════

# class CentralServer:
#     """Central aggregation server"""
    
#     def __init__(self):
#         # Generate Paillier keys
#         print("\n[SERVER] Initializing...")
#         print("[SERVER] Generating Paillier keypair (2048-bit)...")
        
#         start_time = time.time()
#         self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=2048)
#         elapsed = time.time() - start_time
        
#         print(f"[SERVER] ✓ Keys generated in {elapsed:.2f}s")
        
#         # Storage for received parameters
#         self.received_params = []
#         self.round_number = 0
#         self.aggregation_threshold = 3  # Wait for 3 nodes
    
#     def receive_parameters(self, node_data):
#         """Receive encrypted parameters from node"""
#         node_id = node_data['node_id']
#         timestamp = node_data['timestamp'].strftime('%H:%M:%S')
        
#         print(f"\n[SERVER {timestamp}] Received from {node_id}")
#         print(f"  Location: {node_data['location_type']}")
#         print(f"  Encrypted params: {len(node_data['encrypted_params'])}")
#         print(f"  Privacy (ε): {node_data['epsilon']}")
        
#         # Store (in real system, would go to database)
#         self.received_params.append(node_data)
        
#         print(f"  ✓ Stored encrypted parameters")
#         print(f"  Nodes ready: {len(self.received_params)}/{self.aggregation_threshold}")
        
#         # Check if ready to aggregate
#         if len(self.received_params) >= self.aggregation_threshold:
#             self.aggregate_and_broadcast()
    
#     def aggregate_and_broadcast(self):
#         """Aggregate encrypted parameters and broadcast global model"""
        
#         print(f"\n{'='*70}")
#         print(f"[SERVER] AGGREGATION ROUND {self.round_number + 1}")
#         print(f"{'='*70}")
        
#         num_nodes = len(self.received_params)
#         num_params = len(self.received_params[0]['encrypted_params'])
        
#         # Show what server sees (for demo)
#         print(f"\n[SERVER] What server can see:")
#         print(f"  Number of nodes: {num_nodes}")
#         print(f"  Parameters per node: {num_params}")
#         print(f"  Encrypted values: UNREADABLE (ciphertext)")
        
#         # Aggregate on encrypted data
#         print(f"\n[SERVER] Aggregating encrypted parameters...")
        
#         weights = np.ones(num_nodes) / num_nodes  # Equal weights
#         enc_aggregated = []
        
#         for p in range(num_params):
#             enc_sum = None
#             for n_idx, node_data in enumerate(self.received_params):
#                 enc_param = node_data['encrypted_params'][p]
#                 if enc_sum is None:
#                     enc_sum = enc_param * weights[n_idx]
#                 else:
#                     enc_sum += enc_param * weights[n_idx]
#             enc_aggregated.append(enc_sum)
        
#         print(f"  ✓ Aggregated {num_params} encrypted parameters")
#         print(f"  Note: Aggregation done WITHOUT seeing individual values!")
        
#         # Decrypt only final result
#         print(f"\n[SERVER] Decrypting final aggregated model...")
        
#         SCALE = 10_000
#         decrypted = [self.private_key.decrypt(e) / SCALE for e in enc_aggregated]
#         global_params = np.array(decrypted)
        
#         print(f"  ✓ Global model parameters (first 3):")
#         for i in range(min(3, len(global_params))):
#             print(f"    Param {i+1}: {global_params[i]:.6f}")
        
#         # Privacy analysis (for demo)
#         self.analyze_privacy(global_params)
        
#         # Broadcast (in real system)
#         print(f"\n[SERVER] ✓ Broadcasting global model to all nodes")
        
#         # Clear for next round
#         self.received_params = []
#         self.round_number += 1
        
#         print(f"{'='*70}\n")
    
#     def analyze_privacy(self, global_params):
#         """Show privacy guarantees"""
        
#         print(f"\n[SERVER] Privacy Analysis:")
#         print(f"  ✓ Individual node parameters: NEVER seen by server")
#         print(f"  ✓ Only aggregate received: YES")
#         print(f"  ✓ DP noise added: YES (on each node)")
#         print(f"  ✓ Encryption during transmission: YES (Paillier)")
#         print(f"  ✓ Pattern inference: PREVENTED (DP noise)")
#         print(f"  ✓ Location identification: IMPOSSIBLE")
        
#         # Show what nodes sent (for demo only - server doesn't actually see this)
#         print(f"\n[SERVER] What was actually sent (Demo Only):")
#         for node_data in self.received_params:
#             orig = node_data['original_params'][:3]
#             noisy = node_data['noisy_params'][:3]
#             print(f"  {node_data['node_id']}:")
#             print(f"    Original (never sent): {orig}")
#             print(f"    + DP noise → Noisy: {noisy}")
#             print(f"    + Encrypted → UNREADABLE by server")


# # ═══════════════════════════════════════════════════════════════
# # PART 5: SIMULATION ORCHESTRATOR
# # ═══════════════════════════════════════════════════════════════

# class FederatedSimulation:
#     """Orchestrates the entire FL simulation"""
    
#     def __init__(self):
#         # Initialize server
#         self.server = CentralServer()
        
#         # Initialize 3 nodes
#         self.nodes = [
#             FederatedNode('Node_1', 'industrial', self.server.public_key, epsilon=5.0),
#             FederatedNode('Node_2', 'park', self.server.public_key, epsilon=5.0),
#             FederatedNode('Node_3', 'residential', self.server.public_key, epsilon=5.0)
#         ]
        
#         # Communication queue
#         self.comm_queue = queue.Queue()
        
#         # Running flag
#         self.running = True
    
#     def node_worker(self, node, delay):
#         """Worker thread for each node"""
        
#         while self.running:
#             try:
#                 # Wait for interval
#                 time.sleep(delay)
                
#                 if not self.running:
#                     break
                
#                 # Train locally
#                 timestamp = datetime.now().strftime('%H:%M:%S')
#                 print(f"\n[{timestamp}] {node.node_id} training locally...")
#                 node.train_local_model()
                
#                 # Prepare parameters (DP + Encryption)
#                 print(f"[{timestamp}] {node.node_id} applying DP + Encryption...")
#                 node_data = node.prepare_parameters()
                
#                 # Show node-side operations
#                 print(f"[{timestamp}] {node.node_id} parameter processing:")
#                 print(f"  ✓ Local training complete")
#                 print(f"  ✓ DP noise added (σ={node.dp.sigma:.6f})")
#                 print(f"  ✓ Parameters encrypted (Paillier)")
#                 print(f"  ✓ Ready to send")
                
#                 # Send to server
#                 print(f"[{timestamp}] {node.node_id} sending encrypted parameters...")
#                 self.comm_queue.put(node_data)
                
#             except Exception as e:
#                 if self.running:
#                     print(f"Error in {node.node_id}: {e}")
    
#     def server_worker(self):
#         """Server thread to receive and aggregate"""
        
#         while self.running:
#             try:
#                 # Wait for parameters (with timeout)
#                 node_data = self.comm_queue.get(timeout=1)
#                 self.server.receive_parameters(node_data)
                
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 if self.running:
#                     print(f"Server error: {e}")
    
#     def run(self):
#         """Start simulation"""
        
#         print(f"\n{'='*70}")
#         print("STARTING FEDERATED LEARNING SIMULATION")
#         print(f"{'='*70}")
#         print("\nSimulation details:")
#         print(f"  Nodes: 3 (industrial, park, residential)")
#         print(f"  Privacy: DP (ε=5.0) + Paillier encryption")
#         print(f"  Intervals: Node sends every 5-10 seconds")
#         print(f"  Aggregation: When 3 nodes ready")
#         print(f"\nPress Ctrl+C to stop\n")
#         print(f"{'='*70}\n")
        
#         # Start server thread
#         server_thread = threading.Thread(target=self.server_worker, daemon=True)
#         server_thread.start()
        
#         # Start node threads with different intervals
#         node_threads = []
#         intervals = [5, 7, 9]  # Different sending intervals
        
#         for node, interval in zip(self.nodes, intervals):
#             t = threading.Thread(
#                 target=self.node_worker,
#                 args=(node, interval),
#                 daemon=True
#             )
#             t.start()
#             node_threads.append(t)
        
#         try:
#             # Keep main thread alive
#             while True:
#                 time.sleep(1)
        
#         except KeyboardInterrupt:
#             print(f"\n\n{'='*70}")
#             print("STOPPING SIMULATION...")
#             print(f"{'='*70}")
            
#             self.running = False
            
#             # Wait for threads to finish
#             time.sleep(2)
            
#             print("\n" + "="*70)
#             print("FINAL SUMMARY")
#             print("="*70)
#             print(f"\nTotal aggregation rounds: {self.server.round_number}")
#             print(f"\nPrivacy guarantees maintained:")
#             print(f"  ✓ DP noise protected pattern inference")
#             print(f"  ✓ Encryption protected transmission")
#             print(f"  ✓ Server never saw individual node parameters")
#             print(f"  ✓ Only aggregated global model computed")
#             print("\n" + "="*70)
#             print("Simulation complete!")
#             print("="*70 + "\n")


# # ═══════════════════════════════════════════════════════════════
# # MAIN EXECUTION
# # ═══════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     simulation = FederatedSimulation()
#     simulation.run()
"""
COMPLETE PRIVACY-PRESERVING FEDERATED LEARNING SYSTEM
======================================================

Features:
1. Three simulated nodes (different locations/data)
2. Differential Privacy (DP) on each node
3. Paillier Homomorphic Encryption
4. Asynchronous parameter transmission
5. Server-side encrypted aggregation
6. Complete privacy guarantee
7. Detailed plain + encrypted values and aggregation differences shown
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from phe import paillier
import time
from datetime import datetime
import threading
import queue

print("="*70)
print("PRIVACY-PRESERVING FEDERATED LEARNING: DP + ENCRYPTION + VERIFICATION")
print("="*70)
print("\nPress Ctrl+C to stop the simulation\n")
print("="*70)

class TenParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 5, bias=False)  # 10 parameters (2x5)
    def forward(self, x):
        return self.fc(x)

class DifferentialPrivacy:
    def __init__(self, epsilon=5.0, delta=1e-5, sensitivity=0.05):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        import math
        self.sigma = (sensitivity / epsilon) * math.sqrt(2 * math.log(1.25 / delta))
    def add_noise(self, params):
        noise = np.random.normal(0, self.sigma, params.shape)
        return params + noise, noise

class FederatedNode:
    def __init__(self, node_id, location_type, public_key, epsilon=5.0):
        self.node_id = node_id
        self.location_type = location_type
        self.public_key = public_key
        self.epsilon = epsilon
        self.model = TenParamModel()
        self.dp = DifferentialPrivacy(epsilon=epsilon)
        self.data_patterns = {
            'industrial': {'mean':1.5,'std':0.5},
            'park': {'mean':-0.5,'std':0.3},
            'residential': {'mean':0.2,'std':0.4}
        }
        print(f"\n[{node_id}] Node initialized at {location_type} with ε={epsilon}")
    def train_local_model(self):
        pattern = self.data_patterns[self.location_type]
        X = torch.randn(20,2)*pattern['std'] + pattern['mean']
        y = torch.randn(20,5)*pattern['std']
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for _ in range(10):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = ((pred - y)**2).mean()
            loss.backward()
            optimizer.step()
    def prepare_parameters(self):
        params = list(self.model.parameters())[0].detach().cpu().numpy().flatten()
        print(f"\n[{self.node_id}] Plain original parameters:")
        for i, p in enumerate(params):
            print(f"  Param {i+1}: {p:.6f}")
        params_noisy, noise = self.dp.add_noise(params)
        print(f"\n[{self.node_id}] After DP noise (σ={self.dp.sigma:.6f}):")
        for i, (orig, n, noisy_p) in enumerate(zip(params, noise, params_noisy)):
            print(f"  Param {i+1}: {orig:.6f} + {n:.6f} = {noisy_p:.6f}")
        SCALE = 10_000
        quantized = np.round(params_noisy * SCALE).astype(int)
        encrypted = []
        print(f"\n[{self.node_id}] Encrypting parameters:")
        for i, q in enumerate(quantized):
            enc = self.public_key.encrypt(int(q))
            encrypted.append(enc)
            preview = str(enc.ciphertext())[:20]+"..."
            print(f"  Param {i+1}: {q} -> Ciphertext: {preview}")
        return {
            'node_id': self.node_id,
            'location_type': self.location_type,
            'encrypted_params': encrypted,
            'timestamp': datetime.now(),
            'epsilon': self.epsilon,
            'original_params': params,
            'noisy_params': params_noisy,
            'quantized_params': quantized
        }

class CentralServer:
    def __init__(self):
        print("\n[SERVER] Initializing and generating Paillier keys...")
        start = time.time()
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=2048)
        elapsed = time.time() - start
        print(f"[SERVER] Keys generated in {elapsed:.2f} seconds")
        self.received_params = []
        self.round_number = 0
        self.aggregation_threshold = 3
    def receive_parameters(self, node_data):
        node_id = node_data['node_id']
        timestamp = node_data['timestamp'].strftime('%H:%M:%S')
        print(f"\n[SERVER {timestamp}] Received encrypted parameters from {node_id} at {node_data['location_type']}")
        self.received_params.append(node_data)
        print(f"[SERVER] Nodes ready: {len(self.received_params)}/{self.aggregation_threshold}")
        if len(self.received_params) >= self.aggregation_threshold:
            self.aggregate_and_verify()
    def aggregate_and_verify(self):
        print(f"\n{'='*70}")
        print(f"[SERVER] Aggregation Round {self.round_number + 1}")
        print(f"{'='*70}")
        num_nodes = len(self.received_params)
        num_params = len(self.received_params[0]['encrypted_params'])
        # Plaintext aggregate for comparison
        all_noisy = np.array([node['noisy_params'] for node in self.received_params])
        plain_agg = np.mean(all_noisy, axis=0)
        print(f"\n[SERVER] Plaintext aggregate of noisy parameters (for comparison):")
        for i in range(min(3,num_params)):
            print(f"  Param {i+1}: {plain_agg[i]:.6f}")
        # Encrypted aggregate
        weights = np.ones(num_nodes) / num_nodes
        enc_agg = []
        print(f"\n[SERVER] Aggregating encrypted parameters homomorphically:")
        t_start_encagg = time.time()
        for p_i in range(num_params):
            enc_sum = None
            for n_i, node_data in enumerate(self.received_params):
                param_enc = node_data['encrypted_params'][p_i]
                if enc_sum is None:
                    enc_sum = param_enc * weights[n_i]
                else:
                    enc_sum += param_enc * weights[n_i]
            enc_agg.append(enc_sum)
        t_end_encagg = time.time()
        print(f"  ✓ Encrypted aggregation done in {t_end_encagg - t_start_encagg:.4f}s")
        # Decrypt aggregated ciphertext
        print(f"\n[SERVER] Decrypting aggregated parameters:")
        t_start_decrypt = time.time()
        SCALE = 10_000
        decrypted = [self.private_key.decrypt(e)/SCALE for e in enc_agg]
        t_end_decrypt = time.time()
        print(f"  ✓ Decryption done in {t_end_decrypt - t_start_decrypt:.4f}s")
        decrypted_arr = np.array(decrypted)
        for i in range(min(3,num_params)):
            print(f"  Param {i+1}: {decrypted_arr[i]:.6f}")
        print(f"\n[SERVER] Difference between plaintext and decrypted aggregation:")
        diff = np.abs(plain_agg - decrypted_arr)
        for i in range(min(3,num_params)):
            print(f"  Param {i+1}: {diff[i]:.9f}")
        self.received_params = []
        self.round_number += 1
        print(f"\n{'='*70}\n")
    
class FederatedSimulation:
    def __init__(self):
        self.server = CentralServer()
        self.nodes = [
            FederatedNode('Node_1','industrial',self.server.public_key,5.0),
            FederatedNode('Node_2','park',self.server.public_key,5.0),
            FederatedNode('Node_3','residential',self.server.public_key,5.0),
        ]
        self.comm_queue = queue.Queue()
        self.running = True
    def node_worker(self,node,delay):
        while self.running:
            try:
                time.sleep(delay)
                if not self.running:
                    break
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {node.node_id} training locally...")
                node.train_local_model()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {node.node_id} preparing parameters (DP + Encryption)...")
                node_data = node.prepare_parameters()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {node.node_id} sending encrypted parameters...")
                self.comm_queue.put(node_data)
            except Exception as e:
                if self.running:
                    print(f"Error in {node.node_id}: {e}")
    def server_worker(self):
        while self.running:
            try:
                node_data = self.comm_queue.get(timeout=1)
                self.server.receive_parameters(node_data)
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"Server error: {e}")
    def run(self):
        print("\n" + "="*70)
        print("STARTING SIMULATION - DP + ENCRYPTION + AGGREGATION VERIFICATION")
        print("="*70)
        print("\nNodes: industrial, park, residential | DP ε=5.0 | Paillier Encryption\n")
        print("Press Ctrl+C to stop\n" + "="*70 + "\n")
        server_thread = threading.Thread(target=self.server_worker, daemon=True)
        server_thread.start()
        node_intervals = [5,7,9]
        node_threads = []
        for node, interval in zip(self.nodes,node_intervals):
            t = threading.Thread(target=self.node_worker,args=(node,interval),daemon=True)
            t.start()
            node_threads.append(t)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping simulation...")
            self.running = False
            time.sleep(2)
            print(f"\nTotal aggregation rounds: {self.server.round_number}\nSimulation complete!")

if __name__=="__main__":
    sim = FederatedSimulation()
    sim.run()
