import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD DATASET
# =========================================================
dataset_path = "traffic_dataset.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset '{dataset_path}' not found!")

df = pd.read_csv(dataset_path)

# Select numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

if len(numeric_cols) < 4:
    raise ValueError("Dataset must contain at least four numeric traffic columns.")

# Use first four numeric columns
selected_cols = numeric_cols[:4]
traffic_flows = df[selected_cols].mean().to_numpy()

print("Average Traffic Flows (veh/hr):")
for col, flow in zip(selected_cols, traffic_flows):
    print(f"{col}: {flow:.2f}")

# =========================================================
# 2. TRAFFIC DELAY FUNCTION
# =========================================================
CYCLE_TIME = 120  # seconds
SAT_FLOW = 1800   # veh/hr

def compute_delay(green_times):
    if np.sum(green_times) >= CYCLE_TIME:
        return 1e9

    flow_per_sec = traffic_flows / 3600
    capacity = (green_times / CYCLE_TIME) * SAT_FLOW / 3600

    if np.any(capacity <= 0):
        return 1e9

    return np.sum(flow_per_sec / capacity)

# =========================================================
# 3. PARTICLE SWARM OPTIMIZATION (PSO)
# =========================================================
num_particles = 50
num_iterations = 100
dimensions = 4

w = 0.5
c1 = 1.8
c2 = 1.8

pos = np.random.uniform(10, 50, (num_particles, dimensions))
vel = np.random.uniform(-5, 5, (num_particles, dimensions))

pbest = pos.copy()
pbest_vals = np.array([compute_delay(p) for p in pos])

gbest_idx = np.argmin(pbest_vals)
gbest = pbest[gbest_idx].copy()
gbest_val = pbest_vals[gbest_idx]

convergence_curve = []

print("\nRunning PSO Optimization...\n")

for it in range(num_iterations):
    r1 = np.random.rand(num_particles, dimensions)
    r2 = np.random.rand(num_particles, dimensions)

    vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
    pos = np.clip(pos + vel, 5, 60)

    values = np.array([compute_delay(p) for p in pos])

    improved = values < pbest_vals
    pbest[improved] = pos[improved]
    pbest_vals[improved] = values[improved]

    min_idx = np.argmin(pbest_vals)
    if pbest_vals[min_idx] < gbest_val:
        gbest_val = pbest_vals[min_idx]
        gbest = pbest[min_idx].copy()

    convergence_curve.append(gbest_val)

    if (it + 1) % 10 == 0 or it == 0:
        print(f"Iteration {it+1:03}: Best Delay = {gbest_val:.6f}")

# =========================================================
# 4. DISPLAY OPTIMIZED SIGNAL PLAN (REPORT SAFE)
# =========================================================
results_df = pd.DataFrame({
    "Signal Phase": [f"Phase {i+1}" for i in range(dimensions)],
    "Green Time (sec)": np.round(gbest, 2)
})

print("\n=====================================")
print("     BEST TRAFFIC LIGHT TIMING")
print("=====================================")
print(results_df.to_string(index=False))
print(f"\nTotal Delay Score: {round(gbest_val, 6)}")
print(f"Sum of Green Times: {round(np.sum(gbest), 2)} sec")
print("=====================================\n")

# =========================================================
# 5. CONVERGENCE GRAPH
# =========================================================
plt.figure()
plt.plot(convergence_curve)
plt.xlabel("Iteration")
plt.ylabel("Best Delay")
plt.title("PSO Convergence Curve")
plt.grid(True)
plt.show()
