import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD DATASET
# =========================================================
dataset_path = "traffic_dataset.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset '{dataset_path}' not found!")

df = pd.read_csv(dataset_path)

# Use first 4 numeric columns as traffic flows
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) < 4:
    raise ValueError("Dataset must contain at least 4 numeric traffic columns.")

traffic_flows = df[numeric_cols[:4]].mean().to_numpy()

print("Average Traffic Flows (veh/hr):")
for dir_name, flow in zip(['North', 'South', 'East', 'West'], traffic_flows):
    print(f"{dir_name}: {flow:.2f}")

# =========================================================
# 2. TRAFFIC DELAY FUNCTION
# =========================================================
CYCLE_TIME = 120  # seconds
SAT_FLOW = 1800   # veh/hr

def compute_delay(green_times):
    """Compute total intersection delay given green times [N, S, E, W]"""
    if np.sum(green_times) >= CYCLE_TIME:
        return 1e9  # invalid solution

    flow_per_sec = traffic_flows / 3600
    capacity = (green_times / CYCLE_TIME) * SAT_FLOW / 3600

    if np.any(capacity <= 0):
        return 1e9

    delays = flow_per_sec / capacity
    return np.sum(delays)

# =========================================================
# 3. PARTICLE SWARM OPTIMIZATION (PSO)
# =========================================================
num_particles = 50
num_iterations = 100
dimensions = 4  # N, S, E, W

# PSO parameters
w = 0.5
c1 = 1.8
c2 = 1.8

# Initialize particles and velocities
pos = np.random.uniform(10, 50, (num_particles, dimensions))
vel = np.random.uniform(-5, 5, (num_particles, dimensions))

pbest = pos.copy()
pbest_vals = np.array([compute_delay(p) for p in pos])

gbest_idx = np.argmin(pbest_vals)
gbest = pbest[gbest_idx].copy()
gbest_val = pbest_vals[gbest_idx]

# Track convergence
convergence_curve = []

print("\n🚦 Running PSO Optimization...\n")

for it in range(num_iterations):
    r1, r2 = np.random.rand(num_particles, dimensions), np.random.rand(num_particles, dimensions)

    # Update velocity and position
    vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
    pos = np.clip(pos + vel, 5, 60)

    # Evaluate new positions
    values = np.array([compute_delay(p) for p in pos])

    # Update personal bests
    improved = values < pbest_vals
    pbest[improved] = pos[improved]
    pbest_vals[improved] = values[improved]

    # Update global best
    min_idx = np.argmin(pbest_vals)
    if pbest_vals[min_idx] < gbest_val:
        gbest_val = pbest_vals[min_idx]
        gbest = pbest[min_idx].copy()

    convergence_curve.append(gbest_val)  # Save best value for plotting

    if (it + 1) % 10 == 0 or it == 0:
        print(f"Iteration {it+1:03}: Best Delay = {gbest_val:.6f}")

# =========================================================
# 4. DISPLAY BEST TRAFFIC LIGHT TIMING
# =========================================================
traffic_directions = ['North', 'South', 'East', 'West']

best_timing_df = pd.DataFrame({
    'Direction': traffic_directions,
    'Green Time (sec)': [round(g, 2) for g in gbest]
})

print("\n=====================================")
print("     BEST TRAFFIC LIGHT TIMING")
print("=====================================")
print(best_timing_df.to_string(index=False))
print(f"\nTotal Delay Score: {round(gbest_val, 6)}")
print(f"Sum of Green Times: {round(sum(gbest), 2)} sec")
print("=====================================\n")

# =========================================================
# 5. PLOT CONVERGENCE GRAPH
# =========================================================
plt.figure(figsize=(8,5))
plt.plot(convergence_curve, color='blue', linewidth=2)
plt.title("PSO Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Delay")
plt.grid(True)
plt.show()
