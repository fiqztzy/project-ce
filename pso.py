# =========================================================
# STREAMLIT PSO WITH WAITING TIME AND AVERAGE SPEED
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("🚦 PSO Traffic Signal Optimization with Waiting Time & Speed")

# -------------------------
# Upload dataset
# -------------------------
uploaded_file = st.file_uploader("Upload CSV with columns: waiting_time, average_speed", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # check required columns
    required_cols = ["waiting_time", "average_speed"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    # PSO Parameters
    num_particles = 30
    num_iterations = 50
    inertia_weight = 0.7
    c1, c2 = 1.5, 1.5
    velocity_limit = 10
    dimensions = 4  # N,S,E,W

    # -------------------------
    # Define objective function
    # -------------------------
    waiting_times = df["waiting_time"].to_numpy()
    avg_speeds = df["average_speed"].to_numpy()

    def compute_objective(green_times):
        """
        Objective: minimize waiting_time, optionally maximize average speed
        We'll combine as: score = mean_waiting_time / mean_speed
        """
        if np.sum(green_times) >= 120:  # max cycle
            return 1e9
        score = np.mean(waiting_times) / (np.mean(avg_speeds)+1e-3)  # avoid divide by zero
        return score

    # -------------------------
    # Initialize particles
    # -------------------------
    pos = np.random.uniform(10, 50, (num_particles, dimensions))
    vel = np.random.uniform(-velocity_limit, velocity_limit, (num_particles, dimensions))
    pbest = pos.copy()
    pbest_vals = np.array([compute_objective(p) for p in pos])
    gbest_idx = np.argmin(pbest_vals)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_vals[gbest_idx]

    convergence = []
    start_time = time.time()

    # -------------------------
    # PSO Loop
    # -------------------------
    for _ in range(num_iterations):
        r1 = np.random.rand(num_particles, dimensions)
        r2 = np.random.rand(num_particles, dimensions)

        vel = inertia_weight * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -velocity_limit, velocity_limit)
        pos = np.clip(pos + vel, 5, 60)

        values = np.array([compute_objective(p) for p in pos])

        improved = values < pbest_vals
        pbest[improved] = pos[improved]
        pbest_vals[improved] = values[improved]

        min_idx = np.argmin(pbest_vals)
        if pbest_vals[min_idx] < gbest_val:
            gbest_val = pbest_vals[min_idx]
            gbest = pbest[min_idx].copy()

        convergence.append(gbest_val)

    exec_time = time.time() - start_time

    # -------------------------
    # Display Results
    # -------------------------
    st.subheader("✅ Best Traffic Signal Timing")
    directions = ["North", "South", "East", "West"]
    for d, g in zip(directions, gbest):
        st.write(f"{d} Green: {g:.2f} sec")

    st.write(f"Total Objective Score: {gbest_val:.6f}")
    st.write(f"Execution Time: {exec_time:.3f} sec")

    st.subheader("📈 Convergence Curve")
    st.line_chart(convergence)
