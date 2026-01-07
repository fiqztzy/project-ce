# =========================================================
# STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO (USING WAITING TIME & AVG SPEED)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import time

# =========================================================
# 1. APP TITLE
# =========================================================
st.set_page_config(page_title="Traffic Signal Optimization (PSO)", layout="wide")

st.title("🚦 Traffic Signal Optimization using PSO")
st.write("PSO optimization using **waiting time** and **average speed** as objectives.")

# =========================================================
# 2. SIDEBAR – SIMPLE PSO PARAMETERS
# =========================================================
st.sidebar.header("PSO Parameters")
num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_iterations = st.sidebar.slider("Iterations", 20, 200, 50)
inertia_weight = st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.7)
velocity_limit = st.sidebar.slider("Velocity Limit", 1, 20, 10)
c1, c2 = 2.0, 2.0

# =========================================================
# 3. UPLOAD DATASET
# =========================================================
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =========================================================
    # 4. CHECK REQUIRED COLUMNS
    # =========================================================
    required_cols = ["waiting_time", "average_speed"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Dataset must contain column: {col}")
            st.stop()

    waiting_times = df["waiting_time"].to_numpy()
    avg_speeds = df["average_speed"].to_numpy()

    # =========================================================
    # 5. DELAY FUNCTION (OBJECTIVE)
    # =========================================================
    # Minimize waiting time and maximize average speed
    def compute_delay(green_times):
        # For simplicity, simulate weighted sum of waiting time and inverse speed
        # Each green_time affects a quarter of traffic (4 "phases")
        # Green_times: [phase1, phase2, phase3, phase4]
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        # compute weighted waiting time
        wt_score = np.sum(weights * waiting_times[:4])
        # compute weighted inverse speed (higher speed => lower delay)
        speed_score = np.sum(weights * (1 / (avg_speeds[:4] + 1e-6)))
        return wt_score + speed_score  # lower is better

    # =========================================================
    # 6. RUN PSO
    # =========================================================
    if st.button("Run PSO Optimization", type="primary"):
        st.subheader("Running PSO Optimization...")

        dimensions = 4  # 4 phases
        pos = np.random.uniform(5, 60, (num_particles, dimensions))
        vel = np.random.uniform(-velocity_limit, velocity_limit, (num_particles, dimensions))

        pbest = pos.copy()
        pbest_vals = np.array([compute_delay(p) for p in pos])

        gbest_idx = np.argmin(pbest_vals)
        gbest = pbest[gbest_idx].copy()
        gbest_val = pbest_vals[gbest_idx]

        convergence = []
        start_time = time.time()

        with st.spinner("Optimizing traffic signal timings..."):
            for _ in range(num_iterations):
                r1 = np.random.rand(num_particles, dimensions)
                r2 = np.random.rand(num_particles, dimensions)

                vel = inertia_weight * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
                vel = np.clip(vel, -velocity_limit, velocity_limit)
                pos = np.clip(pos + vel, 5, 60)

                values = np.array([compute_delay(p) for p in pos])

                improved = values < pbest_vals
                pbest[improved] = pos[improved]
                pbest_vals[improved] = values[improved]

                min_idx = np.argmin(pbest_vals)
                if pbest_vals[min_idx] < gbest_val:
                    gbest_val = pbest_vals[min_idx]
                    gbest = pbest[min_idx].copy()

                convergence.append(gbest_val)

        exec_time = time.time() - start_time

        # =========================================================
        # 7. RESULTS
        # =========================================================
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success("Best Traffic Light Timing Found")
            for i, g in enumerate(gbest, 1):
                st.write(f"🚦 Phase {i}: **{round(g, 2)} sec**")
            st.write(f"Execution Time: **{exec_time:.3f} sec**")
            st.write(f"Best Objective (Delay) Value: **{round(gbest_val, 6)}**")

        with col2:
            st.subheader("PSO Convergence")
            st.line_chart(convergence)
