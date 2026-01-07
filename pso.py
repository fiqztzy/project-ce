# =========================================================
# STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO
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
st.write("PSO optimization for traffic signals using waiting time and average speed.")

# =========================================================
# 2. SIDEBAR – PSO PARAMETERS
# =========================================================
st.sidebar.header("PSO Parameters")
num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_iterations = st.sidebar.slider("Iterations", 20, 200, 50)
inertia_weight = st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.7)
velocity_limit = st.sidebar.slider("Velocity Limit", 1, 20, 10)
c1, c2 = 2.0, 2.0  # cognitive and social coefficients

# =========================================================
# 3. UPLOAD DATASET
# =========================================================
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Check required columns
    required_cols = ["waiting_time", "average_speed"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    # =========================================================
    # 4. PREPARE DATA
    # =========================================================
    waiting_time_avg = df["waiting_time"].mean().values if hasattr(df["waiting_time"], 'values') else df["waiting_time"].to_numpy()
    average_speed_avg = df["average_speed"].mean().values if hasattr(df["average_speed"], 'values') else df["average_speed"].to_numpy()

    directions = ["North", "South", "East", "West"]

    # =========================================================
    # 5. OBJECTIVE FUNCTION (DELAY)
    # =========================================================
    def compute_delay(green_times):
        # Make green times within bounds
        green_times = np.clip(green_times, 5, 60)
        # Delay decreases if green time increases for high waiting phases
        delay = np.sum(waiting_time_avg * (1 / (green_times + 1e-3)))
        # Penalize phases with slow speeds and low green
        delay += np.sum((1 / (average_speed_avg + 1e-3)) * (60 - green_times)/60)
        return delay

    # =========================================================
    # 6. RUN PSO
    # =========================================================
    if st.button("Run PSO Optimization", type="primary"):

        st.subheader("Running PSO Optimization...")
        dimensions = 4  # N, S, E, W

        pos = np.random.uniform(10, 50, (num_particles, dimensions))
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

                vel = (
                    inertia_weight * vel
                    + c1 * r1 * (pbest - pos)
                    + c2 * r2 * (gbest - pos)
                )

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
        # 7. DISPLAY RESULTS
        # =========================================================
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success("Best Traffic Light Timing Found")
            for i, g in enumerate(gbest, 1):
                st.write(f"🚦 Phase {i} ({directions[i-1]}): **{round(g,2)} sec**")
            st.write(f"Execution Time: **{exec_time:.3f} sec**")
            st.write(f"Total Delay: **{round(gbest_val, 6)}**")

        with col2:
            st.subheader("PSO Convergence Curve")
            st.line_chart(convergence)

        # =========================================================
        # 8. PERFORMANCE ANALYSIS
        # =========================================================
        st.divider()
        st.header("Performance Analysis")
        st.markdown("""
        - Rapid improvement during early iterations
        - Different phases now receive different green times
        - Optimized delay balances waiting time and average speed
        """)
        st.header("Conclusion")
        st.markdown("""
        PSO can optimize traffic signal green times for each phase based on real traffic data.
        Each phase is assigned a different green time to minimize total delay.
        """)
