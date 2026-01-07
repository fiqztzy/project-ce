# =========================================================
# SIMPLE STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO
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
st.write("Simple PSO optimization using waiting time and average speed for 4-phase traffic signals.")

# =========================================================
# 2. SIDEBAR – SIMPLE PSO PARAMETERS
# =========================================================
st.sidebar.header("PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_iterations = st.sidebar.slider("Iterations", 20, 200, 50)
inertia_weight = st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.7)
velocity_limit = st.sidebar.slider("Velocity Limit", 1, 20, 10)

# Fixed coefficients
c1, c2 = 2.0, 2.0
CYCLE_TIME = 120  # total cycle time in seconds

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
    required_cols = ['waiting_time', 'average_speed']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    # =========================================================
    # 5. PREPARE DATA
    # =========================================================
    waiting_time = df['waiting_time']
    average_speed = df['average_speed']

    # Use mean values for simplicity in PSO
    waiting_time_mean = waiting_time.mean()
    average_speed_mean = average_speed.mean()

    directions = ["North", "South", "East", "West"]

    # =========================================================
    # 6. TRAFFIC DELAY FUNCTION
    # =========================================================
    def compute_delay(green_times):
        if np.sum(green_times) > CYCLE_TIME:
            return 1e9  # invalid

        prop = green_times / np.sum(green_times)
        # Delay metric combines waiting time and inverse of average speed
        delay = np.sum(prop * waiting_time_mean) + np.sum(prop * (1 / (average_speed_mean + 1e-3)))
        return delay

    # =========================================================
    # 7. RUN PSO
    # =========================================================
    if st.button("Run PSO Optimization"):

        st.subheader("Running PSO Optimization...")

        dimensions = 4
        pos = np.random.uniform(5, 50, (num_particles, dimensions))
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
        # 8. RESULTS
        # =========================================================
        st.subheader("Optimization Results")

        col1, col2 = st.columns(2)

        with col1:
            st.success("✅ Best Traffic Light Timing Found")
            for i, g in enumerate(gbest, 1):
                st.write(f"🚦 Phase {i} ({directions[i-1]}): **{round(g, 2)} sec**")

            st.write(f"Execution Time: **{exec_time:.3f} sec**")
            st.write(f"Best Delay Metric: **{round(gbest_val, 6)}**")
            st.write(f"Total Green Time: **{round(np.sum(gbest), 2)} sec**")

        with col2:
            st.subheader("PSO Convergence Curve")
            st.line_chart(convergence)

    # =========================================================
    # 9. PERFORMANCE ANALYSIS
    # =========================================================
    st.divider()
    st.header("Performance Analysis")

    st.subheader("Key Metrics Evaluated:")
    st.markdown("""
    - **Convergence Rate:** How quickly the PSO finds a good solution  
    - **Optimization Quality:** How low the delay metric is  
    - **Computational Efficiency:** Execution time for PSO
    """)

    st.subheader("Observations:")
    st.markdown("""
    - Rapid improvement in early iterations  
    - Convergence stabilizes after sufficient iterations  
    - Balanced green time allocation reduces overall delay
    """)

    # =========================================================
    # 10. CONCLUSION
    # =========================================================
    st.header("Conclusion")
    st.markdown("""
    This Streamlit-based PSO system optimizes 4-phase traffic signal timings  
    using waiting time and average speed. Users can adjust PSO parameters  
    interactively and observe the convergence behavior.
    """)
