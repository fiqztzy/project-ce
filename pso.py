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
st.write("Simple PSO for traffic signal optimization using waiting time and average speed.")

# =========================================================
# 2. SIDEBAR – PSO PARAMETERS
# =========================================================
st.sidebar.header("PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_iterations = st.sidebar.slider("Iterations", 20, 200, 50)
inertia_weight = st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.7)
velocity_limit = st.sidebar.slider("Velocity Limit", 1, 20, 10)

# Fixed coefficients
c1, c2 = 2.0, 2.0

# =========================================================
# 3. UPLOAD DATASET
# =========================================================
uploaded_file = st.file_uploader("Upload CSV with columns: waiting_time, average_speed", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if not all(col in df.columns for col in ["waiting_time", "average_speed"]):
        st.error("Dataset must contain columns: ['waiting_time', 'average_speed']")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =========================================================
    # 4. EXTRACT 4 PHASE VALUES
    # =========================================================
    # Take first 4 rows as N, S, E, W
    waiting_time_avg = df["waiting_time"].iloc[:4].to_numpy()
    average_speed_avg = df["average_speed"].iloc[:4].to_numpy()

    directions = ["North", "South", "East", "West"]

    # =========================================================
    # 5. TRAFFIC DELAY FUNCTION
    # =========================================================
    def compute_delay(green_times):
        prop = green_times / green_times.sum()
        delay = np.sum(waiting_time_avg * prop) + np.sum((1 / (average_speed_avg + 1e-3)) * prop)
        return delay

    # =========================================================
    # 6. RUN PSO (BUTTON)
    # =========================================================
    if st.button("Run PSO Optimization", type="primary"):

        st.subheader("Running PSO Optimization...")

        dimensions = 4
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
        # 7. RESULTS
        # =========================================================
        st.subheader("Optimization Results")

        col1, col2 = st.columns(2)

        with col1:
            st.success("Best Traffic Light Timing Found")
            for i, g in enumerate(gbest, 1):
                st.write(f"🚦 Phase {i} ({directions[i-1]}): **{round(g,2)} sec**")

            st.write(f"Execution Time: **{exec_time:.3f} sec**")
            st.write(f"Total Delay: **{round(gbest_val,6)}**")
            st.write(f"Sum of Green Times: **{round(np.sum(gbest),2)} sec**")

        with col2:
            st.subheader("PSO Convergence")
            st.line_chart(convergence)

    # =========================================================
    # PERFORMANCE ANALYSIS
    # =========================================================
    st.divider()
    st.header("Performance Analysis")

    st.subheader("Key Metrics:")
    st.markdown("""
    - **Convergence Rate:** How fast the PSO stabilizes
    - **Optimization Quality:** Minimizing traffic delay
    - **Execution Efficiency:** Low computation time
    """)

    st.subheader("Observations:")
    st.markdown("""
    - Rapid improvement during early iterations
    - Stable convergence after sufficient iterations
    - Particle cooperation leads to better solutions
    """)

    # =========================================================
    # CONCLUSION
    # =========================================================
    st.header("Conclusion")
    st.markdown("""
    This Streamlit-based PSO system optimizes traffic signal green times
    to reduce waiting time and improve average speed. Users can interactively
    adjust parameters and observe convergence behavior.
    """)
