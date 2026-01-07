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
st.write("Optimizing traffic signal timings using PSO with waiting time & average speed.")

# =========================================================
# 2. SIDEBAR – PSO PARAMETERS
# =========================================================
st.sidebar.header("PSO Parameters")
num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_iterations = st.sidebar.slider("Iterations", 20, 200, 50)
inertia_weight = st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.7)
velocity_limit = st.sidebar.slider("Velocity Limit", 1, 20, 10)
c1, c2 = 2.0, 2.0  # cognitive & social coefficients

# =========================================================
# 3. UPLOAD DATASET
# =========================================================
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check required columns
    required_cols = ['waiting_time', 'average_speed']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    waiting_time = df['waiting_time'].to_numpy()
    average_speed = df['average_speed'].to_numpy()
    directions = ["North", "South", "East", "West"]

    # =========================================================
    # 4. TRAFFIC DELAY FUNCTION USING WAITING TIME & AVG SPEED
    # =========================================================
    def compute_delay(green_times):
        # proportional allocation of green times
        prop = green_times / green_times.sum()
        delay = np.sum(waiting_time * prop) + np.sum((1 / (average_speed + 1e-3)) * prop)
        return delay

    # =========================================================
    # 5. RUN PSO
    # =========================================================
    if st.button("Run PSO Optimization", type="primary"):
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
        # 6. DISPLAY RESULTS
        # =========================================================
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success("✅ Best Traffic Light Timing Found")
            for i, g in enumerate(gbest, 1):
                st.write(f"🚦 Phase {i}: **{round(g, 2)} sec**")
            st.write(f"Execution Time: **{exec_time:.3f} sec**")
            st.write(f"Total Delay: **{round(gbest_val, 6)}**")

        with col2:
            st.subheader("PSO Convergence (Scaled)")
            # scale for visibility
            scaled_convergence = np.array(convergence) / 1000  # adjust as needed
            st.line_chart(scaled_convergence)

        # =========================================================
        # 7. PERFORMANCE ANALYSIS
        # =========================================================
        st.divider()
        st.header("Performance Analysis")
        st.subheader("Key Metrics Evaluated:")
        st.markdown("""
        - **Convergence Rate:** How fast the PSO stabilizes  
        - **Optimization Quality:** Ability to minimize total delay  
        - **Computational Efficiency:** Execution time
        """)

        st.subheader("Observations:")
        st.markdown("""
        - Rapid improvement during early iterations  
        - Stable convergence after sufficient iterations  
        - Cooperative particle behavior enhances solution quality
        """)

        st.header("Conclusion")
        st.markdown("""
        This Streamlit-based PSO system demonstrates traffic signal optimization using
        waiting time and average speed. The dashboard allows parameter tuning, observes
        convergence, and finds optimal green times with visible improvement in the graph.
        """)
