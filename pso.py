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
st.write("Optimize traffic signal timings based on waiting time and average speed.")

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
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Cek kolum
    required_cols = ['waiting_time', 'average_speed']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    waiting_time = df['waiting_time'].to_numpy()
    average_speed = df['average_speed'].to_numpy()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =========================================================
    # 4. TRAFFIC DELAY FUNCTION
    # =========================================================
    def compute_delay(green_times):
        """
        Compute a simplified delay metric combining waiting time and average speed.
        green_times: array of 4 phases
        """
        avg_wait = np.mean(waiting_time)
        avg_speed_val = np.mean(average_speed)
        # formula: weighted sum of green times with avg waiting time and inverse speed
        delay = avg_wait * green_times.sum() + (1 / (avg_speed_val + 1e-3)) * green_times.sum()
        return delay

    # =========================================================
    # 5. RUN PSO
    # =========================================================
    if st.button("Run PSO Optimization"):

        st.subheader("Running PSO Optimization...")
        dimensions = 4  # 4 phases

        # Initialize particles and velocities
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
        # 6. RESULTS
        # =========================================================
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success("✅ Best Traffic Light Timing Found")
            for i, g in enumerate(gbest, 1):
                st.write(f"🚦 Phase {i}: **{round(g,2)} sec**")
            st.write(f"Execution Time: **{exec_time:.3f} sec**")
            st.write(f"Total Delay: **{round(gbest_val, 6)}**")

        with col2:
            st.subheader("PSO Convergence")
            st.line_chart(convergence)

        # =========================================================
        # 7. PERFORMANCE ANALYSIS
        # =========================================================
        st.divider()
        st.header("Performance Analysis")
        st.markdown("""
        - **Convergence Rate:** PSO rapidly improves best delay in early iterations.
        - **Optimization Quality:** Finds green time combinations minimizing delay.
        - **Computational Efficiency:** Low execution time even with moderate particle number.
        - **Interpretability:** Simple 4-phase green time allocation.
        """)
        st.header("Conclusion")
        st.markdown("""
        This Streamlit-based PSO system demonstrates traffic signal optimization using
        **waiting time** and **average speed**. Users can tune PSO parameters interactively,
        observe convergence, and obtain optimized green times for each phase.
        """)
