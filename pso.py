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
st.write("Optimize traffic signal green times using PSO based on waiting time and average speed.")

# =========================================================
# 2. SIDEBAR PARAMETERS
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

    # Convert to numpy arrays
    waiting_time_arr = df["waiting_time"].to_numpy()
    average_speed_arr = df["average_speed"].to_numpy()

    # =========================================================
    # 4. PSO OBJECTIVE FUNCTION
    # =========================================================
    def compute_delay(green_times):
        """
        Compute a "delay" based on green times, waiting time, and average speed.
        Each phase gets proportionally weighted by green time.
        """
        green_times = np.clip(green_times, 5, 60)  # ensure reasonable bounds
        prop = green_times / green_times.sum()
        delay = np.sum(waiting_time_arr * prop) + np.sum((1 / (average_speed_arr + 1e-3)) * prop)
        return delay

    # =========================================================
    # 5. RUN PSO (BUTTON)
    # =========================================================
    if st.button("Run PSO Optimization", type="primary"):
        st.subheader("Running PSO Optimization...")

        dimensions = 4  # North, South, East, West
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
        # 6. DISPLAY RESULTS
        # =========================================================
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success("✅ Best Traffic Light Timing Found")
            phases = ["North", "South", "East", "West"]
            for i, g in enumerate(gbest):
                st.write(f"🚦 Phase {i+1} ({phases[i]}): **{round(g,2)} sec**")

            st.write(f"⏱ Execution Time: **{exec_time:.3f} sec**")
            st.write(f"📉 Total Delay Score: **{round(gbest_val,6)}**")

        with col2:
            st.subheader("PSO Convergence")
            # Use smaller y-axis to see variation
            st.line_chart(np.array(convergence))

        # =========================================================
        # 7. PERFORMANCE ANALYSIS
        # =========================================================
        st.divider()
        st.header("Performance Analysis")
        st.subheader("Key Metrics:")
        st.markdown("""
        - **Convergence Rate:** Shows how fast PSO stabilizes
        - **Optimization Quality:** How well traffic delay is minimized
        - **Computational Efficiency:** Execution time
        """)
        st.subheader("Observations:")
        st.markdown("""
        - PSO converges faster with higher inertia weight
        - Particle cooperation improves solution quality
        - Variations in green times reflect better delay minimization
        """)

        st.header("Conclusion")
        st.markdown("""
        This Streamlit-based PSO system demonstrates effective optimization
        of traffic signal timings using waiting time and average speed.  
        Users can interactively tune PSO parameters and observe optimized timings and convergence.
        """)
