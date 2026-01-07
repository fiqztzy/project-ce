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
st.write("Simple and clean PSO parameter tuning for traffic signal optimization.")

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

# =========================================================
# 3. UPLOAD DATASET
# =========================================================
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =========================================================
    # 4. TRAFFIC FLOWS (INTERNAL ONLY)
    # =========================================================
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) < 4:
        st.error("Dataset must contain at least 4 numeric columns.")
        st.stop()

    traffic_flows = df[numeric_cols[:4]].mean().to_numpy()

    # =========================================================
    # 5. TRAFFIC DELAY FUNCTION
    # =========================================================
    CYCLE_TIME = 120
    SAT_FLOW = 1800

    def compute_delay(green_times):
        if np.sum(green_times) >= CYCLE_TIME:
            return 1e9

        flow_per_sec = traffic_flows / 3600
        capacity = (green_times / CYCLE_TIME) * SAT_FLOW / 3600

        if np.any(capacity <= 0):
            return 1e9

        return np.sum(flow_per_sec / capacity)

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
                st.write(f"🚦 Phase {i}: **{round(g)} sec**")

            st.write(f"Execution Time: **{exec_time:.3f} sec**")
            st.write(f"Total Delay: **{round(gbest_val, 6)}**")

        with col2:
            st.subheader("PSO Convergence")
            st.line_chart(convergence)

# =========================================================
# PERFORMANCE ANALYSIS
# =========================================================
st.divider()
st.header("Performance Analysis")

st.subheader("Key Metrics Evaluated:")
st.markdown("""
- **Convergence Rate:** Speed at which the PSO algorithm stabilizes  
- **Optimization Quality:** Ability to reduce traffic delay  
- **Computational Efficiency:** Execution time of the optimization process
""")

st.subheader("Observations:")
st.markdown("""
- Rapid improvement during early iterations  
- Stable convergence after sufficient iterations  
- Cooperative particle behavior improves solution quality
""")

# =========================================================
# CONCLUSION
# =========================================================
st.header("Conclusion")
st.markdown("""
This Streamlit-based system demonstrates how **Particle Swarm Optimization (PSO)** 
can be effectively applied to traffic signal optimization problems. The interactive 
dashboard allows users to adjust parameters, observe convergence behavior, and obtain 
optimized traffic signal timings with low computational cost.
""")
