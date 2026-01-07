# =========================================================
# SIMPLE STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt

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
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    required_cols = ["waiting_time", "average_speed"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    waiting_time = df["waiting_time"].to_numpy()
    average_speed = df["average_speed"].to_numpy()
    directions = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]

    # =========================================================
    # 4. TRAFFIC DELAY FUNCTION (Objective)
    # =========================================================
    def compute_delay(green_times):
        prop = green_times / green_times.sum()  # fraction of cycle per phase
        delay = np.sum(waiting_time * prop) + np.sum((1 / (average_speed + 1e-3)) * prop)
        return delay

    # =========================================================
    # 5. RUN PSO
    # =========================================================
    if st.button("Run PSO Optimization"):

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
            for it in range(num_iterations):
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
                st.write(f"🚦 {directions[i-1]}: **{round(g, 2)} sec**")
            st.write(f"⏱ Execution Time: **{exec_time:.3f} sec**")
            st.write(f"📉 Total Delay: **{round(gbest_val, 6)}**")

        with col2:
            st.subheader("📉 PSO Convergence Curve")
            # Altair chart with zoomed Y-axis
            conv_df = pd.DataFrame({
                "Iteration": list(range(1, len(convergence)+1)),
                "Delay Metric": convergence
            })
            chart = alt.Chart(conv_df).mark_line(point=True).encode(
                x=alt.X("Iteration", scale=alt.Scale(domain=[0, len(convergence)])),
                y=alt.Y("Delay Metric", scale=alt.Scale(domain=[min(convergence)*0.99, max(convergence)*1.01]))
            ).properties(
                width=600,
                height=400,
                title="PSO Convergence"
            )
            st.altair_chart(chart, use_container_width=True)

        # =========================================================
        # 7. PERFORMANCE ANALYSIS
        # =========================================================
        st.divider()
        st.header("Performance Analysis")
        st.markdown("""
        - **Convergence Rate:** Rapid improvement during early iterations  
        - **Optimization Quality:** Able to reduce total delay  
        - **Computational Efficiency:** Low execution time  
        - **Observations:** Cooperative particle behavior improves solution quality
        """)

        st.header("Conclusion")
        st.markdown("""
        This Streamlit-based PSO system optimizes traffic signal timings based on waiting time and average speed.
        The Altair convergence chart makes small improvements visible, allowing better analysis of algorithm performance.
        """)
