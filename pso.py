# =========================================================
# STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import time

# =========================================================
# 1. STREAMLIT APP TITLE
# =========================================================
st.title("🚦 Traffic Signal Optimization using PSO")
st.write("""
This app uses Particle Swarm Optimization (PSO) to find the best green times for a four-way intersection
based on traffic data from North, South, East, and West directions.
""")

# =========================================================
# 2. UPLOAD CSV
# =========================================================
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =========================================================
    # 3. ANALYZE TRAFFIC DATA
    # =========================================================
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) < 4:
        st.error("Dataset must contain at least 4 numeric traffic columns.")
    else:
        traffic_flows = df[numeric_cols[:4]].mean().to_numpy()

        st.subheader("Average Traffic Flows (veh/hr)")
        flow_df = pd.DataFrame({
            'Direction': ['North', 'South', 'East', 'West'],
            'Average Flow (veh/hr)': traffic_flows.round(2)
        })
        st.table(flow_df)

        # =========================================================
        # 4. TRAFFIC DELAY FUNCTION
        # =========================================================
        CYCLE_TIME = 120  # seconds
        SAT_FLOW = 1800   # veh/hr

        def compute_delay(green_times):
            if np.sum(green_times) >= CYCLE_TIME:
                return 1e9
            flow_per_sec = traffic_flows / 3600
            capacity = (green_times / CYCLE_TIME) * SAT_FLOW / 3600
            if np.any(capacity <= 0):
                return 1e9
            delays = flow_per_sec / capacity
            return np.sum(delays)

        # =========================================================
        # 5. PARTICLE SWARM OPTIMIZATION (PSO)
        # =========================================================
        st.subheader("🚀 Running PSO Optimization...")
        num_particles = 50
        num_iterations = 100
        dimensions = 4  # N, S, E, W

        w, c1, c2 = 0.5, 1.8, 1.8

        pos = np.random.uniform(10, 50, (num_particles, dimensions))
        vel = np.random.uniform(-5, 5, (num_particles, dimensions))

        pbest = pos.copy()
        pbest_vals = np.array([compute_delay(p) for p in pos])

        gbest_idx = np.argmin(pbest_vals)
        gbest = pbest[gbest_idx].copy()
        gbest_val = pbest_vals[gbest_idx]

        convergence_curve = []

        start_time = time.time()  # Start timer

        for it in range(num_iterations):
            r1, r2 = np.random.rand(num_particles, dimensions), np.random.rand(num_particles, dimensions)

            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, 5, 60)

            values = np.array([compute_delay(p) for p in pos])

            improved = values < pbest_vals
            pbest[improved] = pos[improved]
            pbest_vals[improved] = values[improved]

            min_idx = np.argmin(pbest_vals)
            if pbest_vals[min_idx] < gbest_val:
                gbest_val = pbest_vals[min_idx]
                gbest = pbest[min_idx].copy()

            convergence_curve.append(gbest_val)

        exec_time = time.time() - start_time  # End timer

        # =========================================================
        # 6. DISPLAY RESULTS SIDE-BY-SIDE LIKE YOUR IMAGE
        # =========================================================
        st.subheader("📊 Optimization Results")
        col1, col2 = st.columns(2)

        # Left column: Best timing
        with col1:
            st.success("Best Traffic Light Timing Found")
            directions = ['North', 'South', 'East', 'West']
            for i, (dir_name, green_time) in enumerate(zip(directions, gbest), 1):
                st.write(f"🚦 Phase {i} ({dir_name}) Green Time: **{round(green_time)} seconds**")
            st.write(f"⏱ Execution Time: **{exec_time:.4f} seconds**")
            st.write(f"📝 Total Delay Score: **{round(gbest_val, 6)}**")
            st.write(f"🕒 Sum of Green Times: **{round(sum(gbest), 2)} sec**")

        # Right column: Convergence graph
        with col2:
            st.subheader("PSO Convergence Curve")
            st.line_chart(convergence_curve)
