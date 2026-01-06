# =========================================================
# STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO
# (Matplotlib-free, works on Streamlit Cloud)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

st.title("🚦 Traffic Signal Optimization using PSO")
st.write("""
This app finds the best green times for a four-way intersection
using Particle Swarm Optimization (PSO) based on traffic data.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) < 4:
        st.error("Dataset must contain at least 4 numeric traffic columns.")
    else:
        traffic_flows = df[numeric_cols[:4]].mean().to_numpy()
        st.subheader("Average Traffic Flows (veh/hr)")
        st.table(pd.DataFrame({
            'Direction': ['North', 'South', 'East', 'West'],
            'Average Flow (veh/hr)': traffic_flows.round(2)
        }))

        # Traffic delay function
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

        # PSO parameters
        st.subheader("🚀 Running PSO Optimization...")
        num_particles = 50
        num_iterations = 100
        dimensions = 4
        w, c1, c2 = 0.5, 1.8, 1.8

        pos = np.random.uniform(10, 50, (num_particles, dimensions))
        vel = np.random.uniform(-5, 5, (num_particles, dimensions))
        pbest = pos.copy()
        pbest_vals = np.array([compute_delay(p) for p in pos])
        gbest_idx = np.argmin(pbest_vals)
        gbest = pbest[gbest_idx].copy()
        gbest_val = pbest_vals[gbest_idx]

        convergence_curve = []

        for it in range(num_iterations):
            r1, r2 = np.random.rand(num_particles, dimensions), np.random.rand(num_particles, dimensions)
            vel = w*vel + c1*r1*(pbest-pos) + c2*r2*(gbest-pos)
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
            if (it+1) % 10 == 0 or it==0:
                st.write(f"Iteration {it+1:03}: Best Delay = {gbest_val:.6f}")

        # Display results
        st.subheader("✅ Best Traffic Light Timing")
        st.table(pd.DataFrame({
            'Direction': ['North','South','East','West'],
            'Green Time (sec)': [round(g,2) for g in gbest]
        }))
        st.write(f"Total Delay Score: {round(gbest_val,6)}")
        st.write(f"Sum of Green Times: {round(sum(gbest),2)} sec")

        # Plot convergence using Streamlit-native line chart
        st.subheader("📈 PSO Convergence Curve")
        st.line_chart(convergence_curve)
