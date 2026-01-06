import os
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Traffic Signal Optimization using PSO", layout="centered")

st.title("🚦 Traffic Signal Optimization using PSO")

# =========================================================
# 1. LOAD DATASET
# =========================================================
dataset_path = "traffic_dataset.csv"

if not os.path.exists(dataset_path):
    st.error("Dataset 'traffic_dataset.csv' not found.")
    st.stop()

df = pd.read_csv(dataset_path)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

if len(numeric_cols) < 4:
    st.error("Dataset must contain at least four numeric traffic columns.")
    st.stop()

selected_cols = numeric_cols[:4]
traffic_flows = df[selected_cols].mean().to_numpy()

st.subheader("Average Traffic Flows (veh/hr)")
flow_df = pd.DataFrame({
    "Traffic Approach": [f"Phase {i+1}" for i in range(4)],
    "Average Flow (veh/hr)": np.round(traffic_flows, 2)
})
st.dataframe(flow_df, use_container_width=True)

# =========================================================
# 2. TRAFFIC DELAY FUNCTION
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

    return np.sum(flow_per_sec / capacity)

# =========================================================
# 3. PSO PARAMETERS (USER CONTROLS)
# =========================================================
st.sidebar.header("PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 20, 100, 50)
num_iterations = st.sidebar.slider("Number of Iterations", 50, 200, 100)

w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.5)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.5, 3.0, 1.8)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.5, 3.0, 1.8)

run_button = st.sidebar.button("▶ Run Optimization")

# =========================================================
# 4. PARTICLE SWARM OPTIMIZATION
# =========================================================
if run_button:

    dimensions = 4

    pos = np.random.uniform(10, 50, (num_particles, dimensions))
    vel = np.random.uniform(-5, 5, (num_particles, dimensions))

    pbest = pos.copy()
    pbest_vals = np.array([compute_delay(p) for p in pos])

    gbest_idx = np.argmin(pbest_vals)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_vals[gbest_idx]

    convergence_curve = []

    progress_bar = st.progress(0)

    for it in range(num_iterations):
        r1 = np.random.rand(num_particles, dimensions)
        r2 = np.random.rand(num_particles, dimensions)

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
        progress_bar.progress((it + 1) / num_iterations)

    st.success("Optimization Completed Successfully!")

    # =========================================================
    # 5. RESULTS DISPLAY
    # =========================================================
    st.subheader("Optimized Traffic Signal Timing")

    results_df = pd.DataFrame({
        "Signal Phase": [f"Phase {i+1}" for i in range(4)],
        "Optimized Green Time (sec)": np.round(gbest, 2)
    })

    st.dataframe(results_df, use_container_width=True)

    st.markdown(f"""
    **Total Delay Score:** `{round(gbest_val, 6)}`  
    **Sum of Green Times:** `{round(np.sum(gbest), 2)} sec`
    """)

    # =========================================================
    # 6. CONVERGENCE CURVE (STREAMLIT NATIVE)
    # =========================================================
    st.subheader("PSO Convergence Curve")

    convergence_df = pd.DataFrame({
        "Iteration": range(1, len(convergence_curve) + 1),
        "Best Delay": convergence_curve
    })

    st.line_chart(convergence_df.set_index("Iteration"))

