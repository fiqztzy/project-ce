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
st.set_page_config(page_title="Traffic Signal Optimization PSO", layout="wide")

st.title("🚦 Traffic Signal Optimization using PSO")
st.write("""
This application optimizes traffic signal green times for a four-way intersection
using **Particle Swarm Optimization (PSO)** with performance-tuned parameters.
""")

# =========================================================
# 2. SIDEBAR – PSO PERFORMANCE PARAMETERS
# =========================================================
st.sidebar.header("⚙️ PSO Performance Parameters")

num_particles = st.sidebar.slider("Number of Particles", 20, 100, 50, step=10)
num_iterations = st.sidebar.slider("Number of Iterations", 50, 300, 150, step=10)

w_max = st.sidebar.slider("Max Inertia Weight (w_max)", 0.5, 1.2, 0.9)
w_min = st.sidebar.slider("Min Inertia Weight (w_min)", 0.1, 0.5, 0.4)

c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 1.0, 3.0, 2.0)
c2 = st.sidebar.slider("Social Coefficient (c2)", 1.0, 3.0, 2.0)

early_stop = st.sidebar.slider("Early Stopping Patience", 10, 50, 20)

# =========================================================
# 3. UPLOAD DATASET
# =========================================================
uploaded_file = st.file_uploader("📂 Upload traffic_dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # =========================================================
    # 4. EXTRACT TRAFFIC FLOWS
    # =========================================================
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) < 4:
        st.error("❌ Dataset must contain at least 4 numeric traffic flow columns.")
        st.stop()

    traffic_flows = df[numeric_cols[:4]].mean().to_numpy()

    directions = ["North", "South", "East", "West"]

    st.subheader("🚘 Average Traffic Flows (veh/hr)")
    st.table(pd.DataFrame({
        "Direction": directions,
        "Average Flow (veh/hr)": traffic_flows.round(2)
    }))

    # =========================================================
    # 5. TRAFFIC DELAY FUNCTION
    # =========================================================
    CYCLE_TIME = 120   # seconds
    SAT_FLOW = 1800    # veh/hr

    def compute_delay(green_times):
        if np.sum(green_times) >= CYCLE_TIME:
            return 1e9

        flow_per_sec = traffic_flows / 3600
        capacity = (green_times / CYCLE_TIME) * SAT_FLOW / 3600

        if np.any(capacity <= 0):
            return 1e9

        delay = flow_per_sec / capacity
        return np.sum(delay)

    # =========================================================
    # 6. PARTICLE SWARM OPTIMIZATION
    # =========================================================
    st.subheader("🚀 Running PSO Optimization")

    dimensions = 4
    pos = np.random.uniform(10, 50, (num_particles, dimensions))
    vel = np.random.uniform(-5, 5, (num_particles, dimensions))

    pbest = pos.copy()
    pbest_vals = np.array([compute_delay(p) for p in pos])

    gbest_idx = np.argmin(pbest_vals)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_vals[gbest_idx]

    convergence_curve = []

    best_prev = np.inf
    stall_counter = 0

    start_time = time.time()

    for it in range(num_iterations):

        # Dynamic inertia weight
        w = w_max - (w_max - w_min) * (it / num_iterations)

        r1 = np.random.rand(num_particles, dimensions)
        r2 = np.random.rand(num_particles, dimensions)

        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -10, 10)

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

        # Early stopping
        if abs(best_prev - gbest_val) < 1e-6:
            stall_counter += 1
        else:
            stall_counter = 0

        best_prev = gbest_val

        if stall_counter >= early_stop:
            st.info(f"🛑 Early stopping at iteration {it+1}")
            break

    exec_time = time.time() - start_time

    # =========================================================
    # 7. RESULTS DISPLAY
    # =========================================================
    st.subheader("📈 Optimization Results")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ Best Traffic Light Timing Found")
        for i, (d, g) in enumerate(zip(directions, gbest), 1):
            st.write(f"🚦 Phase {i} ({d}): **{round(g, 2)} sec**")

        st.write(f"🕒 **Execution Time:** {exec_time:.4f} seconds")
        st.write(f"📉 **Total Delay Score:** {round(gbest_val, 6)}")
        st.write(f"⏱ **Total Green Time:** {round(np.sum(gbest), 2)} sec")

    with col2:
        st.subheader("📉 PSO Convergence Curve")
        st.line_chart(convergence_curve)
