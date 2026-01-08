# =========================================================
# STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt

# =========================================================
# 1. APP CONFIG
# =========================================================
st.set_page_config(page_title="Traffic Signal Optimization (PSO)", layout="wide")

st.title("🚦 Traffic Signal Optimization using PSO")
st.write("""
This application optimizes traffic signal green times for a four-phase intersection
using **Particle Swarm Optimization (PSO)** to **reduce congestion and improve
intersection performance**.
""")

# =========================================================
# 2. SIDEBAR PSO PARAMETERS
# =========================================================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_iterations = st.sidebar.slider("Number of Generations", 20, 300, 100)
w = st.sidebar.slider("Inertia Weight (w)", 0.3, 1.0, 0.7)
vmax = st.sidebar.slider("Velocity Limit", 1, 20, 10)

c1, c2 = 2.0, 2.0

# =========================================================
# 3. LOAD DATASET FROM GITHUB
# =========================================================
DATA_URL = "https://raw.githubusercontent.com/username/traffic-pso/main/traffic_dataset(2).csv"

try:
    df = pd.read_csv(DATA_URL)
    st.success("✅ Dataset loaded successfully from GitHub")
except Exception as e:
    st.error("❌ Failed to load dataset from GitHub")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================================================
# 4. DATA VALIDATION
# =========================================================
required_cols = ["waiting_time", "vehicle_count"]

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain: waiting_time, vehicle_count")
    st.stop()

# =========================================================
# 5. TRAFFIC STATISTICS
# =========================================================
avg_wait = df["waiting_time"].mean()
avg_count = df["vehicle_count"].mean()

st.subheader("📈 Traffic Statistics")
st.write(f"⏳ Average Waiting Time: **{avg_wait:.2f} sec**")
st.write(f"🚗 Average Vehicle Count: **{avg_count:.2f} vehicles**")

# =========================================================
# 6. FITNESS FUNCTION
# =========================================================
def compute_delay(green_times):
    green_times = np.clip(green_times, 5, 60)
    proportions = green_times / np.sum(green_times)

    phase_counts = df["vehicle_count"].values.reshape(-1, 4)
    delays = np.sum(phase_counts / proportions, axis=1)

    avg_delay = np.mean(delays)
    balance_penalty = 0.1 / (np.var(green_times) + 1e-6)

    return avg_delay + balance_penalty

# =========================================================
# 7. RUN PSO
# =========================================================
if st.button("🚀 Run PSO Optimization", type="primary"):

    start_time = time.time()
    dimensions = 4

    pos = np.random.uniform(10, 50, (num_particles, dimensions))
    vel = np.random.uniform(-vmax, vmax, (num_particles, dimensions))

    pbest = pos.copy()
    pbest_val = np.array([compute_delay(p) for p in pos])

    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    convergence = []

    with st.spinner("Optimizing traffic signal timings..."):
        for gen in range(num_iterations):

            r1, r2 = np.random.rand(), np.random.rand()

            vel = (
                w * vel
                + c1 * r1 * (pbest - pos)
                + c2 * r2 * (gbest - pos)
            )

            vel = np.clip(vel, -vmax, vmax)
            pos = np.clip(pos + vel, 5, 60)

            values = np.array([compute_delay(p) for p in pos])

            improved = values < pbest_val
            pbest[improved] = pos[improved]
            pbest_val[improved] = values[improved]

            best_idx = np.argmin(pbest_val)
            if pbest_val[best_idx] < gbest_val:
                gbest_val = pbest_val[best_idx]
                gbest = pbest[best_idx].copy()

            convergence.append(gbest_val)

    exec_time = time.time() - start_time

    # =========================================================
    # 8. RESULTS
    # =========================================================
    st.subheader("📊 Optimization Results")

    phases = ["North", "South", "East", "West"]
    col1, col2 = st.columns(2)

    with col1:
        for i, g in enumerate(gbest):
            st.write(f"🚦 Phase {i+1} ({phases[i]}): **{g:.2f} sec**")

        st.write(f"⏱ Execution Time: **{exec_time:.3f} sec**")
        st.write(f"📉 Best Fitness Value: **{gbest_val:.6f}**")

    with col2:
        df_convergence = pd.DataFrame({
            "Generation": range(1, len(convergence) + 1),
            "Fitness": convergence
        })

        chart = alt.Chart(df_convergence).mark_line().encode(
            x="Generation",
            y="Fitness"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
