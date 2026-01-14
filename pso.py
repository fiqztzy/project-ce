# =========================================================
# STREAMLIT TRAFFIC SIGNAL OPTIMIZATION USING PSO (FIXED)
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
(**North, South, East, West**) using **Particle Swarm Optimization (PSO)**.
""")

# =========================================================
# 2. SIDEBAR PSO PARAMETERS
# =========================================================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_generations = st.sidebar.slider("Number of Generations", 20, 300, 100)
w = st.sidebar.slider("Inertia Weight (w)", 0.3, 1.0, 0.7)
vmax = st.sidebar.slider("Velocity Limit", 1, 20, 10)

c1, c2 = 2.0, 2.0

# =========================================================
# 3. LOAD DATASET
# =========================================================
DATA_FILE = "traffic_dataset (2).csv"

try:
    df = pd.read_csv(DATA_FILE)
    st.success("✅ Dataset loaded successfully")
except FileNotFoundError:
    st.error("❌ Dataset not found")
    st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(df)

# =========================================================
# 4. DATA VALIDATION
# =========================================================
required_cols = {"direction", "waiting_time", "vehicle_count"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must contain: direction, waiting_time, vehicle_count")
    st.stop()

# =========================================================
# 5. PREPARE TRAFFIC DEMAND PER DIRECTION
# =========================================================
direction_order = ["North", "South", "East", "West"]

df_grouped = df.groupby("direction").mean().reindex(direction_order)

wait = df_grouped["waiting_time"].values
veh = df_grouped["vehicle_count"].values

st.subheader("📊 Traffic Demand by Direction")
st.dataframe(df_grouped)

# =========================================================
# 6. FITNESS FUNCTION (FIXED)
# =========================================================
def compute_fitness(green_times):
    green_times = np.clip(green_times, 5, 60)

    # Delay model (higher demand → more green time needed)
    delay = np.sum((wait * veh) / green_times)

    # Penalize extreme imbalance
    balance_penalty = np.var(green_times)

    return delay + 0.3 * balance_penalty

# =========================================================
# 7. RUN PSO
# =========================================================
if st.button("🚀 Run PSO Optimization", type="primary"):

    start_time = time.time()
    dimensions = 4

    pos = np.random.uniform(10, 50, (num_particles, dimensions))
    vel = np.random.uniform(-vmax, vmax, (num_particles, dimensions))

    pbest = pos.copy()
    pbest_val = np.array([compute_fitness(p) for p in pos])

    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    convergence = []

    with st.spinner("Optimizing traffic signals..."):
        for gen in range(num_generations):

            r1 = np.random.rand(num_particles, dimensions)
            r2 = np.random.rand(num_particles, dimensions)

            vel = (
                w * vel
                + c1 * r1 * (pbest - pos)
                + c2 * r2 * (gbest - pos)
            )

            vel = np.clip(vel, -vmax, vmax)
            pos = np.clip(pos + vel, 5, 60)

            fitness = np.array([compute_fitness(p) for p in pos])

            improved = fitness < pbest_val
            pbest[improved] = pos[improved]
            pbest_val[improved] = fitness[improved]

            best_idx = np.argmin(pbest_val)
            if pbest_val[best_idx] < gbest_val:
                gbest_val = pbest_val[best_idx]
                gbest = pbest[best_idx].copy()

            convergence.append(gbest_val)

    exec_time = time.time() - start_time

    # =========================================================
    # 8. RESULTS
    # =========================================================
    st.subheader("✅ Optimization Results")

    col1, col2 = st.columns(2)

    with col1:
        st.success("Optimal Green Times")
        for i, d in enumerate(direction_order):
            st.write(f"**{d}**: {gbest[i]:.2f} sec")

        st.write(f"**Total Green Time:** {np.sum(gbest):.2f} sec")
        st.write(f"**Best Fitness Value:** {gbest_val:.6f}")
        st.write(f"**Execution Time:** {exec_time:.3f} sec")

    with col2:
        df_conv = pd.DataFrame({
            "Generation": range(1, len(convergence) + 1),
            "Fitness": convergence
        })

        chart = alt.Chart(df_conv).mark_line().encode(
            x="Generation",
            y="Fitness"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

# =========================================================
# 9. CONCLUSION
# =========================================================
st.divider()
st.header("📌 Conclusion")

st.markdown("""
This improved PSO-based traffic signal optimization system dynamically allocates
green times according to real traffic demand per direction. The enhanced fitness
function ensures meaningful convergence, improved exploration, and realistic
signal timing optimization suitable for real-world deployment.
""")
