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

st.title(" Traffic Signal Optimization using PSO")
st.write("""
This application optimizes traffic signal green times for a four-phase intersection
using **Particle Swarm Optimization (PSO)**.
Green times remain **random**, while the fitness value dynamically changes
based on PSO parameters.
""")

# =========================================================
# 2. SIDEBAR PSO PARAMETERS
# =========================================================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_generations = st.sidebar.slider("Number of Generations", 20, 300, 100)
w = st.sidebar.slider("Inertia Weight (w)", 0.3, 1.0, 0.7)
vmax = st.sidebar.slider("Velocity Limit", 1, 20, 10)

c1, c2 = 1.8, 1.8

# =========================================================
# 3. LOAD DATASET
# =========================================================
DATA_FILE = "traffic_dataset (2).csv"

try:
    df = pd.read_csv(DATA_FILE)
    st.success(" Dataset loaded successfully")
except:
    st.error(" Dataset not found")
    st.stop()

st.subheader(" Dataset Preview")
st.dataframe(df.head())

# =========================================================
# 4. DATA VALIDATION
# =========================================================
if not {"waiting_time", "vehicle_count"}.issubset(df.columns):
    st.error("Dataset must contain waiting_time and vehicle_count")
    st.stop()

# =========================================================
# 5. TRAFFIC STATISTICS (GLOBAL)
# =========================================================
avg_wait = df["waiting_time"].mean()
avg_vehicle = df["vehicle_count"].mean()

st.subheader(" Traffic Statistics")
st.write(f"Average Waiting Time: **{avg_wait:.2f} sec**")
st.write(f"Average Vehicle Count: **{avg_vehicle:.2f} vehicles**")

# =========================================================
# 6. FITNESS FUNCTION (FIXED BUT RANDOMIZED)
# =========================================================
def compute_fitness(green_times):
    green_times = np.clip(green_times, 5, 60)

    total_green = np.sum(green_times)

    # Base delay component
    demand = avg_wait * avg_vehicle
    delay = demand / total_green

    # Balance penalty to avoid extreme imbalance
    balance_penalty = np.var(green_times)

    # Small stochastic noise (to keep fitness dynamic)
    noise = np.random.uniform(0.95, 1.05)

    fitness = delay * noise + 0.05 * balance_penalty
    return fitness

# =========================================================
# 7. RUN PSO
# =========================================================
if st.button(" Run PSO Optimization", type="primary"):

    start_time = time.time()
    dimensions = 4  # North, South, East, West

    # Initialize particles randomly
    pos = np.random.uniform(10, 50, (num_particles, dimensions))
    vel = np.random.uniform(-vmax, vmax, (num_particles, dimensions))

    pbest = pos.copy()
    pbest_val = np.array([compute_fitness(p) for p in pos])

    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    convergence = []

    with st.spinner("Optimizing traffic signal timings..."):
        for gen in range(num_generations):

            r1, r2 = np.random.rand(), np.random.rand()

            # Velocity update
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
    st.subheader(" Optimization Results")

    phases = ["North", "South", "East", "West"]
    col1, col2 = st.columns(2)

    with col1:
        st.success("Optimal Green Times")
        for i, g in enumerate(gbest):
            st.write(f"{phases[i]}: **{g:.2f} sec**")

        st.write(f"Total Green Time: **{np.sum(gbest):.2f} sec**")
        st.write(f"Best Fitness Value: **{gbest_val:.6f}**")
        st.write(f"Execution Time: **{exec_time:.3f} sec**")

    with col2:
        df_conv = pd.DataFrame({
            "Generation": range(1, len(convergence) + 1),
            "Fitness": convergence
        })

        chart = alt.Chart(df_conv).mark_line().encode(
            x=alt.X("Generation", title="Generation"),
            y=alt.Y("Fitness", title="Fitness Value (Lower is Better)")
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

# =========================================================
# 9. CONCLUSION
# =========================================================
st.divider()
st.header("Conclusion")

st.markdown("""
The PSO-based traffic signal optimization successfully explores randomized
green time configurations while maintaining sensitivity to parameter changes.
The inclusion of stochastic elements in the fitness function prevents premature
convergence and ensures varied optimization outcomes when particle size and
generation count are modified.
""")
