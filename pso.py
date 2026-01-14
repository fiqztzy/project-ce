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

# =========================================================
# 2. SIDEBAR PARAMETERS
# =========================================================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
num_generations = st.sidebar.slider("Number of Generations", 20, 300, 100)
w = st.sidebar.slider("Inertia Weight (w)", 0.3, 1.0, 0.7)
vmax = st.sidebar.slider("Velocity Limit", 1, 20, 10)

c1, c2 = 1.8, 1.8

# =========================================================
# 3. LOAD DATA
# =========================================================
DATA_FILE = "traffic_dataset (2).csv"

try:
    df = pd.read_csv(DATA_FILE)
    st.success("✅ Dataset loaded")
except:
    st.error("❌ Dataset not found")
    st.stop()

# =========================================================
# 4. PREPARE TRAFFIC DEMAND (PER PHASE)
# =========================================================
df = df.sample(frac=1).reset_index(drop=True)

phase_demand = df.groupby(df.index % 4).mean()

waiting = phase_demand["waiting_time"].values
vehicles = phase_demand["vehicle_count"].values

st.subheader("📊 Traffic Demand Per Phase")
for i, d in enumerate(["North", "South", "East", "West"]):
    st.write(f"{d}: {waiting[i]:.2f} sec, {vehicles[i]:.1f} vehicles")

# =========================================================
# 5. FITNESS FUNCTION (FIXED)
# =========================================================
def compute_fitness(green_times):
    green_times = np.clip(green_times, 5, 60)

    # Delay per phase
    delay = np.sum((waiting * vehicles) / green_times)

    # Penalize imbalance
    imbalance_penalty = np.var(green_times)

    return delay + imbalance_penalty

# =========================================================
# 6. RUN PSO
# =========================================================
if st.button("🚀 Run PSO Optimization"):

    start = time.time()
    dim = 4

    pos = np.random.uniform(10, 50, (num_particles, dim))
    vel = np.random.uniform(-vmax, vmax, (num_particles, dim))

    pbest = pos.copy()
    pbest_val = np.array([compute_fitness(p) for p in pos])

    gbest = pbest[np.argmin(pbest_val)]
    gbest_val = pbest_val.min()

    convergence = []

    for _ in range(num_generations):
        r1, r2 = np.random.rand(), np.random.rand()

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
            gbest = pbest[best_idx]
            gbest_val = pbest_val[best_idx]

        convergence.append(gbest_val)

    exec_time = time.time() - start

    # =========================================================
    # 7. RESULTS
    # =========================================================
    st.subheader("✅ Optimization Results")

    phases = ["North", "South", "East", "West"]
    for i in range(4):
        st.write(f"{phases[i]}: **{gbest[i]:.2f} sec**")

    st.write(f"**Total Green Time:** {np.sum(gbest):.2f} sec")
    st.write(f"**Best Fitness Value:** {gbest_val:.4f}")
    st.write(f"**Execution Time:** {exec_time:.3f} sec")

    # Convergence Graph
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
# 8. CONCLUSION
# =========================================================
st.divider()
st.markdown("""
### 🧠 Conclusion
The improved PSO formulation dynamically adapts traffic signal timings by
considering per-phase demand. Changes in particle count and generations now
directly affect the optimization outcome, producing diverse and realistic
solutions suitable for adaptive traffic control.
""")
