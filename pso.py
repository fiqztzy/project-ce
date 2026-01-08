# =========================================================
# TRAFFIC SIGNAL OPTIMIZATION USING PSO (STREAMLIT)
# SINGLE FOLDER VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# =========================================================
# 1. APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Traffic Signal Optimization (PSO)",
    layout="wide"
)

st.title("🚦 Traffic Signal Optimization using PSO")
st.write("""
This application uses **Particle Swarm Optimization (PSO)** to optimize
traffic light green times for a **four-way intersection**.
""")

# =========================================================
# 2. LOAD DATASET (AUTO FROM SAME FOLDER)
# =========================================================
DATASET_PATH = "traffic_dataset (2).csv"

if not os.path.exists(DATASET_PATH):
    st.error("❌ traffic_dataset.csv not found in the project folder.")
    st.stop()

df = pd.read_csv(DATASET_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================================================
# 3. EXTRACT TRAFFIC FLOWS (4 PHASES)
# =========================================================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

if len(numeric_cols) < 4:
    st.error("Dataset must contain at least 4 numeric traffic flow columns.")
    st.stop()

directions = ["North", "South", "East", "West"]
traffic_flows = df[numeric_cols[:4]].mean().to_numpy()

st.subheader("🚘 Average Traffic Flow (veh/hr)")
st.table(pd.DataFrame({
    "Direction": directions,
    "Average Flow (veh/hr)": traffic_flows.round(2)
}))

# =========================================================
# 4. SIDEBAR – PSO PARAMETERS
# =========================================================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 20, 100, 50, 10)
num_iterations = st.sidebar.slider("Number of Iterations", 50, 300, 150, 10)

w_max = st.sidebar.slider("Max Inertia Weight (w_max)", 0.6, 1.2, 0.9)
w_min = st.sidebar.slider("Min Inertia Weight (w_min)", 0.1, 0.6, 0.4)

c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 1.0, 3.0, 2.0)
c2 = st.sidebar.slider("Social Coefficient (c2)", 1.0, 3.0, 2.0)

early_stop = st.sidebar.slider("Early Stopping Patience", 10, 50, 20)

# =========================================================
# 5. FITNESS FUNCTION (TOTAL DELAY)
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
# 6. RUN PSO
# =========================================================
st.subheader("🚀 Running PSO Optimization")

dimensions = 4  # North, South, East, West

pos = np.random.uniform(10, 50, (num_particles, dimensions))
vel = np.random.uniform(-5, 5, (num_particles, dimensions))

pbest = pos.copy()
pbest_vals = np.array([compute_delay(p) for p in pos])

gbest_idx = np.argmin(pbest_vals)
gbest = pbest[gbest_idx].copy()
gbest_val = pbest_vals[gbest_idx]

convergence_curve = []
stall_counter = 0
best_prev = np.inf

start_time = time.time()

for it in range(num_iterations):

    # Dynamic inertia weight
    w = w_max - (w_max - w_min) * (it / num_iterations)

    r1 = np.random.rand(num_particles, dimensions)
    r2 = np.random.rand(num_particles, dimensions)

    vel = (
        w * vel
        + c1 * r1 * (pbest - pos)
        + c2 * r2 * (gbest - pos)
    )

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
# 7. DISPLAY RESULTS
# =========================================================
st.subheader("📈 Optimization Results")

col1, col2 = st.columns(2)

# ---- Best Result ----
with col1:
    st.success("✅ Best Traffic Light Timing Found")
    for i, (d, g) in enumerate(zip(directions, gbest), 1):
        st.write(f"🚦 Phase {i} ({d}) : **{round(g, 2)} seconds**")

    st.write(f"🕒 Execution Time: **{exec_time:.4f} seconds**")
    st.write(f"📉 Total Delay (Fitness Value): **{round(gbest_val, 6)}**")
    st.write(f"⏱ Total Green Time: **{round(np.sum(gbest), 2)} sec**")

# ---- Convergence Graph ----
with col2:
    st.subheader("📉 PSO Convergence Curve")

    convergence_df = pd.DataFrame({
        "Iteration Number": range(1, len(convergence_curve) + 1),
        "Best Total Delay (Fitness Value)": convergence_curve
    })

    st.caption(
        "X-axis: Iteration Number | "
        "Y-axis: Best Total Delay (Fitness Value)"
    )

    st.line_chart(convergence_df.set_index("Iteration Number"))
