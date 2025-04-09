import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === CONFIGURATION ===
task_id = 2
run_id = 0
grid_path = f"data/grid_files/grid_{task_id}.npy"
log_path = f"data/proj_ii_solutions/{task_id}_{run_id}.csv"

# === LOAD GRID & LOG ===
grid = np.load(grid_path)
log = pd.read_csv(log_path)

rows, cols = grid.shape
n_steps = len(log)

# === EXTRACT AGENT TRAJECTORIES ===
tom_pos = list(zip(log["Tom_X"], log["Tom_Y"]))
jerry_pos = list(zip(log["Jerry_X"], log["Jerry_Y"]))
spike_pos = list(zip(log["Spike_X"], log["Spike_Y"]))

# === SETUP PLOT ===
fig, ax = plt.subplots()
ax.set_xlim(-0.5, cols - 0.5)
ax.set_ylim(-0.5, rows - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_title(f"Task {task_id} - Run {run_id} - Agent Simulation")

# Draw obstacles
for r in range(rows):
    for c in range(cols):
        if grid[r, c] == 1:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))

# Agent markers & trails
tom_dot, = ax.plot([], [], 'ro', markersize=8, label='Tom')
jerry_dot, = ax.plot([], [], 'go', markersize=8, label='Jerry')
spike_dot, = ax.plot([], [], 'bo', markersize=8, label='Spike')

tom_trail, = ax.plot([], [], 'r--', linewidth=1, alpha=0.5)
jerry_trail, = ax.plot([], [], 'g--', linewidth=1, alpha=0.5)
spike_trail, = ax.plot([], [], 'b--', linewidth=1, alpha=0.5)

ax.legend(loc='upper right')

# === ANIMATION UPDATE ===
def update(i):
    # Set current agent positions (as lists)
    tom_dot.set_data([tom_pos[i][1]], [tom_pos[i][0]])
    jerry_dot.set_data([jerry_pos[i][1]], [jerry_pos[i][0]])
    spike_dot.set_data([spike_pos[i][1]], [spike_pos[i][0]])

    # Set trail paths
    tom_trail.set_data([p[1] for p in tom_pos[:i+1]], [p[0] for p in tom_pos[:i+1]])
    jerry_trail.set_data([p[1] for p in jerry_pos[:i+1]], [p[0] for p in jerry_pos[:i+1]])
    spike_trail.set_data([p[1] for p in spike_pos[:i+1]], [p[0] for p in spike_pos[:i+1]])

    return tom_dot, jerry_dot, spike_dot, tom_trail, jerry_trail, spike_trail

# === RUN ANIMATION ===
ani = animation.FuncAnimation(
    fig, update, frames=n_steps, interval=250, blit=True
)
plt.show()