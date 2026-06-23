"""viz_trajectories — the figure that makes the result SHOWABLE: side-by-side trajectories,
RANDOM model (wanders, 0/3) vs STRUCTURED model (reaches all 3 goals), same world, same planner.
Writes world_models_trajectories.png at the repo root. Gradient-free, no GPU.

  python3 viz_trajectories.py
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_canon import (GOALS, OBST, Oracle, WorldModelRandom,    # noqa: E402
                         WorldModelStructured, explore, smooth_plan, true_step)

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Circle                               # noqa: E402


def tour_traj(model, rng, use_planner=True):
    s = np.array([[0., 0., 0., 0.]]); traj = [s[0, :2].copy()]; gi = t = 0; marks = []
    while gi < len(GOALS) and t < 300:
        a = smooth_plan(model, s, GOALS[gi], rng) if use_planner else rng.uniform(-1, 1, 2)
        s = true_step(s, a[None]); t += 1; traj.append(s[0, :2].copy())
        if np.linalg.norm(s[0, :2]-GOALS[gi]) < 0.6:
            marks.append(len(traj)-1); gi += 1
    return np.array(traj), gi, marks


def draw(ax, traj, reached, title):
    for cx, cy, r in OBST:                                          # obstacles
        ax.add_patch(Circle((cx, cy), r, color="0.7", zorder=1))
        ax.add_patch(Circle((cx, cy), r, fill=False, color="0.4", lw=1.2, zorder=2))
    for i, g in enumerate(GOALS):                                  # goals
        ax.scatter(*g, marker="*", s=420, c="#f4b400", edgecolors="#a07400", lw=1.2, zorder=4)
        ax.annotate(f"goal {i+1}", g, textcoords="offset points", xytext=(8, 8), fontsize=9, weight="bold")
    ax.scatter(*traj[0], marker="s", s=90, c="#0f9d58", zorder=5, label="start")
    ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap="viridis", s=10, zorder=3)
    ax.plot(traj[:, 0], traj[:, 1], color="0.3", lw=0.6, alpha=0.5, zorder=2)
    ax.scatter(*traj[-1], marker="X", s=120, c="#db4437", zorder=6, label="end")
    col = "#0f9d58" if reached == 3 else "#db4437"
    ax.set_title(f"{title}\n{reached}/3 goals reached", color=col, fontsize=13, weight="bold")
    ax.set_xlim(-1.2, 4.2); ax.set_ylim(-1.2, 4.2); ax.set_aspect("equal")
    ax.grid(alpha=0.15); ax.legend(loc="lower left", fontsize=8, framealpha=0.9)


def main():
    rng = np.random.default_rng(0)
    S, A, Snext = explore(rng)
    mr = WorldModelRandom(rng); ms = WorldModelStructured()
    mr.fit(S, A, Snext); ms.fit(S, A, Snext)

    tr_r, r_r, _ = tour_traj(mr, np.random.default_rng(1))
    tr_s, r_s, _ = tour_traj(ms, np.random.default_rng(1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    draw(axes[0], tr_r, r_r, "RANDOM model (random features)")
    draw(axes[1], tr_s, r_s, "STRUCTURED model (closed form)")
    fig.suptitle("Closed-Form World Models — Structure Beats Scale  ·  gradient-free, no GPU",
                 fontsize=14, weight="bold")
    fig.text(0.5, 0.005, "same world · same planner · only the world model's BASIS changes",
             ha="center", fontsize=10, style="italic", color="0.35")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "world_models_trajectories.png")
    fig.savefig(out, dpi=130); print(f"image written: {out}")
    print(f"  random {r_r}/3   |   structured {r_s}/3")


if __name__ == "__main__":
    main()
