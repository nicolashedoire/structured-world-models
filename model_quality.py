"""model_quality — BENCH 2: quality of the world MODELS, WITHOUT agent or planning.
We compare the RANDOM-feature model to the STRUCTURED model on what really matters for planning:
not the one-step R² (misleading), but the compounding ROLLOUT error (@10, @20, @50). This is
where structure wins, especially in high dimension.

  python3 model_quality.py
"""
from __future__ import annotations
import numpy as np

from benchlib import BENCH, RandomFeat, Structured, explore


def rollout_err(step, model, cfg, rng, horizons=(10, 20, 50), B=300):
    Z = rng.uniform(cfg["lo"], cfg["hi"], (B, len(cfg["lo"]))); seq = rng.uniform(-1, 1, (B, max(horizons), cfg["adim"]))
    at, am = Z.copy(), Z.copy(); errs = {}
    for t in range(max(horizons)):
        at = step(at, seq[:, t]); am = model.step(am, seq[:, t])
        if t+1 in horizons:
            errs[t+1] = np.linalg.norm(at-am, axis=1).mean()
    return errs


def main():
    rng = np.random.default_rng(0)
    print("=== BENCH 2 — MODEL QUALITY (no agent): one-step R² vs ROLLOUT error ===\n")
    print(f"  {'benchmark':20} | {'model':10} | {'R² 1-step':>9} | {'roll@10':>8} | {'roll@20':>8} | {'roll@50':>8}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, cfg in BENCH.items():
        S, A, Snext = explore(cfg, rng)
        struct = Structured(cfg["feat"], cfg["pos"], cfg["vel"], cfg["dt"], cfg["semi"], cfg["vclip"])
        randm = RandomFeat(rng, len(cfg["lo"])+cfg["adim"])
        struct.fit(S, A, Snext); randm.fit(S, A, Snext)
        tr = cfg["step"](S, A)
        for label, m in [("random", randm), ("STRUCTURED", struct)]:
            r2 = 1-((tr-m.step(S, A))**2).sum()/((tr-tr.mean(0))**2).sum()
            e = rollout_err(cfg["step"], m, cfg, np.random.default_rng(5))
            print(f"  {name if label=='random' else '':20} | {label:10} | {r2:>9.4f} | {e[10]:>8.3f} | {e[20]:>8.3f} | {e[50]:>8.3f}")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print("\n  → One-step R² can be ≈1 for BOTH (in low dim). It is the ROLLOUT error @50")
    print("    that reveals the truth: in high dim, random drifts, structured does not.")


if __name__ == "__main__":
    main()
