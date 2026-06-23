"""planner_cem_solution — BENCH 1: ORACLE FIRST. Validate that the planner (CEM + warm-start)
solves the environments WITH THE TRUE PHYSICS. As long as this bench fails, comparing learned
models is meaningless (we could not tell whether a failure comes from the model or the planner).

  python3 planner_cem_solution.py
"""
from __future__ import annotations
import numpy as np

from benchlib import BENCH, Oracle, run_episode


def main():
    print("=== BENCH 1 — ORACLE FIRST: does CEM + true physics solve the environments? ===\n")
    print(f"  {'benchmark':20} | {'result (oracle)':32} | verdict")
    print(f"  {'-'*20}-+-{'-'*32}-+--------")
    allok = True
    for name, cfg in BENCH.items():
        m = run_episode(name, cfg, Oracle(cfg["step"]), np.random.default_rng(1))
        ok = cfg["win"](m); allok &= ok
        print(f"  {name:20} | {cfg['fmt'](m):32} | {'OK ✅' if ok else 'FAIL ❌'}")
    print("\n  " + ("→ planner VALIDATED: we can now judge the models (bench 3)."
                    if allok else "→ planner INSUFFICIENT: tune CEM/cost BEFORE judging the models."))


if __name__ == "__main__":
    main()
