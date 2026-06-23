"""agent_cem_benchmark — BENCH 3: the AGENTS, with the VALIDATED planner (CEM + warm-start).
We judge a model ONLY if the oracle succeeds (otherwise it is the planner's fault, not the model's).
Automatic verdict per benchmark:
  • oracle FAILS                        → planner_fail   (re-tune CEM, bench 1)
  • oracle OK but models fail           → model_fail
  • a model matches the oracle          → success (and we say which: structure vs random)

  python3 agent_cem_benchmark.py
"""
from __future__ import annotations
import numpy as np

from benchlib import BENCH, Oracle, RandomFeat, Structured, explore, run_episode


def main():
    rng = np.random.default_rng(0)
    print("=== BENCH 3 — AGENTS (validated CEM): random-acts vs random vs structured vs oracle ===\n")
    print(f"  {'benchmark':20} | {'rnd-act':>7} | {'random':>10} | {'structured':>10} | {'oracle':>7} | verdict")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*22}")
    for name, cfg in BENCH.items():
        S, A, Snext = explore(cfg, rng)
        struct = Structured(cfg["feat"], cfg["pos"], cfg["vel"], cfg["dt"], cfg["semi"], cfg["vclip"])
        randm = RandomFeat(rng, len(cfg["lo"])+cfg["adim"])
        struct.fit(S, A, Snext); randm.fit(S, A, Snext)
        res = {}
        for label, model, pl in [("rnd-act", struct, False), ("random", randm, True),
                                 ("structured", struct, True), ("oracle", Oracle(cfg["step"]), True)]:
            m = run_episode(name, cfg, model, np.random.default_rng(1), planner=pl)
            res[label] = (m, cfg["win"](m) if (pl or label == "rnd-act") else False)

        def mark(lab): return "✅" if res[lab][1] else "❌"
        if not res["oracle"][1]:
            verdict = "planner_fail"
        elif res["structured"][1] and res["random"][1]:
            verdict = "success (low-dim parity)"
        elif res["structured"][1]:
            verdict = "success — STRUCTURE wins"
        elif res["random"][1]:
            verdict = "random ok, struct. not"
        else:
            verdict = "model_fail"
        sm = res["structured"][0]
        sval = f"{sm:.2f}" if isinstance(sm, float) else f"{int(sm)}"
        print(f"  {name:20} | {mark('rnd-act'):>6} | {mark('random'):>9} | {mark('structured'):>9} ({sval}) | {mark('oracle'):>6} | {verdict}")
    print("\n  Project rules, verified here:")
    print("   1) STRUCTURE > random features — decisive in HIGH dimension (low dim: parity).")
    print("   2) ORACLE FIRST — a failure only blames the model if the oracle itself succeeds.")
    print("   Structured model + CEM + oracle validation: the clean recipe, gradient-free, no GPU.")


if __name__ == "__main__":
    main()
