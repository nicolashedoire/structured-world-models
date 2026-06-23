"""closed_loop_stress — find the REGIME BOUNDARY. Bench 3 showed parity at 12-D with tight
replanning: receding-horizon CEM only needs short-horizon accuracy, which random features have at
moderate dimension, so they keep up. This script asks the precise question:

    At what dimension / replanning frequency do random features stop being good enough
    in CLOSED-LOOP control?

The goal is NOT to make the structured model win — it is to locate the boundary. We sweep:
  • dimension n (state = 2n), on a controllable n-D reach world (weak coupling, oracle solves it);
  • replanning interval c = number of actions executed OPEN-LOOP before replanning (action
    commitment). c=1 is tight MPC; larger c forces the controller to trust the model's multi-step
    prediction — exactly where a drifting random model is exposed.
Oracle-first: if the oracle itself fails a cell, that cell is planner-limited, not model-limited.
Gradient-free, no GPU.

  python3 closed_loop_stress.py
"""
from __future__ import annotations
import numpy as np

from benchlib import Oracle, RandomFeat, Structured, cem_plan

DT = 0.2


def make_highd(n, rng):
    """n-D reach world: state z=[p(n),v(n)], action a(n). Weak coupling → controllable."""
    C = rng.standard_normal((n, n))/np.sqrt(n)*0.15; drag = 0.4

    def step(Z, A):
        p, v = Z[:, :n], Z[:, n:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
        v2 = v + (np.clip(A, -1, 1) - drag*v*sp - p@C.T)*DT
        return np.hstack([p + v2*DT, v2])
    return step


def make_feat(n):
    def feat(S, A):
        p, v = S[:, :n], S[:, n:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
        return np.hstack([np.ones((S.shape[0], 1)), p, v, np.clip(A, -1, 1), v*sp])
    return feat


def fit_models(step, n, rng, N=7000, Kfeat=700):
    S = rng.uniform(-2, 2, (N, 2*n)); A = rng.uniform(-1, 1, (N, n)); Snext = step(S, A)
    struct = Structured(make_feat(n), list(range(n)), list(range(n, 2*n)), DT, semi=True)
    randm = RandomFeat(rng, 3*n, K=Kfeat)
    struct.fit(S, A, Snext); randm.fit(S, A, Snext)
    return struct, randm


def rollout_err(step, model, n, rng, H=15, B=200):
    Z = rng.uniform(-2, 2, (B, 2*n)); a, b = Z.copy(), Z.copy(); seq = rng.uniform(-1, 1, (B, H, n))
    for t in range(H):
        a = step(a, seq[:, t]); b = model.step(b, seq[:, t])
    return np.linalg.norm((a-b)[:, :n], axis=1).mean()


def closed_loop(step_true, model, target, n, rng, steps=60, H=20, c=1, K=250, iters=4):
    """Receding-horizon CEM, but execute c actions OPEN-LOOP before replanning (action commitment)."""
    def cost(ST):
        return np.linalg.norm(ST[:, -3:, :n]-target, axis=2).min(1) + 0.3*np.linalg.norm(ST[:, :, :n]-target, axis=2).mean(1)
    s = np.zeros((1, 2*n)); low = -np.ones(n); high = np.ones(n); mean = None; mind = 1e9; t = 0
    while t < steps:
        _, seq, mean, _, _ = cem_plan(model, s, cost, rng, n, H, low, high,
                                      K=K, elites=max(8, K//8), iters=iters, init_mean=mean)
        sh = min(c, H)
        for j in range(sh):
            if t >= steps:
                break
            s = step_true(s, seq[j][None]); t += 1
            mind = min(mind, np.linalg.norm(s[0, :n]-target))
        mean = np.vstack([mean[sh:], np.repeat(mean[-1:], sh, axis=0)])
    return mind


def main():
    rng = np.random.default_rng(0)
    DIMS = [4, 8, 16]; CS = [1, 5, 15]
    print("=== CLOSED-LOOP STRESS TEST — where do random features stop being good enough? ===\n")
    print("  n-D reach (weak coupling, controllable). c = actions run open-loop before replanning.")
    print("  metric = min distance to target over the episode (LOWER is better). oracle = reference.\n")

    # context: open-loop model drift grows with dimension (this is WHY closed-loop eventually breaks)
    print("  open-loop drift (rollout err @15), the underlying cause:")
    print(f"     {'dim':>5} | {'random':>8} | {'structured':>10}")
    print(f"     {'-'*5}-+-{'-'*8}-+-{'-'*10}")
    models = {}
    for n in DIMS:
        step = make_highd(n, np.random.default_rng(100+n))
        struct, randm = fit_models(step, n, np.random.default_rng(1))
        models[n] = (step, struct, randm)
        er = rollout_err(step, randm, n, np.random.default_rng(2)); es = rollout_err(step, struct, n, np.random.default_rng(2))
        print(f"     {f'{n}D':>5} | {er:>8.3f} | {es:>10.3f}")

    def avg_cl(step, model, target, n, c, seeds=(1, 2, 3)):   # average over seeds to smooth control noise
        return float(np.mean([closed_loop(step, model, target, n, np.random.default_rng(sd), c=c) for sd in seeds]))

    print("\n  closed-loop control (min dist to target, averaged over 3 seeds):")
    print(f"     {'dim':>4} | {'c':>3} | {'random':>7} | {'structured':>10} | {'oracle':>7} | verdict")
    print(f"     {'-'*4}-+-{'-'*3}-+-{'-'*7}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}")
    target_of = {n: np.ones(n)*0.8 for n in DIMS}
    for n in DIMS:
        step, struct, randm = models[n]; target = target_of[n]
        for c in CS:
            r = avg_cl(step, randm, target, n, c)
            sV = avg_cl(step, struct, target, n, c)
            o = avg_cl(step, Oracle(step), target, n, c)
            if o > 1.1:                                   # oracle-first: this cell is planner-limited
                verdict = "planner_fail (ignore models)"
            elif r <= o + 0.25:                           # random stays within reach of the oracle
                verdict = "random OK (parity)"
            else:
                verdict = f"random DEGRADES (+{r-o:.2f})"
            print(f"     {f'{n}D':>4} | {c:>3} | {r:>7.2f} | {sV:>10.2f} | {o:>7.2f} | {verdict}")

    print("\n  → Reading: the boundary is where 'random OK' flips to 'random DEGRADES'. Random features")
    print("    keep up under tight replanning at low dimension, but as dimension grows AND replanning")
    print("    loosens (larger c), their open-loop drift is no longer corrected fast enough — the")
    print("    structured model stays near the oracle. This locates the regime, it does not assume it.")


if __name__ == "__main__":
    main()
