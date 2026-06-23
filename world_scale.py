"""world_scale — SCALE the world model. The agent bench showed the wall is not the planner
(3/3 with the oracle) but the model's FIDELITY when stretched over a horizon. Here we stretch the
world itself: an n-dimensional dynamical system (coupled positions+velocities, nonlinear drag),
n = 2 → 32. We measure, gradient-free (closed form), how it holds:
  • one-step R² (local fidelity)
  • ROLLOUT error over H steps (what matters for planning) — SINGLE model vs ENSEMBLE
No gradient, no GPU.

  python3 world_scale.py
"""
from __future__ import annotations
import numpy as np

DT = 0.2


def make_world(n, rng):
    """n-dim world: state z=[p(n),v(n)], action a(n). Coupling C between dimensions = difficulty."""
    C = rng.standard_normal((n, n)) / np.sqrt(n) * 0.5    # inter-dimension coupling (normalized)
    drag = 0.4

    def step(Z, A):
        p, v = Z[:, :n], Z[:, n:]
        sp = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        v2 = v + (A - drag*v*sp - p @ C.T)*DT             # nonlinear drag + coupled restoring force
        p2 = p + v2*DT
        return np.hstack([p2, v2])
    return step


def learn(step, n, rng, N=8000, K=1000, M=1):
    d_in = 3*n                                            # [p,v,a]
    feats = [(rng.standard_normal((d_in, K))/np.sqrt(d_in)*3.0, rng.standard_normal(K)) for _ in range(M)]
    Z = rng.uniform(-2, 2, (N, 2*n)); A = rng.uniform(-1, 1, (N, n))
    X = np.hstack([Z, A]); dZ = step(Z, A) - Z
    mu, sd = X.mean(0), X.std(0)+1e-9
    phi = lambda XX, R, b: np.cos((XX-mu)/sd @ R + b)
    Ws = [np.linalg.solve(phi(X, R, b).T@phi(X, R, b) + 1.0*np.eye(K), phi(X, R, b).T@dZ) for (R, b) in feats]

    def model(Z, A):                                      # ensemble mean
        XX = np.hstack([Z, A])
        return Z + np.mean([phi(XX, R, b) @ W for (R, b), W in zip(feats, Ws)], 0)
    return model, mu, sd


def r2(true, pred):
    return 1 - ((true-pred)**2).sum()/((true-true.mean(0))**2).sum()


def rollout_err(step, model, n, rng, H=20, B=200):
    """Open-loop rollout error: unroll H steps with the model vs the true physics."""
    Z = rng.uniform(-2, 2, (B, 2*n)); Zt = Z.copy(); Zm = Z.copy()
    seq = rng.uniform(-1, 1, (B, H, n)); errs = []
    for t in range(H):
        Zt = step(Zt, seq[:, t]); Zm = model(Zm, seq[:, t])
        errs.append(np.linalg.norm((Zt-Zm)[:, :n], axis=1).mean())   # POSITION error (what matters)
    return errs[H//2-1], errs[-1]                          # error at mid-horizon and at H


def main():
    rng = np.random.default_rng(0)
    print("=== SCALE the world model — gradient-free, closed form ===\n")
    print(f"  {'world dim':>10} | {'R² 1-step':>9} | {'rollout err @H/2':>16} | {'rollout err @H':>14} | {'ensemble @H':>12}")
    print(f"  {'(state)':>10} | {'(single)':>9} | {'(single model)':>16} | {'(single)':>14} | {'(3 models)':>12}")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*16}-+-{'-'*14}-+-{'-'*12}")
    for n in (2, 4, 8, 16, 32):
        step = make_world(n, np.random.default_rng(100+n))
        # SINGLE model
        m1, _, _ = learn(step, n, np.random.default_rng(1), M=1)
        Zt = rng.uniform(-2, 2, (3000, 2*n)); At = rng.uniform(-1, 1, (3000, n))
        R2 = r2(step(Zt, At), m1(Zt, At))
        eh2, eh = rollout_err(step, m1, n, np.random.default_rng(2))
        # ENSEMBLE of 3
        m3, _, _ = learn(step, n, np.random.default_rng(1), M=3)
        _, eh_ens = rollout_err(step, m3, n, np.random.default_rng(2))
        tag = "✅" if R2 > 0.95 else ("~" if R2 > 0.8 else "❌")
        print(f"  {'%dD (%d)' % (n, 2*n):>10} | {R2:>8.3f}{tag} | {eh2:>16.3f} | {eh:>14.3f} | {eh_ens:>11.3f}{'  ↓' if eh_ens<eh else ''}")
    print("\n  → one-step R² = local fidelity ; rollout err = what wrecks planning (compounding error).")
    print("    You SEE the scaling: where the closed form holds, where the rollout drifts, and by how much")
    print("    the ENSEMBLE (self-consistency) trims the compounding error. All gradient-free, no GPU.")


if __name__ == "__main__":
    main()
