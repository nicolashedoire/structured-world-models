"""world_scale_solution — THE scaling solution, verified here. We stop learning p' directly: we
IMPOSE p' = p + v'·dt and learn only Δv, with a PHYSICAL/INVARIANT basis
φ = [1, p, v, a, v·||v||] instead of random features. Closed form.
The world has quadratic drag v·||v|| + a linear coupling p·Cᵀ — so the basis contains exactly the
right generators → R²=1 exactly. (Honesty: it works BECAUSE the basis matches the world; the version
without that knowledge is in world_scale_robust.py.)

  python3 world_scale_solution.py
"""
from __future__ import annotations
import numpy as np

DT = 0.2


def make_world(n, rng):
    C = rng.standard_normal((n, n)) / np.sqrt(n) * 0.5; drag = 0.4

    def step(Z, A):
        p, v = Z[:, :n], Z[:, n:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
        v2 = v + (A - drag*v*sp - p @ C.T)*DT; p2 = p + v2*DT
        return np.hstack([p2, v2])
    return step


def features(Z, A, n):
    p, v = Z[:, :n], Z[:, n:]; speed = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
    return np.hstack([np.ones((Z.shape[0], 1)), p, v, A, v*speed])   # invariant basis


def main():
    print("=== Scaling solution — structured model + invariant features, gradient-free ===\n")
    print(f"  {'world dim':>11} | {'features':>8} | {'N':>5} | {'R² 1-step':>9} | {'rollout err @20':>15}")
    print(f"  {'-'*11}-+-{'-'*8}-+-{'-'*5}-+-{'-'*9}-+-{'-'*15}")
    for n in (16, 32, 64):
        rng = np.random.default_rng(100+n); step = make_world(n, rng)
        F0 = features(np.zeros((1, 2*n)), np.zeros((1, n)), n).shape[1]; N = 20*F0
        Z = rng.uniform(-2, 2, (N, 2*n)); A = rng.uniform(-1, 1, (N, n))
        Y = step(Z, A)[:, n:] - Z[:, n:]                  # learn Δv ONLY
        F = features(Z, A, n)
        W = np.linalg.solve(F.T@F + 1e-6*np.eye(F.shape[1]), F.T@Y)

        def model(Z, A):
            p, v = Z[:, :n], Z[:, n:]; v2 = v + features(Z, A, n) @ W; return np.hstack([p+v2*DT, v2])

        Zt = rng.uniform(-2, 2, (3000, 2*n)); At = rng.uniform(-1, 1, (3000, n))
        tr = step(Zt, At); pr = model(Zt, At)
        R2 = 1 - ((tr-pr)**2).sum()/((tr-tr.mean(0))**2).sum()
        Zr = rng.uniform(-2, 2, (200, 2*n)); a, bb = Zr.copy(), Zr.copy(); seq = rng.uniform(-1, 1, (200, 20, n))
        for t in range(20):
            a = step(a, seq[:, t]); bb = model(bb, seq[:, t])
        eh = np.linalg.norm((a-bb)[:, :n], axis=1).mean()
        print(f"  {'%dD (%d)' % (n, 2*n):>11} | {F0:>8} | {N:>5} | {R2:>9.6f} | {eh:>15.6f}")
    print("\n  → R²=1, rollout=0: the gradient-free model SCALES up to 64D. Random features do NOT.")
    print("    Rule: structure > learned features > local models > more K/N.")


if __name__ == "__main__":
    main()
