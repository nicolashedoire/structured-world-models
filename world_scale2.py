"""world_scale2 — the SCALING LAW of the gradient-free world model. At 16D (where it broke down),
does scaling CAPACITY (K features) and DATA (N) restore fidelity — WITHOUT gradient? If R² climbs
back toward 1 as K and N grow, then the model SCALES: all that's missing is closed-form compute,
not gradient descent.

  python3 world_scale2.py
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


def fit_eval(step, n, N, K, rng):
    d_in = 3*n
    R = rng.standard_normal((d_in, K))/np.sqrt(d_in)*3.0; b = rng.standard_normal(K)
    Z = rng.uniform(-2, 2, (N, 2*n)); A = rng.uniform(-1, 1, (N, n))
    X = np.hstack([Z, A]); dZ = step(Z, A)-Z; mu, sd = X.mean(0), X.std(0)+1e-9
    phi = lambda XX: np.cos((XX-mu)/sd @ R + b)
    P = phi(X); W = np.linalg.solve(P.T@P + 1.0*np.eye(K), P.T@dZ)
    model = lambda Z, A: Z + phi(np.hstack([Z, A])) @ W
    Zt = rng.uniform(-2, 2, (3000, 2*n)); At = rng.uniform(-1, 1, (3000, n))
    tr = step(Zt, At); pr = model(Zt, At)
    R2 = 1 - ((tr-pr)**2).sum()/((tr-tr.mean(0))**2).sum()
    # rollout error @H
    Zr = rng.uniform(-2, 2, (200, 2*n)); Zt2 = Zr.copy(); Zm = Zr.copy(); seq = rng.uniform(-1, 1, (200, 20, n))
    for t in range(20):
        Zt2 = step(Zt2, seq[:, t]); Zm = model(Zm, seq[:, t])
    eh = np.linalg.norm((Zt2-Zm)[:, :n], axis=1).mean()
    return R2, eh


def main():
    n = 16
    print(f"=== Gradient-free scaling law at {n}D (16 positions + 16 velocities) ===\n")
    print(f"  {'N data':>10} | {'K=1000':>14} | {'K=2000':>14} | {'K=4000':>14}")
    print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    step = make_world(n, np.random.default_rng(116))
    for N in (8000, 16000, 32000):
        row = []
        for K in (1000, 2000, 4000):
            R2, eh = fit_eval(step, n, N, K, np.random.default_rng(7))
            tag = "✅" if R2 > 0.95 else ("~" if R2 > 0.85 else "❌")
            row.append(f"R²={R2:.3f}{tag} e{eh:.1f}")
        print(f"  {N:>10} | {row[0]:>14} | {row[1]:>14} | {row[2]:>14}")
    print("\n  R² = one-step fidelity ; e = rollout error @20. If R² climbs with K and N →")
    print("  the world model SCALES in high dimension WITHOUT gradient: all that's missing is")
    print("  closed-form compute (a bigger SVD/solve), never backpropagation. No GPU.")


if __name__ == "__main__":
    main()
