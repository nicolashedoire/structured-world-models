"""agent_canon — THE CANONICAL AGENT. An autonomous gradient-free agent (no gradient descent,
no GPU) that LEARNS a world's dynamics ONLINE then PLANS a tour of goals while avoiding obstacles.
The project's demonstrable result:

    random model      : 0/3
    structured model  : 3/3
    oracle            : 3/3

The key, measured point: it was neither the planning nor the dimension that blocked it — it was the
BASIS of the world model. RANDOM features have a good one-step R² but their rollout DRIFTS → the
agent fails. The STRUCTURED model (impose p'=p+v'·dt and learn only Δv on an invariant basis)
reaches R²=1 → oracle quality → the agent succeeds. All in closed form.

  python3 agent_canon.py
"""
from __future__ import annotations
import numpy as np

# ─── the world (unknown to the agent) ───────────────────────────────────────
DT, DRAG, CURRENT = 0.2, 0.4, 0.2
OBST = [(1.4, 0.9, 0.4), (0.8, 2.0, 0.4), (2.5, 1.2, 0.4)]      # (cx, cy, radius)
GOALS = [np.array([3., 2.]), np.array([0., 3.]), np.array([3., 0.])]


def true_step(S, A):
    """True physics: quadratic drag v·||v|| + constant downward current."""
    x, y, vx, vy = S.T; sp = np.sqrt(vx*vx+vy*vy)+1e-6
    vx2 = vx + (A[:, 0]-DRAG*vx*sp)*DT
    vy2 = vy + (A[:, 1]-DRAG*vy*sp-CURRENT)*DT
    return np.stack([x+vx2*DT, y+vy2*DT, vx2, vy2], 1)


def obst_pen(P):                                               # obstacle-crossing penalty, over (Kp,H,2)
    pen = np.zeros(P.shape[:-1])
    for cx, cy, r in OBST:
        pen += np.maximum(0, r-np.sqrt((P[..., 0]-cx)**2+(P[..., 1]-cy)**2))
    return pen.sum(-1)


# ─── 1. BASELINES: random-feature world model + oracle ──────────────────────
class WorldModelRandom:
    """Baseline: learns the full Δstate via random Fourier features cos(...) on [state,action].
    Good one-step R², but the multi-step rollout drifts (see world_scale.py)."""
    def __init__(self, rng, K=1200):
        self.R = rng.standard_normal((6, K))*1.5; self.b = rng.standard_normal(K); self.K = K

    def phi(self, X): return np.cos((X-self.mu)/self.sd @ self.R + self.b)

    def fit(self, S, A, Snext):
        SA = np.hstack([S, A]); self.mu, self.sd = SA.mean(0), SA.std(0)+1e-9
        P = self.phi(SA); self.W = np.linalg.solve(P.T@P + np.eye(self.K), P.T@(Snext-S))

    def step(self, S, A): return S + self.phi(np.hstack([S, A])) @ self.W


class Oracle:
    """Model = true physics (planning upper bound)."""
    def fit(self, *a): pass
    def step(self, S, A): return true_step(S, A)


# ─── 2. STRUCTURED world model (the official one) ───────────────────────────
class WorldModelStructured:
    """IMPOSE p' = p + v'·dt and learn only Δv, on an INVARIANT basis
    φ = [1, p, v, a, v·||v||]. Closed form W=(ΦᵀΦ+λI)⁻¹ΦᵀΔv. Scales to 64D (world_scale_solution.py)."""
    def features(self, S, A):
        p, v = S[:, :2], S[:, 2:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
        return np.hstack([np.ones((S.shape[0], 1)), p, v, A, v*sp])

    def fit(self, S, A, Snext):
        F = self.features(S, A); dV = (Snext-S)[:, 2:]
        self.W = np.linalg.solve(F.T@F + 1e-6*np.eye(F.shape[1]), F.T@dV)

    def step(self, S, A):
        v2 = S[:, 2:] + self.features(S, A) @ self.W
        return np.hstack([S[:, :2]+v2*DT, v2])


# ─── online exploration + planning ──────────────────────────────────────────
def explore(rng, n=2500):
    """The agent acts at random and observes transitions (state, action → next state)."""
    S, A, Snext = [], [], []; s = np.array([[0., 0., 0., 0.]])
    for _ in range(n):
        a = rng.uniform(-1, 1, 2); s2 = true_step(s, a[None])
        S.append(s[0]); A.append(a); Snext.append(s2[0]); s = s2
        if s[0, :2].max() > 4 or s[0, :2].min() < -4:
            s = np.array([[0., 0., 0., 0.]])
    return np.array(S), np.array(A), np.array(Snext)


def smooth_plan(model, s, goal, rng, Kp=1000, H=20):
    """Gradient-free predictive control: sample CORRELATED action sequences (sustained direction
    + noise), roll them out in the model, keep the one that gets closest (argmin)."""
    base = rng.uniform(-1, 1, (Kp, 1, 2))
    seqs = np.clip(base + rng.uniform(-0.3, 0.3, (Kp, H, 2)), -1, 1)
    st = np.tile(s, (Kp, 1)).astype(float); P = np.empty((Kp, H, 2))
    for t in range(H):
        st = model.step(st, seqs[:, t]); P[:, t] = st[:, :2]
    dfin = np.linalg.norm(P[:, -3:]-goal, axis=2).min(1)
    dmean = np.linalg.norm(P-goal, axis=2).mean(1)
    return seqs[(dfin + 0.3*dmean + 2.0*obst_pen(P)).argmin(), 0]


# ─── 3. EVALUATION ──────────────────────────────────────────────────────────
def one_step_r2(model, rng):
    St = rng.uniform(-3, 3, (3000, 4)); St[:, 2:] = rng.uniform(-2, 2, (3000, 2)); At = rng.uniform(-1, 1, (3000, 2))
    tr = true_step(St, At); pr = model.step(St, At)
    return 1 - ((tr-pr)**2).sum()/((tr-tr.mean(0))**2).sum()


def rollout_err(model, rng, H=20, B=200):
    Z = rng.uniform(-2, 2, (B, 4)); Z[:, 2:] = rng.uniform(-1, 1, (B, 2)); a, b = Z.copy(), Z.copy()
    seq = rng.uniform(-1, 1, (B, H, 2))
    for t in range(H):
        a = true_step(a, seq[:, t]); b = model.step(b, seq[:, t])
    return np.linalg.norm((a-b)[:, :2], axis=1).mean()


def tour(model, rng, use_planner=True):
    s = np.array([[0., 0., 0., 0.]]); reached = hits = gi = t = 0; mind = [9.9]*len(GOALS)
    while gi < len(GOALS) and t < 300:
        a = smooth_plan(model, s, GOALS[gi], rng) if use_planner else rng.uniform(-1, 1, 2)
        s = true_step(s, a[None]); t += 1
        for cx, cy, r in OBST:
            if (s[0, 0]-cx)**2+(s[0, 1]-cy)**2 < r*r:
                hits += 1
        d = np.linalg.norm(s[0, :2]-GOALS[gi]); mind[gi] = min(mind[gi], d)
        if d < 0.6:
            reached += 1; gi += 1
    return reached, hits, mind


def main():
    rng = np.random.default_rng(0)
    print("=== CANONICAL AGENT — gradient-free, structured world model (no GPU) ===\n")
    print("  1·EXPLORE online (2500 transitions, random actions) then LEARN in closed form.\n")
    S, A, Snext = explore(rng)
    models = {"random (cos)": WorldModelRandom(rng), "STRUCTURED": WorldModelStructured(), "oracle": Oracle()}
    for m in models.values():
        m.fit(S, A, Snext)

    print("  2·FIDELITY of the learned world model:")
    print(f"     {'model':18} | {'R² 1-step':>9} | {'rollout err @20':>15}")
    print(f"     {'-'*18}-+-{'-'*9}-+-{'-'*15}")
    for name, m in models.items():
        print(f"     {name:18} | {one_step_r2(m, np.random.default_rng(3)):>9.4f} | {rollout_err(m, np.random.default_rng(4)):>15.4f}")

    print("\n  3·TOUR: 3 goals, 3 obstacles (planner = smooth, gradient-free):")
    print(f"     {'strategy':22} | {'goals':>5} | {'dist per goal':>18} | {'hits':>4}")
    print(f"     {'-'*22}-+-{'-'*5}-+-{'-'*18}-+-{'-'*4}")
    rows = [("random actions", models["STRUCTURED"], False)] + [(f"model {n}", m, True) for n, m in models.items()]
    for name, m, pl in rows:
        r, h, md = tour(m, np.random.default_rng(1), use_planner=pl)
        flag = " ✅" if r == 3 else ""
        print(f"     {name:22} | {r}/3{flag:>2} | {' '.join(f'{x:4.2f}' for x in md):>18} | {h:>4}")

    print("\n  → random model 0/3, structured model 3/3, oracle 3/3.")
    print("    Scaling does not come from more random features, but from a BASIS that respects")
    print("    the causal structure of the world. Closed form, gradient-free, no GPU.")


if __name__ == "__main__":
    main()
