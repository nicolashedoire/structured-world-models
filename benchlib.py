"""benchlib — shared library for the gradient-free control benchmarks.
Dynamics (Gym-like) · world models (structured / random / oracle) · CEM planner with
warm-start (cross-entropy method, receding horizon) · benchmark config. No gradient, no GPU.

Two project rules, encoded here:
  1) STRUCTURE > random features   (model: impose pos'=pos+vel·dt, learn only Δvel)
  2) ORACLE FIRST                  (verify CEM+oracle solves the task BEFORE judging models)

Imported by: planner_cem_solution.py · model_quality.py · agent_cem_benchmark.py
"""
from __future__ import annotations
import numpy as np


def angnorm(x): return (x + np.pi) % (2*np.pi) - np.pi


# ─── true dynamics (actions normalized to [-1,1]) ───────────────────────────
def pendulum_step(S, A):
    th, thd = S[:, 0], S[:, 1]; u = 2*np.clip(A[:, 0], -1, 1)            # torque ±2
    thd2 = np.clip(thd + (15*np.sin(th) + 3*u)*0.05, -8, 8)
    return np.stack([th + thd2*0.05, thd2], 1)


def mcar_step(S, A):
    x, v = S[:, 0], S[:, 1]; a = np.clip(A[:, 0], -1, 1)
    v2 = np.clip(v + a*0.0015 - 0.0025*np.cos(3*x), -0.07, 0.07)
    x2 = np.clip(x + v2, -1.2, 0.6); v2 = np.where((x2 <= -1.2) & (v2 < 0), 0, v2)
    return np.stack([x2, v2], 1)


def cartpole_step(S, A):
    x, th, xd, thd = S[:, 0], S[:, 1], S[:, 2], S[:, 3]; f = 10*np.clip(A[:, 0], -1, 1)
    ct, st = np.cos(th), np.sin(th)
    temp = (f + 0.05*thd**2*st)/1.1
    thacc = (9.8*st - ct*temp)/(0.5*(4/3 - 0.1*ct**2/1.1))
    xacc = temp - 0.05*thacc*ct/1.1
    return np.stack([x+0.02*xd, th+0.02*thd, xd+0.02*xacc, thd+0.02*thacc], 1)


HN = 6
HC = np.random.default_rng(99).standard_normal((HN, HN))/np.sqrt(HN)*0.15   # WEAK coupling: easy to drive, still 12-D
HTARGET = np.ones(HN)*0.8                                    # reachable target (validated by bench 1 / oracle)


def highd_step(S, A):
    p, v = S[:, :HN], S[:, HN:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
    v2 = v + (np.clip(A, -1, 1) - 0.4*v*sp - p@HC.T)*0.2
    return np.hstack([p + v2*0.2, v2])


# ─── STRUCTURED bases (known physics) ───────────────────────────────────────
def pendulum_feat(S, A):
    th, thd = S[:, 0:1], S[:, 1:2]
    return np.hstack([np.ones_like(th), np.sin(th), np.cos(th), thd, A[:, 0:1]])


def mcar_feat(S, A):
    x = S[:, 0:1]
    return np.hstack([np.ones_like(x), np.cos(3*x), np.sin(3*x), x, A[:, 0:1]])


def cartpole_feat(S, A):
    th, xd, thd = S[:, 1:2], S[:, 2:3], S[:, 3:4]; st, ct = np.sin(th), np.cos(th)
    return np.hstack([np.ones_like(th), st, ct, thd, thd**2, st*thd**2, ct*thd**2,
                      xd, A[:, 0:1], A[:, 0:1]*ct, st*ct])


def highd_feat(S, A):
    p, v = S[:, :HN], S[:, HN:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
    return np.hstack([np.ones((S.shape[0], 1)), p, v, np.clip(A, -1, 1), v*sp])


# ─── world models ───────────────────────────────────────────────────────────
class Structured:
    def __init__(self, feat, pos, vel, dt, semi=True, vclip=None):
        self.feat, self.pos, self.vel, self.dt, self.semi, self.vclip = feat, pos, vel, dt, semi, vclip

    def fit(self, S, A, Snext):
        F = self.feat(S, A); self.W = np.linalg.solve(F.T@F + 1e-6*np.eye(F.shape[1]), F.T@(Snext-S)[:, self.vel])

    def step(self, S, A):
        vel2 = S[:, self.vel] + self.feat(S, A)@self.W
        if self.vclip is not None:
            vel2 = np.clip(vel2, *self.vclip)
        S2 = S.copy(); S2[:, self.vel] = vel2
        S2[:, self.pos] = S[:, self.pos] + (vel2 if self.semi else S[:, self.vel])*self.dt
        return S2


class RandomFeat:
    def __init__(self, rng, d, K=1000):
        self.R = rng.standard_normal((d, K))/np.sqrt(d)*2.0; self.b = rng.standard_normal(K); self.K = K

    def fit(self, S, A, Snext):
        X = np.hstack([S, A]); self.mu, self.sd = X.mean(0), X.std(0)+1e-9
        P = self.phi(X); self.W = np.linalg.solve(P.T@P + np.eye(self.K), P.T@(Snext-S))

    def phi(self, X): return np.cos((X-self.mu)/self.sd @ self.R + self.b)

    def step(self, S, A): return S + self.phi(np.hstack([S, A]))@self.W


class Oracle:
    def __init__(self, step): self.s = step
    def fit(self, *a): pass
    def step(self, S, A): return self.s(S, A)


# ─── CEM planner (cross-entropy method) + vectorized rollout ────────────────
def rollout(model, s, seqs):
    K, H, _ = seqs.shape; st = np.tile(s, (K, 1)).astype(float); ST = np.empty((K, H, s.shape[1]))
    for t in range(H):
        st = model.step(st, seqs[:, t]); ST[:, t] = st
    return ST


def cem_plan(model, s, cost, rng, adim, H, low, high,
             K=400, elites=48, iters=5, alpha=0.25, init_mean=None, init_std=None):
    low = np.asarray(low, float); high = np.asarray(high, float)
    mean = np.zeros((H, adim)) if init_mean is None else init_mean.copy()
    std = np.ones((H, adim))*((high-low)/2) if init_std is None else init_std.copy()
    best_seq, best_score = None, np.inf
    for _ in range(iters):
        seqs = np.clip(rng.normal(mean, std, (K, H, adim)), low, high)
        scores = cost(rollout(model, s, seqs))
        idx = np.argpartition(scores, elites)[:elites]; elite = seqs[idx]
        mean = alpha*mean + (1-alpha)*elite.mean(0); std = alpha*std + (1-alpha)*(elite.std(0)+1e-3)
        j = idx[np.argmin(scores[idx])]
        if scores[j] < best_score:
            best_score, best_seq = scores[j], seqs[j]
    return best_seq[0], best_seq, mean, std, best_score


# ─── benchmark config ───────────────────────────────────────────────────────
BENCH = {
    "Pendulum": dict(
        step=pendulum_step, feat=pendulum_feat, pos=[0], vel=[1], dt=0.05, semi=True, vclip=(-8, 8),
        lo=[-np.pi, -8], hi=[np.pi, 8], adim=1, H=30, K=400, elites=40, iters=5, steps=200,
        s0=np.array([[np.pi, 0.]]),
        cost=lambda ST: (angnorm(ST[:, :, 0])**2 + 0.1*ST[:, :, 1]**2).sum(1),
        metric=lambda tr: (np.abs(angnorm(tr[-20:, 0])) < 0.25).mean(),
        fmt=lambda m: f"{m*100:3.0f}% of time UPRIGHT (end)", win=lambda m: m > 0.7),
    "MountainCar": dict(
        step=mcar_step, feat=mcar_feat, pos=[0], vel=[1], dt=1.0, semi=True, vclip=(-0.07, 0.07),
        lo=[-1.2, -0.07], hi=[0.6, 0.07], adim=1, H=80, K=350, elites=35, iters=4, steps=200,
        s0=np.array([[-0.5, 0.]]),
        cost=lambda ST: 0.6 - ST[:, :, 0].max(1),
        metric=lambda tr: tr[:, 0].max(),
        fmt=lambda m: f"x_max = {m:+.2f}  (flag 0.45)", win=lambda m: m >= 0.45),
    "CartPole": dict(
        step=cartpole_step, feat=cartpole_feat, pos=[0, 1], vel=[2, 3], dt=0.02, semi=False, vclip=None,
        lo=[-2, -0.5, -2, -2], hi=[2, 0.5, 2, 2], adim=1, H=18, K=400, elites=40, iters=5, steps=200,
        s0=np.array([[0., 0.05, 0., 0.]]),
        cost=lambda ST: (ST[:, :, 1]**2 + 0.05*ST[:, :, 0]**2).sum(1),
        metric=None, fmt=lambda m: f"{int(m):3d}/200 steps balanced", win=lambda m: m >= 195),
    "High-D reach (12-D)": dict(
        step=highd_step, feat=highd_feat, pos=list(range(HN)), vel=list(range(HN, 2*HN)), dt=0.2, semi=True, vclip=None,
        lo=[-2]*(2*HN), hi=[2]*(2*HN), adim=HN, H=25, K=600, elites=60, iters=6, steps=70,
        s0=np.zeros((1, 2*HN)),
        cost=lambda ST: np.linalg.norm(ST[:, -3:, :HN]-HTARGET, axis=2).min(1) + 0.3*np.linalg.norm(ST[:, :, :HN]-HTARGET, axis=2).mean(1),
        metric=lambda tr: np.linalg.norm(tr[:, :HN]-HTARGET, axis=1).min(),
        fmt=lambda m: f"min dist to target = {m:.2f}", win=lambda m: m < 1.0),   # threshold calibrated on the oracle (~0.83)
}


def explore(cfg, rng, N=8000):
    S = rng.uniform(cfg["lo"], cfg["hi"], (N, len(cfg["lo"]))); A = rng.uniform(-1, 1, (N, cfg["adim"]))
    return S, A, cfg["step"](S, A)


def run_episode(name, cfg, model, rng, planner=True):
    """Receding-horizon MPC with CEM + WARM-START (re-center mean from one step to the next)."""
    s = cfg["s0"].copy(); traj = [s[0].copy()]; balanced = 0
    low = -np.ones(cfg["adim"]); high = np.ones(cfg["adim"]); mean = None
    for _ in range(cfg["steps"]):
        if planner:
            a, _, mean, _, _ = cem_plan(model, s, cfg["cost"], rng, cfg["adim"], cfg["H"], low, high,
                                        K=cfg["K"], elites=cfg["elites"], iters=cfg["iters"], init_mean=mean)
            mean = np.vstack([mean[1:], mean[-1:]])             # shift the warm-start
        else:
            a = rng.uniform(-1, 1, cfg["adim"])
        s = cfg["step"](s, a[None]); traj.append(s[0].copy())
        if name.startswith("CartPole"):
            if abs(s[0, 1]) < 0.21 and abs(s[0, 0]) < 2.4:
                balanced += 1
            else:
                break
    return balanced if name.startswith("CartPole") else cfg["metric"](np.array(traj))
