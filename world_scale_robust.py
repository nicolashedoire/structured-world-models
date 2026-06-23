"""world_scale_robust — the HONEST version of the structured model. We do NOT tell the model that
the right feature is v·||v||. We give it a GENERIC library with DECOYS:
  φ = [1, p, v, a, v·||v||, v·||v||², p², v², p⊙v]
and it must recover the dynamics on its own by gradient-free selection (SINDy / sequential
thresholded least squares — STLSQ). We measure: R², rollout, AND recovery (does it keep v·||v|| and
p,a? does it drop the decoys?). If yes and it scales to 64D → the solution holds WITHOUT cheating.
Closed form, no gradient, no GPU.

  python3 world_scale_robust.py
"""
from __future__ import annotations
import numpy as np

DT = 0.2
# library blocks (name, function) — each returns (N, n) except '1' (N,1)
BLOCKS = ["1", "p", "v", "a", "v|v|", "v|v|^2", "p^2", "v^2", "p*v"]
TRUE = {"p", "a", "v|v|"}                                   # the true generators (linear coupling p, action a, drag v|v|)


def make_world(n, rng):
    C = rng.standard_normal((n, n)) / np.sqrt(n) * 0.5; drag = 0.4

    def step(Z, A):
        p, v = Z[:, :n], Z[:, n:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
        v2 = v + (A - drag*v*sp - p @ C.T)*DT; p2 = p + v2*DT
        return np.hstack([p2, v2])
    return step, C


def library(Z, A, n):
    p, v = Z[:, :n], Z[:, n:]; sp = np.linalg.norm(v, axis=1, keepdims=True)+1e-6
    cols = {"1": np.ones((Z.shape[0], 1)), "p": p, "v": v, "a": A,
            "v|v|": v*sp, "v|v|^2": v*sp*sp, "p^2": p*p, "v^2": v*v, "p*v": p*v}
    F = np.hstack([cols[b] for b in BLOCKS])
    spans = []; i = 0                                      # column range of each block
    for b in BLOCKS:
        w = cols[b].shape[1]; spans.append((b, i, i+w)); i += w
    return F, spans


def stlsq(F, Y, lam=1e-4, thresh=0.05, iters=8):
    """Sequential thresholded least squares (SINDy), per output — gradient-free."""
    P = F.shape[1]; W = np.zeros((P, Y.shape[1]))
    Fn = F / (np.linalg.norm(F, axis=0, keepdims=True)+1e-9)  # normalize for fair thresholding
    for j in range(Y.shape[1]):
        active = np.ones(P, bool)
        for _ in range(iters):
            Fa = Fn[:, active]
            w = np.linalg.solve(Fa.T@Fa + lam*np.eye(Fa.shape[1]), Fa.T@Y[:, j])
            big = np.abs(w) > thresh
            if big.all() or not big.any():
                break
            idx = np.where(active)[0]; active[idx[~big]] = False
        idx = np.where(active)[0]
        Fa = Fn[:, active]
        W[idx, j] = np.linalg.solve(Fa.T@Fa + lam*np.eye(Fa.shape[1]), Fa.T@Y[:, j])
    return W  # coefficients in the NORMALIZED frame (reconvert by /colnorm)


def main():
    print("=== HONEST structured model — generic library + decoys + SINDy selection ===\n")
    print(f"  library: {BLOCKS}   (true generators: {sorted(TRUE)})\n")
    print(f"  {'dim':>9} | {'#lib':>5} | {'RIDGE full':>18} | {'SINDy (selection)':>30}")
    print(f"  {'world':>9} | {'feat':>5} | {'R² · rollout@20':>18} | {'R² · rollout · kept blocks':>30}")
    print(f"  {'-'*9}-+-{'-'*5}-+-{'-'*18}-+-{'-'*30}")
    for n in (16, 32, 64):
        rng = np.random.default_rng(200+n); step, C = make_world(n, rng)
        F0 = library(np.zeros((1, 2*n)), np.zeros((1, n)), n)[0].shape[1]; N = 30*F0
        Z = rng.uniform(-2, 2, (N, 2*n)); A = rng.uniform(-1, 1, (N, n))
        Y = step(Z, A)[:, n:] - Z[:, n:]
        F, spans = library(Z, A, n)

        def mk_model(W):
            def model(Z, A):
                p, v = Z[:, :n], Z[:, n:]; v2 = v + library(Z, A, n)[0] @ W; return np.hstack([p+v2*DT, v2])
            return model

        def evalm(model):
            Zt = rng.uniform(-2, 2, (3000, 2*n)); At = rng.uniform(-1, 1, (3000, n))
            tr = step(Zt, At); pr = model(Zt, At)
            R2 = 1 - ((tr-pr)**2).sum()/((tr-tr.mean(0))**2).sum()
            Zr = rng.uniform(-2, 2, (200, 2*n)); a, bb = Zr.copy(), Zr.copy(); sq = rng.uniform(-1, 1, (200, 20, n))
            for t in range(20):
                a = step(a, sq[:, t]); bb = model(bb, sq[:, t])
            return R2, np.linalg.norm((a-bb)[:, :n], axis=1).mean()

        # full RIDGE (no selection)
        Wr = np.linalg.solve(F.T@F + 1e-3*np.eye(F.shape[1]), F.T@Y)
        R2r, ehr = evalm(mk_model(Wr))
        # SINDy
        Wn = stlsq(F, Y); W = Wn / (np.linalg.norm(F, axis=0)+1e-9)[:, None]
        R2s, ehs = evalm(mk_model(W))
        kept = [b for (b, i, j) in spans if np.abs(W[i:j]).max() > 1e-6]
        kept_str = "+".join(k for k in kept if k != "1")
        print(f"  {'%dD' % n:>9} | {F0:>5} | {('%.4f · %.3f' % (R2r, ehr)):>18} | {('%.4f · %.3f · %s' % (R2s, ehs, kept_str)):>30}")
    print("\n  → If SINDy keeps {p, a, v|v|} and drops the decoys, AND scales to 64D: the solution holds")
    print("    WITHOUT being told the right feature. Full ridge also works but does NOT SELECT")
    print("    (keeps the decoys → less interpretable, more fragile out-of-distribution). All gradient-free.")


if __name__ == "__main__":
    main()
