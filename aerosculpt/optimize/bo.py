"""Bayesian optimization for expensive, noisy CFD objectives.

A Gaussian process (Matern 5/2, one length scale per variable, fitted noise) models the
objective on the unit cube; each iteration proposes the feasible point with the largest
expected improvement. Every proposal costs one CFD run, so "iterations" in the product
means exactly "CFD evaluations after the baseline".
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

SQRT5 = math.sqrt(5.0)


def _matern52(A, B, ls, sf2):
    d = (A[:, None, :] - B[None, :, :]) / ls
    r = np.sqrt(np.maximum((d * d).sum(-1), 1e-30))
    return sf2 * (1.0 + SQRT5 * r + 5.0 / 3.0 * r * r) * np.exp(-SQRT5 * r)


class GP:
    def __init__(self, X, y, noise_floor=1e-4, seed=0):
        self.X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.ym = float(y.mean())
        self.ys = float(y.std()) or 1.0
        self.yn = (y - self.ym) / self.ys
        self.noise_floor = noise_floor
        self._fit(np.random.default_rng(seed))

    def _unpack(self, th):
        d = self.X.shape[1]
        ls = np.exp(th[:d])
        sf2 = math.exp(th[d])
        sn2 = math.exp(th[d + 1]) + self.noise_floor
        return ls, sf2, sn2

    def _nll(self, th):
        ls, sf2, sn2 = self._unpack(th)
        n = len(self.yn)
        K = _matern52(self.X, self.X, ls, sf2) + sn2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.yn))
        nll = 0.5 * self.yn @ a + np.log(np.diag(L)).sum() + 0.5 * n * math.log(2 * math.pi)
        # weak priors keep length scales and noise in a sane range with few points
        nll += 0.5 * ((th[:-2] - math.log(0.4)) ** 2).sum() / 1.5 ** 2
        nll += 0.5 * ((th[-1] - math.log(0.05)) ** 2) / 2.0 ** 2
        return float(nll)

    def _fit(self, rng):
        d = self.X.shape[1]
        best = None
        starts = [np.r_[np.full(d, math.log(0.4)), 0.0, math.log(0.05)]]
        for _ in range(3):
            starts.append(np.r_[rng.uniform(math.log(0.1), math.log(2.0), d), rng.uniform(-1, 1),
                                rng.uniform(math.log(1e-3), math.log(0.3))])
        bounds = [(math.log(0.03), math.log(10.0))] * d + [(-4.0, 3.0), (math.log(1e-6), math.log(1.0))]
        for th0 in starts:
            r = minimize(self._nll, th0, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        ls, sf2, sn2 = self._unpack(self.theta)
        self.ls, self.sf2, self.sn2 = ls, sf2, sn2
        K = _matern52(self.X, self.X, ls, sf2) + sn2 * np.eye(len(self.yn))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.yn))

    def predict(self, Xs):
        """Posterior mean and standard deviation in original units."""
        Xs = np.atleast_2d(Xs)
        Ks = _matern52(Xs, self.X, self.ls, self.sf2)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.sf2 - (v * v).sum(0), 1e-12)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys

    def relevance(self):
        """1 / length scale, normalised: which variables the objective is sensitive to."""
        r = 1.0 / self.ls
        return r / r.sum()


def expected_improvement(mu, sd, best, xi=0.0):
    imp = best - mu - xi
    z = imp / np.maximum(sd, 1e-12)
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def initial_design(d, n, seed=0):
    if n <= 0:
        return np.zeros((0, d))
    s = qmc.LatinHypercube(d=d, seed=seed)
    return s.random(n)


def propose(X, y, feasible, rng, n_candidates=3000, gp=None):
    """Next point to evaluate: maximise expected improvement over feasible candidates.

    feasible(x) -> bool is a cheap geometric check (volume, limits); it is applied to the
    best-ranked candidates only, in order, so expensive checks stay rare.
    """
    X = np.asarray(X, float)
    d = X.shape[1]
    gp = gp or GP(X, y, seed=int(rng.integers(1 << 30)))
    best_i = int(np.argmin(y))
    cand = [rng.random((n_candidates, d))]
    # local perturbations around the incumbents exploit what the model already believes
    order = np.argsort(y)[:3]
    for i in order:
        for scale in (0.03, 0.08, 0.18):
            cand.append(np.clip(X[i] + rng.normal(0, scale, (n_candidates // 10, d)), 0, 1))
    C = np.vstack(cand)
    mu, sd = gp.predict(C)
    ei = expected_improvement(mu, sd, float(np.min(y)), xi=0.01 * float(np.std(y) or 1.0))
    # refine the few best with a local optimiser
    top = np.argsort(-ei)[:5]
    extra = []
    for i in top:
        f = lambda x: -float(expected_improvement(*gp.predict(x[None]), float(np.min(y)))[0])
        r = minimize(f, C[i], method="L-BFGS-B", bounds=[(0, 1)] * d, options={"maxiter": 40})
        extra.append(r.x)
    C = np.vstack([C, np.array(extra)])
    mu, sd = gp.predict(C)
    ei = expected_improvement(mu, sd, float(np.min(y)))
    for i in np.argsort(-ei)[:400]:
        x = C[i]
        if np.min(np.linalg.norm(X - x, axis=1)) < 1e-3:
            continue  # do not waste a CFD run on a point already evaluated
        if feasible(x):
            return x, float(ei[i]), gp
    # nothing feasible near the EI optimum: fall back to a random feasible point
    for _ in range(2000):
        x = rng.random(d)
        if feasible(x):
            return x, 0.0, gp
    return X[best_i], 0.0, gp
