"""Shared experiment core: measurement streams, methods (SDIR, Landweber, RED, PnP) and stopping rules.

Step sizes (no rescaling by ||G'||):
    mu_n     = min{ gamma0 ||r|| / ||G'(u)^* r||, gamma1 }
    lambda_n = min{ nu0 ||r||^2 / ||Psi_a||, nu1 / ||Psi_a||, nu2 },  Psi_a = alpha_n (u - D(u)).
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np


@dataclasses.dataclass
class P:
    gamma0: float = 0.5
    gamma1: float = 1.5
    nu0: float = 0.1
    nu1: float = 0.9
    nu2: float = 1.0
    alpha0: float = 0.9
    s: float = 0.1
    tau0: float = 1.1
    rho: float = 0.1
    N0: int = 1000
    omega: float = 0.1      # RED weight
    beta: float = 0.5       # PnP averaging
    R: float = np.inf       # safeguard radius

    def tau(self, m):
        return self.tau0 * (1.0 + np.log(m)) ** self.rho

    def nmax(self, m):
        return int(round(self.N0 * max(1.0, np.sqrt(m / 64.0))))

    def alpha(self, n):
        return self.alpha0 * (n + 1.0) ** (-self.s)


class Stream:
    """Nested measurement stream with Welford updates; z_m from the data only."""

    def __init__(self, v, sig, seed):
        self.v, self.sig = v, sig
        self.rng = np.random.default_rng(seed)
        self.m, self.mean, self.M2 = 0, np.zeros_like(v), 0.0
        self.first = None

    def advance_to(self, m):
        while self.m < m:
            vi = self.v + self.sig * self.rng.standard_normal(self.v.shape)
            if self.first is None:
                self.first = vi.copy()
            self.m += 1
            d = vi - self.mean
            self.mean += d / self.m
            self.M2 += float(np.sum(d * (vi - self.mean)))
        z = np.sqrt(self.M2 / (self.m - 1)) if self.m > 1 else np.nan
        return self.mean.copy(), z

    @property
    def sigma(self):
        return self.sig * np.sqrt(self.v.size)


def step(prob, u, data, n, par, method):
    r = prob.forward(u) - data
    rn = np.linalg.norm(r)
    g = prob.jac_adj(u, r)
    gn = np.linalg.norm(g)
    mu = min(par.gamma0 * rn / gn, par.gamma1) if gn > 0 else par.gamma1
    info = {}
    if method == 'landweber':
        w = u - mu * g
    elif method == 'dddir':
        psi = par.alpha(n) * (u - prob.denoiser(u))
        pn = np.linalg.norm(psi)
        lam = min(par.nu0 * rn ** 2 / pn, par.nu1 / pn, par.nu2) if pn > 0 else 0.0
        w = u - mu * g - lam * psi
        info['ratio'] = lam * pn / max(mu * gn, 1e-300)
    elif method == 'red':
        w = u - mu * (g + par.omega * (u - prob.denoiser(u)))
    elif method == 'pnp':
        x = u - mu * g
        w = par.beta * prob.denoiser(x) + (1 - par.beta) * x
    else:
        raise ValueError(method)
    return w, rn, info


def run(prob, data, par, method, thr, nmax, refs=None, n_extra=0, keep_iter=()):
    """Iterate until ||G(u_n)-data|| <= thr or n = nmax; optionally continue n_extra steps.

    Returns dict with u_stop, n_stop, reason, err history (per ref), res history, ratios, kept iterates.
    """
    refs = refs or {}
    u = prob.u0.copy()
    out = dict(res=[], ratio=[], err={k: [] for k in refs}, kept={})
    n, stopped = 0, False
    t0 = time.time()
    total = None
    while True:
        for k, ref in refs.items():
            out['err'][k].append(np.linalg.norm(u - ref) / np.linalg.norm(ref))
        if n in keep_iter:
            out['kept'][n] = u.copy()
        rn = np.linalg.norm(prob.forward(u) - data)
        if not stopped and (rn <= thr or n >= nmax):
            stopped = True
            out.update(u_stop=u.copy(), n_stop=n, reason='dp' if rn <= thr else 'budget',
                       time=time.time() - t0)
            total = n + n_extra
        if stopped and n >= total:
            out['res'].append(rn)
            break
        if np.linalg.norm(u - prob.u0) >= par.R:
            break
        w, rn, info = step(prob, u, data, n, par, method)
        out['res'].append(rn)
        if 'ratio' in info:
            out['ratio'].append(info['ratio'])
        u = w
        n += 1
    return out


def sig_entry(prob, rel=0.1):
    return rel * np.linalg.norm(prob.v_exact) / np.sqrt(prob.v_exact.size)


def psnr(u, ref):
    rng = ref.max() - ref.min()
    return 10 * np.log10(rng ** 2 / np.mean((u - ref) ** 2))
