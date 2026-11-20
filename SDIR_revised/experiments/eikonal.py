"""Eikonal travel-time problem: fast marching (first order) with exact discrete adjoint.

Slowness s = exp(u) on an n x n grid over [0,1]^2, point source, |grad T| = s.
G(u) = (T(x_1), ..., T(x_K)) at receiver nodes. G'(u)^* w is computed by a reverse sweep
through the acceptance order of fast marching (exact derivative of the discrete map).
"""
import heapq

import numpy as np
from numba import njit


@njit(cache=True)
def _local(Ta, Tb, hs):
    # returns T, dT/dTa, dT/dTb, dT/dhs ; Ta, Tb may be inf
    if Tb < Ta:
        Ta, Tb = Tb, Ta
        swap = True
    else:
        swap = False
    if Tb == np.inf or Tb - Ta >= hs:
        T, da, db, dh = Ta + hs, 1.0, 0.0, 1.0
    else:
        d = Ta - Tb
        D = np.sqrt(2.0 * hs * hs - d * d)
        T = 0.5 * (Ta + Tb + D)
        da = 0.5 * (1.0 - d / D)
        db = 0.5 * (1.0 + d / D)
        dh = hs / D
    if swap:
        return T, db, da, dh
    return T, da, db, dh


@njit(cache=True)
def fmm(s, h, src_i, src_j, r0=0.0):
    n1, n2 = s.shape
    N = n1 * n2
    T = np.full(N, np.inf)
    state = np.zeros(N, np.int8)          # 0 far, 1 band, 2 accepted
    order = np.empty(N, np.int64)
    pa = np.full(N, -1, np.int64)          # parent in x
    pb = np.full(N, -1, np.int64)          # parent in y
    ca = np.zeros(N)
    cb = np.zeros(N)
    cs = np.zeros(N)                       # dT/ds_{sidx[k]}
    sidx = np.arange(N)
    # heap as arrays (simple binary heap of (T, idx))
    hv = np.empty(8 * N)
    hi = np.empty(8 * N, np.int64)
    hn = 0
    k0 = src_i * n2 + src_j
    hn = 0
    cnt = 0
    # exact initialisation in a disc of radius r0 around the source (constant slowness there)
    R = int(r0 / h) + 1
    for i in range(max(0, src_i - R), min(n1, src_i + R + 1)):
        for j in range(max(0, src_j - R), min(n2, src_j + R + 1)):
            dist = h * np.sqrt((i - src_i) ** 2 + (j - src_j) ** 2)
            if dist <= r0 or (i == src_i and j == src_j):
                q = i * n2 + j
                T[q] = dist * s[src_i, src_j]
                cs[q] = dist
                sidx[q] = k0
                state[q] = 1
                hv[hn] = T[q]
                hi[hn] = q
                c = hn
                hn += 1
                while c > 0:
                    p = (c - 1) // 2
                    if hv[p] <= hv[c]:
                        break
                    hv[p], hv[c] = hv[c], hv[p]
                    hi[p], hi[c] = hi[c], hi[p]
                    c = p
    while hn > 0:
        # pop min
        v = hv[0]
        k = hi[0]
        hn -= 1
        if hn > 0:
            hv[0] = hv[hn]
            hi[0] = hi[hn]
            p = 0
            while True:
                l = 2 * p + 1
                r = l + 1
                m = p
                if l < hn and hv[l] < hv[m]:
                    m = l
                if r < hn and hv[r] < hv[m]:
                    m = r
                if m == p:
                    break
                hv[p], hv[m] = hv[m], hv[p]
                hi[p], hi[m] = hi[m], hi[p]
                p = m
        if state[k] == 2 or v > T[k]:
            continue
        state[k] = 2
        order[cnt] = k
        cnt += 1
        i = k // n2
        j = k % n2
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii = i + di
            jj = j + dj
            if ii < 0 or ii >= n1 or jj < 0 or jj >= n2:
                continue
            q = ii * n2 + jj
            if state[q] == 2 or sidx[q] != q:
                continue
            # upwind neighbours among accepted
            ax = -1
            Ta = np.inf
            for d2 in (-1, 1):
                i2 = ii + d2
                if 0 <= i2 < n1:
                    q2 = i2 * n2 + jj
                    if state[q2] == 2 and T[q2] < Ta:
                        Ta = T[q2]
                        ax = q2
            ay = -1
            Tb = np.inf
            for d2 in (-1, 1):
                j2 = jj + d2
                if 0 <= j2 < n2:
                    q2 = ii * n2 + j2
                    if state[q2] == 2 and T[q2] < Tb:
                        Tb = T[q2]
                        ay = q2
            hs = h * s[ii, jj]
            Tn, da, db, dh = _local(Ta, Tb, hs)
            if Tn < T[q]:
                T[q] = Tn
                pa[q] = ax
                pb[q] = ay
                ca[q] = da if ax >= 0 else 0.0
                cb[q] = db if ay >= 0 else 0.0
                cs[q] = dh * h
                state[q] = 1
                # push
                hv[hn] = Tn
                hi[hn] = q
                c = hn
                hn += 1
                while c > 0:
                    p = (c - 1) // 2
                    if hv[p] <= hv[c]:
                        break
                    hv[p], hv[c] = hv[c], hv[p]
                    hi[p], hi[c] = hi[c], hi[p]
                    c = p
    return T, order[:cnt], pa, pb, ca, cb, cs, sidx


@njit(cache=True)
def adjoint_sweep(lam0, order, pa, pb, ca, cb, cs, sidx):
    lam = lam0.copy()
    g = np.zeros(lam.size)
    for t in range(order.size - 1, -1, -1):
        k = order[t]
        lk = lam[k]
        if lk == 0.0:
            continue
        g[sidx[k]] += lk * cs[k]
        if pa[k] >= 0:
            lam[pa[k]] += lk * ca[k]
        if pb[k] >= 0:
            lam[pb[k]] += lk * cb[k]
    return g


def make_true_logslowness(x, y):
    u = np.zeros_like(x)
    u[(x - 0.28) ** 2 + (y - 0.70) ** 2 <= 0.13 ** 2] = 0.5          # slow inclusion
    u[(np.abs(x - 0.72) <= 0.12) & (np.abs(y - 0.30) <= 0.12)] = -0.4  # fast inclusion
    return u


class Eikonal:
    def __init__(self, n=65, K=125, refine=2, seed=7, src=(0.5, 0.5), r0=0.08):
        self.n, self.h, self.r0 = n, 1.0 / (n - 1), r0
        g = np.linspace(0, 1, n)
        self.X, self.Y = np.meshgrid(g, g, indexing='ij')
        self.src = (int(round(src[0] * (n - 1))), int(round(src[1] * (n - 1))))
        rng = np.random.default_rng(seed)
        cand = np.array([k for k in range(n * n) if k != self.src[0] * n + self.src[1]])
        self.rec = np.sort(rng.choice(cand, K, replace=False))
        self.u_true = make_true_logslowness(self.X, self.Y)
        # data on a finer grid (same physical receiver locations)
        nf = refine * (n - 1) + 1
        gf = np.linspace(0, 1, nf)
        Xf, Yf = np.meshgrid(gf, gf, indexing='ij')
        uf = make_true_logslowness(Xf, Yf)
        Tf, *_ = fmm(np.exp(uf), 1.0 / (nf - 1), refine * self.src[0], refine * self.src[1], r0)
        ri, rj = self.rec // n, self.rec % n
        self.v_exact_fine = Tf[(refine * ri) * nf + refine * rj]
        self._cache = None

    def _solve(self, u):
        if self._cache is not None and self._cache[0] is u:
            return self._cache[1]
        out = fmm(np.exp(u), self.h, self.src[0], self.src[1], self.r0)
        self._cache = (u, out)
        return out

    def forward(self, u):
        T = self._solve(u)[0]
        return T[self.rec]

    def jac_adj(self, u, w):
        T, order, pa, pb, ca, cb, cs, sidx = self._solve(u)
        lam = np.zeros(self.n * self.n)
        np.add.at(lam, self.rec, w)
        g = adjoint_sweep(lam, order, pa, pb, ca, cb, cs, sidx)
        return (g * np.exp(u).ravel()).reshape(self.n, self.n)
