"""Test problems: deblurring (phi = 1.5), phase retrieval (phaseless tomography), eikonal."""
import types

import numpy as np
from skimage import data as skdata
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import radon, resize

from eikonal import Eikonal


def tv_prox(weight, n_iter=30):
    return lambda u: denoise_tv_chambolle(u, weight=weight, max_num_iter=n_iter, eps=1e-12)


def deblurring(denoiser, phi=1.5, n=256):
    u = resize(skdata.camera().astype(float) / 255.0, (n, n), anti_aliasing=True)
    x = np.fft.fftfreq(n) * n
    X, Y = np.meshgrid(x, x, indexing='ij')
    k = np.exp(-(X ** 2 + Y ** 2) / (2 * phi ** 2))
    k /= k.sum()
    K = np.real(np.fft.fft2(k))

    def G(v):
        return np.real(np.fft.ifft2(K * np.fft.fft2(v)))

    p = types.SimpleNamespace(name='deblurring', u_true=u, u0=np.zeros_like(u), forward=G,
                              jac_adj=lambda v, r: G(r), denoiser=denoiser, BG=1.0)
    p.v_exact = G(u)
    return p


def phase_retrieval(n=64, n_angles=60, tv_weight=0.002):
    theta = np.linspace(1.0, 180.0, n_angles)
    yy, xx = np.mgrid[:n, :n]
    disc = ((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2) <= ((n - 1) / 2) ** 2
    idx = np.flatnonzero(disc.ravel())
    import os
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'radon_{n}_{n_angles}.npy')
    if os.path.exists(cache):
        A = np.load(cache)
    else:
        cols = []
        for i in idx:
            e = np.zeros(n * n)
            e[i] = 1.0
            cols.append(radon(e.reshape(n, n), theta=theta, circle=True).ravel())
        A = np.stack(cols, axis=1)
        keep = np.linalg.norm(A, axis=1) > 0
        A = A[keep]
        A /= np.linalg.norm(A, 2)
        np.save(cache, A)
    img = resize(skdata.shepp_logan_phantom(), (n, n), anti_aliasing=True)
    u_true = img.ravel()[idx]

    def embed(u):
        z = np.zeros(n * n)
        z[idx] = u
        return z.reshape(n, n)

    Dimg = tv_prox(tv_weight)
    cache = {'u': None, 'Au': None}

    def Au(u):
        if cache['u'] is not u:
            cache['u'], cache['Au'] = u, A @ u
        return cache['Au']

    p = types.SimpleNamespace(name='phase', u_true=u_true, A=A, embed=embed, idx=idx, n=n,
                              forward=lambda u: Au(u) ** 2,
                              jac_adj=lambda u, r: 2.0 * (A.T @ (Au(u) * r)),
                              denoiser=lambda u: Dimg(embed(u)).ravel()[idx])
    p.v_exact = p.forward(u_true)
    a1 = (A @ np.ones_like(u_true)) ** 2
    p.c_init = lambda vb: np.sqrt(max(np.dot(a1, vb) / np.dot(a1, a1), 1e-12))
    p.u0 = p.c_init(p.v_exact) * np.ones_like(u_true)
    return p


def eikonal(tv_weight=0.01, fine_data=False):
    E = Eikonal()
    p = types.SimpleNamespace(name='eikonal', u_true=E.u_true, u0=np.zeros_like(E.u_true),
                              forward=E.forward, jac_adj=E.jac_adj, denoiser=tv_prox(tv_weight), E=E)
    p.v_exact = E.v_exact_fine.copy() if fine_data else E.forward(E.u_true)
    return p
