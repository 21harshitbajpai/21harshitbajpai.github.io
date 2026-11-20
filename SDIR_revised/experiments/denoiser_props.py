"""Empirical properties of the denoisers (Table tab:denoiser)."""
import numpy as np, torch, sys
from skimage import data as skd
from skimage.transform import resize
import problems, config
torch.set_num_threads(2)
rng = np.random.default_rng(5)
cam = resize(skd.camera() / 255.0, (256, 256), anti_aliasing=True)
blur = problems.deblurring(lambda u: u).forward
phase = problems.phase_retrieval()
ph_img = phase.embed(phase.u_true)
eik = problems.eikonal().u_true


def pair_stats(D, base_imgs, n=150):
    lip, mono = [], []
    for _ in range(n):
        x0 = base_imgs[rng.integers(len(base_imgs))]
        kind = rng.integers(3)
        x = x0 + 0.05 * rng.standard_normal(x0.shape) if kind == 0 else (blur(x0) if (kind == 1 and x0.shape == (256, 256)) else x0 + 0.2 * rng.random(x0.shape))
        d = 10.0 ** rng.choice([-3, -2, -1])
        y = x + d * rng.standard_normal(x.shape)
        Dx, Dy = D(x), D(y)
        Px, Py = x - Dx, y - Dy
        lip.append(np.linalg.norm(Dx - Dy) / np.linalg.norm(x - y))
        q = np.linalg.norm(Px - Py) ** 2
        if q > 1e-14:
            mono.append(np.sum((Px - Py) * (x - y)) / q)
    rho = min(mono)
    return max(lip), rho, max(0.0, 1 - 2 * rho)


def jac_stats(model, imgs, iters=40):
    with torch.no_grad():
        return _jac_stats(model, imgs, iters)


def _jac_stats(model, imgs, iters):
    nJ, lmin = [], []
    for x0 in imgs:
        x = torch.tensor(x0, dtype=torch.float32)[None, None]
        D = lambda z: z - model(z)
        v = torch.randn_like(x); v /= v.norm()
        for _ in range(iters):           # ||J_D|| by power iteration on J^T J
            _, Jv = torch.func.jvp(D, (x,), (v,)); _, f = torch.func.vjp(D, x); w = f(Jv)[0]
            s = w.norm().sqrt(); v = w / w.norm()
        nJ.append(float(s))
        v = torch.randn_like(x); v /= v.norm(); c = 3.0
        for _ in range(iters):           # lambda_min(sym J_Psi), Psi = N_theta, shifted power iteration
            _, Jv = torch.func.jvp(model, (x,), (v,)); _, f = torch.func.vjp(model, x); JTv = f(v)[0]
            w = c * v - 0.5 * (Jv + JTv); v = w / w.norm()
        _, Jv = torch.func.jvp(model, (x,), (v,))
        lmin.append(float((v * Jv).sum()))
    return max(nJ), min(lmin)


rows = []
for w, img, lab in [(0.005, ph_img, 'TV prox, $w=0.005$ (phase retrieval)'), (0.1, eik, 'TV prox, $w=0.1$ (eikonal)')]:
    rows.append((lab,) + pair_stats(problems.tv_prox(w), [img, img + 0.3]))
jr = []
for path, lab in [('cnn_mse.pt', 'CNN, MSE pre-training'), ('cnn.pt', r'CNN, with (D1) penalty')]:
    D = config.cnn_denoiser(path)
    rows.append((lab,) + pair_stats(D, [cam, cam[::-1].copy(), resize(skd.moon() / 255., (256, 256))]))
    model = torch.load(path, weights_only=False).eval()
    tests = [cam, cam + 0.05 * rng.standard_normal(cam.shape), blur(cam), resize(skd.moon() / 255., (256, 256))]
    jr.append((lab,) + jac_stats(model, tests))
L = [r'\begin{tabular}{lccc}\toprule',
     r'Denoiser & $\max\frac{\|D x-D y\|}{\|x-y\|}$ & $\min\frac{\langle \Psi x-\Psi y,x-y\rangle}{\|\Psi x-\Psi y\|^2}$ & empirical $p_c$ \\ \midrule']
L += [f'{a} & {b:.3f} & {c:.3f} & {d:.3f} \\\\' for a, b, c, d in rows]
L += [r'\midrule', r' & $\max_x\|J_D(x)\|$ & $\min_x\lambda_{\min}(\mathrm{sym}\,J_\Psi(x))$ & \\ \midrule']
L += [f'{a} & {b:.3f} & ${c:+.3f}$ & -- \\\\' for a, b, c in jr]
L += [r'\bottomrule\end{tabular}']
open('paper_out/tables/denoiser.tex', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
