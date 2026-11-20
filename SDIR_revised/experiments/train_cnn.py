"""Train a DnCNN-type residual denoiser D = I - N_theta with a monotonicity penalty on sym J_{N_theta}.

Stage 1: MSE pre-training (predict the noise).  Stage 2: fine-tuning with
    loss = MSE + w * mean( max(0, eps - lambda_min(sym J_N(x)))^2 ),
lambda_min estimated by power iteration on c*I - sym J (Rayleigh quotient differentiated).
Test image (cameraman) is excluded from training.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from skimage import data as skd
from skimage.color import rgb2gray
from skimage.transform import rescale

torch.set_num_threads(2)
torch.manual_seed(0)
rng = np.random.default_rng(0)


def net(L=8, C=32):
    layers = [nn.Conv2d(1, C, 3, padding=1), nn.ReLU()]
    for _ in range(L - 2):
        layers += [nn.Conv2d(C, C, 3, padding=1), nn.ReLU()]
    layers += [nn.Conv2d(C, 1, 3, padding=1)]
    return nn.Sequential(*layers)


def load_images():
    names = ['astronaut', 'coffee', 'chelsea', 'rocket', 'moon', 'page', 'text', 'coins', 'brick',
             'grass', 'gravel', 'clock', 'horse', 'cell', 'immunohistochemistry', 'hubble_deep_field']
    ims = []
    for n in names:
        x = getattr(skd, n)()
        if x.ndim == 3:
            x = rgb2gray(x[..., :3])
        x = x.astype(float)
        x = x / x.max() if x.max() > 1.5 else x
        ims.append(x)
        ims.append(rescale(x, 0.5, anti_aliasing=True))
    return ims


IMS = load_images()


def batch(B, P):
    out = np.empty((B, 1, P, P))
    for b in range(B):
        im = IMS[rng.integers(len(IMS))]
        while min(im.shape) <= P + 1:
            im = IMS[rng.integers(len(IMS))]
        i, j = rng.integers(im.shape[0] - P), rng.integers(im.shape[1] - P)
        p = im[i:i + P, j:j + P]
        if rng.random() < 0.5:
            p = p[::-1]
        p = np.rot90(p, rng.integers(4))
        out[b, 0] = p
    x = torch.tensor(out, dtype=torch.float32)
    s = torch.rand(B, 1, 1, 1) * 0.10
    noise = s * torch.randn_like(x)
    return x + noise, noise


def lam_min_sym(model, x, iters=6, c=2.0):
    """Rayleigh-quotient estimate of lambda_min(sym J_N(x)) per sample (differentiable in theta)."""
    v = torch.randn_like(x)
    v = v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    for _ in range(iters):
        with torch.no_grad():
            _, Jv = torch.func.jvp(model, (x,), (v,))
            _, vjp = torch.func.vjp(model, x)
            JTv = vjp(v)[0]
            w = c * v - 0.5 * (Jv + JTv)
            v = w / w.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    _, Jv = torch.func.jvp(model, (x,), (v,))
    return (v * Jv).flatten(1).sum(1)          # v^T sym(J) v = v^T J v


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'cnn.pt'
    n1, n2 = int(sys.argv[2]) if len(sys.argv) > 2 else 3000, int(sys.argv[3]) if len(sys.argv) > 3 else 800
    model = net()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n1)
    t0 = time.time()
    for it in range(n1):
        y, noise = batch(32, 48)
        loss = ((model(y) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 250 == 0:
            print(f'pre {it} mse {loss.item():.2e} {time.time()-t0:.0f}s', flush=True)
    torch.save(model, out.replace('.pt', '_mse.pt'))
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    for it in range(n2):
        y, noise = batch(16, 48)
        mse = ((model(y) - noise) ** 2).mean()
        yp, _ = batch(4, 96)
        lm = lam_min_sym(model, yp)
        pen = (torch.relu(0.02 - lm) ** 2).mean()
        loss = mse + 10.0 * pen
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            print(f'pen {it} mse {mse.item():.2e} lam_min {lm.min().item():+.3f} {time.time()-t0:.0f}s', flush=True)
    torch.save(model, out)
    print('done', time.time() - t0)
