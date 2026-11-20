import itertools, json, numpy as np, sys
from core import P
import problems, config
from quick import trial
which = sys.argv[1] if len(sys.argv) > 1 else 'dddir'
prob = problems.deblurring(config.cnn_denoiser(sys.argv[2] if len(sys.argv) > 2 else 'cnn.pt'))
res = []
if which == 'dddir':
    grid = itertools.product([1.02, 1.1], [(0.1, 0.9), (0.04, 5.0), (0.01, 20.0), (0.2, 1.0)], [0.1, 1.1], [1.0])
    for tau0, (nu0, nu1), s, nu2 in grid:
        par = P(gamma0=0.5, gamma1=1.5, nu0=nu0, nu1=nu1, nu2=nu2, s=s, tau0=tau0, N0=500)
        r = [trial(prob, par, 'dddir', m)[:3] for m in (16, 256)]
        row = dict(tau0=tau0, nu0=nu0, nu1=nu1, s=s, nu2=nu2, r=[(round(float(a), 4), int(b), c) for a, b, c in r])
        print(row, flush=True); res.append(row)
    for tau0 in [1.02, 1.1]:
        par = P(gamma0=0.5, gamma1=1.5, tau0=tau0, N0=500)
        print('LW', tau0, [trial(prob, par, 'landweber', m)[:3] for m in (16, 256)], flush=True)
else:
    for tau0 in [1.02]:
        for om in [0.01, 0.05, 0.2]:
            par = P(gamma0=0.5, gamma1=1.5, tau0=tau0, N0=500, omega=om)
            r = [trial(prob, par, 'red', m)[:3] for m in (16, 256)]; print('RED', om, r, flush=True)
        for be in [0.1, 0.3, 1.0]:
            par = P(gamma0=0.5, gamma1=1.5, tau0=tau0, N0=500, beta=be)
            r = [trial(prob, par, 'pnp', m)[:3] for m in (16, 256)]; print('PnP', be, r, flush=True)
