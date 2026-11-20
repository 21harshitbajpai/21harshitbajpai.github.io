import itertools
from core import P
import problems, config
from quick import trial
prob = problems.deblurring(config.cnn_denoiser())
for g0, (nu0, nu1), nu2 in itertools.product([0.5, 0.3], [(0.2, 1.0), (0.24, 1.0), (0.48, 0.5), (1.0, 0.24)], [1.0, 2.0]):
    if nu0 * nu1 >= g0 - g0 ** 2: continue
    par = P(gamma0=g0, gamma1=1.5, nu0=nu0, nu1=nu1, nu2=nu2, s=0.1, tau0=1.02, N0=500)
    r = [trial(prob, par, 'dddir', m)[:3] for m in (16, 256)]
    print('dddir', g0, nu0, nu1, nu2, [(round(float(a), 4), b, c) for a, b, c in r], flush=True)
for om in [0.5, 1.0]:
    par = P(gamma0=0.5, gamma1=1.5, tau0=1.02, N0=500, omega=om)
    print('RED', om, [(round(float(a), 4), b, c) for a, b, c in [trial(prob, par, 'red', m)[:3] for m in (16, 256)]], flush=True)
for be in [0.5]:
    par = P(gamma0=0.5, gamma1=1.5, tau0=1.02, N0=500, beta=be)
    print('PnP', be, [(round(float(a), 4), b, c) for a, b, c in [trial(prob, par, 'pnp', m)[:3] for m in (16, 256)]], flush=True)
