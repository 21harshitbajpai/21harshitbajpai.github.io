from core import P
import problems, config
from quick import trial
prob = problems.deblurring(config.cnn_denoiser())
for nu0, nu1, nu2, s in [(0.48, 0.5, 2.0, 0.5), (0.48, 0.5, 1.0, 0.5), (0.24, 1.0, 2.0, 0.5), (0.48, 0.5, 2.0, 1.1)]:
    par = P(gamma0=0.5, gamma1=1.5, nu0=nu0, nu1=nu1, nu2=nu2, s=s, tau0=1.02, N0=500)
    r = [trial(prob, par, 'dddir', m)[:3] for m in (16, 256)]
    print('dddir', nu0, nu1, nu2, s, [(round(float(a), 4), b, c) for a, b, c in r], flush=True)
