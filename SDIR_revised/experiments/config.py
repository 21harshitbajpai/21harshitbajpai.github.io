"""Problem setups and tuned parameters (tuned on the held-out noise realisation, seed 999)."""
import copy
import os

import numpy as np

import problems
from core import P

EXTRA = 1500        # iterations beyond n_dp for the semi-convergence study (first realisation, m = 64)
HERE = os.path.dirname(os.path.abspath(__file__))


def cnn_denoiser(path=os.path.join(HERE, 'cnn.pt')):
    import torch
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 2)))
    model = torch.load(path, map_location='cpu', weights_only=False).eval()

    def D(u):
        with torch.no_grad():
            x = torch.as_tensor(u, dtype=torch.float32)[None, None]
            return (x - model(x))[0, 0].double().numpy()
    return D


# ---- tuned parameter sets (filled in after tuning) ----
TUNED = {
    'deblurring': dict(base=dict(gamma0=0.5, gamma1=1.5, nu0=0.2, nu1=1.0, nu2=2.0, alpha0=0.9, s=0.1,
                                 tau0=1.02, N0=500),
                       red=dict(omega=0.2), pnp=dict(beta=0.3)),
    'phase': dict(base=dict(gamma0=1.4, gamma1=12.0, nu0=8.0, nu1=0.5, nu2=1.0, alpha0=0.9, s=0.5,
                            tau0=1.02, N0=2000),
                  red=dict(omega=0.002), pnp=dict(beta=0.02), tv=0.005),
    'eikonal': dict(base=dict(gamma0=0.5, gamma1=10.0, nu0=10.0, nu1=0.9, nu2=10.0, alpha0=0.9, s=1.1,
                              tau0=1.02, N0=3000),
                    red=dict(omega=0.003), pnp=dict(beta=0.01), tv=0.1),
}


def setup(name, tuned=None):
    t = copy.deepcopy(tuned or TUNED[name.replace('_fine', '')])
    if name == 'deblurring':
        prob = problems.deblurring(cnn_denoiser())
    elif name == 'phase':
        prob = problems.phase_retrieval(tv_weight=t.get('tv', 0.002))
    elif name == 'eikonal':
        prob = problems.eikonal(tv_weight=t.get('tv', 0.1))
    elif name == 'eikonal_fine':
        prob = problems.eikonal(tv_weight=t.get('tv', 0.1), fine_data=True)
    base = P(**t['base'])
    pars = {k: copy.deepcopy(base) for k in ['dddir', 'oracle', 'landweber']}
    pars['red'] = P(**{**t['base'], **t['red']})
    pars['pnp'] = P(**{**t['base'], **t['pnp']})
    return prob, pars
