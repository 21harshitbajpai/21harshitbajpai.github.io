"""Run all methods for one problem over replications x m (nested streams), save results.

usage: python run_all.py <problem> <reps> [ms...]
"""
import os
import pickle
import sys
import time

import numpy as np
from skimage.metrics import structural_similarity

from core import Stream, run, sig_entry, psnr
import config

prob_name = sys.argv[1]
reps = int(sys.argv[2])
ms = [int(x) for x in sys.argv[3:]] or [4, 16, 64, 256, 1024]
prob, pars = config.setup(prob_name)
methods = ['dddir', 'oracle', 'landweber', 'red', 'pnp']
img = getattr(prob, 'embed', None)


def quality(u):
    a = img(u) if img else u
    b = img(prob.u_true) if img else prob.u_true
    if a.ndim != 2:
        return np.nan, np.nan
    dr = b.max() - b.min()
    return psnr(a, b), structural_similarity(a, b, data_range=dr)


rows, recon, hist = [], {}, {}
uhat = np.load(f'uhat_{prob_name}.npy') if os.path.exists(f'uhat_{prob_name}.npy') else None
T0 = time.time()
for rep in range(reps):
    st = Stream(prob.v_exact, sig_entry(prob), 1000 + rep)
    for m in sorted(ms):
        vb, z = st.advance_to(m)
        if hasattr(prob, 'c_init'):
            prob.u0 = prob.c_init(vb) * np.ones_like(prob.u_true)
        eta = np.linalg.norm(vb - prob.v_exact)
        for meth in methods:
            par = pars[meth]
            thr = par.tau(m) * (eta if meth == 'oracle' else z / np.sqrt(m))
            core_m = 'dddir' if meth == 'oracle' else meth
            extra = config.EXTRA if (rep == 0 and m == 64) else 0
            o = run(prob, vb, par, core_m, thr, par.nmax(m), refs={'t': prob.u_true}, n_extra=extra)
            e = o['err']['t'][o['n_stop']]
            ps, ss = quality(o['u_stop'])
            eh = np.linalg.norm(o['u_stop'] - uhat) / np.linalg.norm(uhat) if uhat is not None else np.nan
            rows.append(dict(rep=rep, m=m, method=meth, err=e, err_uhat=eh, n=o['n_stop'], reason=o['reason'],
                             time=o['time'], psnr=ps, ssim=ss, z_sigma=z / st.sigma,
                             ratio_noise=eta * np.sqrt(m) / z))
            if rep == 0:
                recon[(meth, m)] = o['u_stop']
                if m == 64:
                    hist[meth] = dict(err=o['err']['t'], res=o['res'], ratio=o['ratio'], n_stop=o['n_stop'],
                                      thr=thr)
            print(f'rep {rep} m {m} {meth:9s} err {e:.4f} n {o["n_stop"]} {o["reason"]} '
                  f'{o["time"]:.1f}s  [{time.time()-T0:.0f}s]', flush=True)
    # DDIR on a single measurement with exact noise level
    par = pars['oracle']
    v1 = st.first
    if hasattr(prob, 'c_init'):
        prob.u0 = prob.c_init(v1) * np.ones_like(prob.u_true)
    o = run(prob, v1, par, 'dddir', 1.1 * np.linalg.norm(v1 - prob.v_exact), par.nmax(1), refs={'t': prob.u_true})
    ps, ss = quality(o['u_stop'])
    rows.append(dict(rep=rep, m=1, method='ddir1', err=o['err']['t'][o['n_stop']], err_uhat=np.nan, n=o['n_stop'],
                     reason=o['reason'], time=o['time'], psnr=ps, ssim=ss, z_sigma=np.nan, ratio_noise=np.nan))
    if rep == 0:
        recon[('ddir1', 1)] = o['u_stop']
    print(f'rep {rep} DDIR(m=1) err {rows[-1]["err"]:.4f} n {o["n_stop"]}', flush=True)
    pickle.dump(dict(rows=rows, recon=recon, hist=hist, ms=ms, u_true=prob.u_true, uhat=uhat,
                     pars={k: vars(v) for k, v in pars.items()}), open(f'results_{prob_name}.pkl', 'wb'))
print('done', time.time() - T0)
