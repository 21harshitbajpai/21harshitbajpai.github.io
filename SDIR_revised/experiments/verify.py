"""Verification experiments (Section 4.8): deblurring checks + (D1) ablation, phase-retrieval decay ablation."""
import os
import pickle
import sys

import numpy as np

import config
import problems
from core import P, Stream, run, sig_entry

part = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MS = [4, 16, 64, 256, 1024]


def vartheta_min(par, BG=1.0, kappa=0.0):
    gam = min(par.gamma0 / BG, par.gamma1)
    a = (1 - kappa) * gam - par.gamma0 ** 2 - par.nu0 * par.nu1
    return np.inf if a <= 0 else max((1 + kappa) * gam / a, (1 + kappa) / (1 - kappa))


def C_D(par, vt, BG=1.0, kappa=0.0):
    gam = min(par.gamma0 / BG, par.gamma1)
    return (1 - kappa - (1 + kappa) / vt) * gam - par.gamma0 ** 2 - par.nu0 * par.nu1


if part == 'deblur':
    prob, pars = config.setup('deblurring')
    practical = pars['dddir']
    compliant = P(**{**vars(practical), 'gamma0': 0.25, 'nu0': 0.1, 'nu1': 0.2, 'tau0': 1.7})
    out = {}
    e0 = np.linalg.norm(prob.u0 - prob.u_true)
    for name, par in [('compliant', compliant), ('practical', practical)]:
        vt = 1.01 * vartheta_min(par)
        cd = C_D(par, vt) if np.isfinite(vt) else np.nan
        rows = []
        for rep in range(REPS):
            st = Stream(prob.v_exact, sig_entry(prob), 1000 + rep)
            for m in MS:
                vb, z = st.advance_to(m)
                eta = np.linalg.norm(vb - prob.v_exact)
                tau = par.tau(m)
                inE = bool(abs(z - st.sigma) <= st.sigma / 2 and np.isfinite(vt) and eta <= tau * z / (vt * np.sqrt(m)))
                keep = range(0, 400) if (rep == 0 and m == 256 and name == 'practical') else ()
                o = run(prob, vb, par, 'dddir', tau * z / np.sqrt(m), par.nmax(m), refs={'t': prob.u_true}, keep_iter=keep)
                e = np.array(o['err']['t'][:o['n_stop'] + 1]) * np.linalg.norm(prob.u_true)
                viol = int(np.sum(e[1:] > e[:-1] * (1 + 1e-12)))
                bound = 2 * m * e0 ** 2 / (cd * tau ** 2 * st.sigma ** 2) if cd > 0 else np.nan
                rows.append(dict(rep=rep, m=m, inE=inE, err=e[-1] / np.linalg.norm(prob.u_true), n=o['n_stop'],
                                 reason=o['reason'], viol=viol, bound=bound,
                                 ressum=float(np.sum(np.square(o['res'][:o['n_stop']]))), ressum_bound=e0 ** 2 / (2 * cd) if cd > 0 else np.nan))
                if keep:
                    snaps = [o['kept'][k] for k in sorted(o['kept'])]
                    P_ = [x - prob.denoiser(x) for x in snaps]
                    best = np.inf
                    for j in range(len(snaps)):
                        for k in range(j + 1, len(snaps)):
                            d = snaps[j] - snaps[k]
                            nd = np.sum(d * d)
                            if nd > 1e-14:
                                best = min(best, np.sum((P_[j] - P_[k]) * d) / nd)
                    out['traj_mono'] = best
                print(name, rep, m, rows[-1], flush=True)
        out[name] = dict(rows=rows, vartheta=vt, C_D=cd)
    # stability for fixed j
    js = [5, 20]
    exact = run(prob, prob.v_exact, practical, 'dddir', -1.0, max(js), keep_iter=js)['kept']
    stab = {j: {} for j in js}
    for m in MS:
        vals = {j: [] for j in js}
        for rep in range(REPS):
            vb, z = Stream(prob.v_exact, sig_entry(prob), 1000 + rep).advance_to(m)
            o = run(prob, vb, practical, 'dddir', -1.0, max(js), keep_iter=js)
            for j in js:
                vals[j].append(np.sum((o['kept'][j] - exact[j]) ** 2))
        for j in js:
            stab[j][m] = float(np.mean(vals[j]))
        print('stab', m, {j: stab[j][m] for j in js}, flush=True)
    out['stab'] = stab
    # (D1) ablation: CNN without the monotonicity penalty
    prob_mse = problems.deblurring(config.cnn_denoiser(os.path.join(config.HERE, 'cnn_mse.pt')))
    abl = []
    for rep in range(REPS):
        st = Stream(prob.v_exact, sig_entry(prob), 1000 + rep)
        for m in [16, 256, 1024]:
            vb, z = st.advance_to(m)
            for lab, pr in [('penalty', prob), ('mse', prob_mse)]:
                o = run(pr, vb, practical, 'dddir', practical.tau(m) * z / np.sqrt(m), practical.nmax(m),
                        refs={'t': prob.u_true})
                abl.append(dict(rep=rep, m=m, cnn=lab, err=o['err']['t'][o['n_stop']], n=o['n_stop'], reason=o['reason']))
        print('abl', rep, abl[-2:], flush=True)
    out['abl_d1'] = abl
    pickle.dump(out, open('verify_deblur.pkl', 'wb'))

elif part == 'phase_s':
    prob, pars = config.setup('phase')
    base = pars['dddir']
    rows = []
    for s in [0.1, 0.5, 1.1, 1.5]:
        par = P(**{**vars(base), 's': s})
        for rep in range(REPS):
            st = Stream(prob.v_exact, sig_entry(prob), 1000 + rep)
            for m in [16, 256, 1024]:
                vb, z = st.advance_to(m)
                prob.u0 = prob.c_init(vb) * np.ones_like(prob.u_true)
                o = run(prob, vb, par, 'dddir', par.tau(m) * z / np.sqrt(m), par.nmax(m), refs={'t': prob.u_true})
                rows.append(dict(s=s, rep=rep, m=m, err=o['err']['t'][o['n_stop']], n=o['n_stop'], reason=o['reason']))
                print(rows[-1], flush=True)
    pickle.dump(rows, open('verify_phase_s.pkl', 'wb'))
