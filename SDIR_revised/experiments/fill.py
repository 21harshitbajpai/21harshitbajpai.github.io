"""Fill the LaTeX templates with numbers computed from the result files."""
import pickle
import re
import sys

import numpy as np

import config
import problems
from sentences import S


def L(name):
    return pickle.load(open(f'results_{name}.pkl', 'rb'))


def mean(R, meth, m, key='err'):
    return float(np.mean([r[key] for r in R['rows'] if r['method'] == meth and r['m'] == m]))


def vals(R, key, meth=None):
    return [r[key] for r in R['rows'] if (meth is None or r['method'] == meth) and not np.isnan(r[key])]


def f3(x):
    return f'{x:.3f}'


def f4(x):
    return f'{x:.4f}'


N = {}
db, ph, ek, ekf = L('deblurring'), L('phase'), L('eikonal'), L('eikonal_fine')
reps = lambda R: len({r['rep'] for r in R['rows']})
N['reps_db'], N['reps_ph'], N['reps_ek'] = reps(db), reps(ph), reps(ekf)
N['tv_ph'] = config.TUNED['phase']['tv']
dbf = f4  # deblurring errors are small: four digits
for tag, R, f in [('db', db, dbf), ('ph', ph, f3)]:
    for m in [4, 64, 1024]:
        N[f'{tag}_dd_{m}'] = f(mean(R, 'dddir', m))
    N[f'{tag}_lw_64'] = f(mean(R, 'landweber', 64))
    N[f'{tag}_n_1024'] = f"{mean(R, 'dddir', 1024, 'n'):.0f}"
    N[f'{tag}_n_64'] = f"{mean(R, 'dddir', 64, 'n'):.0f}"
    N[f'{tag}_lwn_64'] = f"{mean(R, 'landweber', 64, 'n'):.0f}"
    N[f'{tag}_d1'] = f(mean(R, 'ddir1', 1))
    N[f'fac_{tag}'] = f"{mean(R, 'ddir1', 1) / mean(R, 'dddir', 1024):.1f}"
    N[f'orc_{tag}'] = f"{max(abs(mean(R, 'dddir', m) - mean(R, 'oracle', m)) for m in R['ms']):.4f}"
    N[f'{tag}_redn_1024'] = f"{mean(R, 'red', 1024, 'n'):.0f}"
    N[f'{tag}_red_t'] = f"{mean(R, 'red', 1024, 'time'):.0f}"
    N[f'{tag}_dd_t'] = f"{mean(R, 'dddir', 1024, 'time'):.1f}"
    ms = np.array(R['ms'])
    mse = np.array([np.mean([r['err'] ** 2 for r in R['rows'] if r['method'] == 'dddir' and r['m'] == m]) for m in ms])
    b = -np.polyfit(np.log(ms), np.log(mse), 1)[0]
    N[f'slope_{tag}'] = f'{b:.2f}'
    N[f'theta_{tag}'] = f'{b / (2 * (1 - b)):.2f}'
sd = []
for R in (db, ph):
    for m in [64, 256, 1024]:
        v = [r['err'] for r in R['rows'] if r['method'] == 'dddir' and r['m'] == m]
        sd.append(np.std(v) / np.mean(v) * 100)
N['sd_max'] = f'{np.ceil(max(sd)):.0f}'
zs = vals(db, 'z_sigma') + vals(ph, 'z_sigma')
N['zs_min'], N['zs_max'] = f'{min(zs):.2f}', f'{max(zs):.2f}'
nr = vals(db, 'ratio_noise') + vals(ph, 'ratio_noise')
N['nr_min'], N['nr_max'] = f'{min(nr):.2f}', f'{max(nr):.2f}'
N['tau1024'] = f"{config.P(tau0=1.02).tau(1024):.2f}"
# verification
V = pickle.load(open('verify_deblur.pkl', 'rb'))
N['vt_c'] = f"{V['compliant']['vartheta']:.2f}"
N['reps_v'] = len({r['rep'] for r in V['compliant']['rows']})
Ps = pickle.load(open('verify_phase_s.pkl', 'rb'))
N['reps_s'] = len({r['rep'] for r in Ps})
# u-hat
for tag, R in [('ph', ph), ('ek', ekf)]:
    N[f'{tag}_eh_4'] = f3(mean(R, 'dddir', 4, 'err_uhat'))
    N[f'{tag}_eh_1024'] = f3(mean(R, 'dddir', 1024, 'err_uhat'))
    N[f'{tag}_uhat'] = f3(np.linalg.norm(R['uhat'] - R['u_true']) / np.linalg.norm(R['u_true']))
# eikonal appendix
for k in ['dddir', 'landweber', 'red', 'pnp']:
    pass
N['ek_dd_4'], N['ek_dd_1024'] = f3(mean(ekf, 'dddir', 4)), f3(mean(ekf, 'dddir', 1024))
N['ek_d1'] = f3(mean(ekf, 'ddir1', 1))
N['ek_lw_1024'] = f3(mean(ekf, 'landweber', 1024))
N['ek_red_1024'], N['ek_pnp_1024'] = f3(mean(ekf, 'red', 1024)), f3(mean(ekf, 'pnp', 1024))
N['ek_sg_1024'] = f3(mean(ek, 'dddir', 1024))
E = problems.eikonal(fine_data=True)
N['model_err'] = f"{np.linalg.norm(E.forward(E.u_true) - E.v_exact) / np.linalg.norm(E.v_exact):.3f}"
N.update(S)

out = []
for tpl, dst in [('sec4_template.tex', 'paper_out/sec4.tex'), ('appD_template.tex', 'paper_out/appD.tex')]:
    t = open(tpl).read()
    missing = set(re.findall(r'\{\{(\w+)\}\}', t)) - set(N)
    if missing:
        print('MISSING', tpl, missing)
    t = re.sub(r'\{\{(\w+)\}\}', lambda mm: str(N.get(mm.group(1), '??' + mm.group(1))), t)
    open(dst, 'w').write(t)
for k in sorted(N):
    if k not in S:
        print(k, '=', N[k])
