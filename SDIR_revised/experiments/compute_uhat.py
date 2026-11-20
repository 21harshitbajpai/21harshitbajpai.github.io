"""Exact-data limit u_hat (Algorithm 2) for a problem: python compute_uhat.py <problem> [n_iter]"""
import sys, numpy as np, time
from core import run
import config
name = sys.argv[1]; n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
prob, pars = config.setup(name)
par = pars['dddir']
t = time.time()
o = run(prob, prob.v_exact, par, 'dddir', -1.0, n_iter, refs={'t': prob.u_true})
u = o['u_stop']
print(name, 'iters', n_iter, 'rel err to u_dag', np.linalg.norm(u - prob.u_true) / np.linalg.norm(prob.u_true),
      'final residual', o['res'][-1] / np.linalg.norm(prob.v_exact), f'{time.time()-t:.0f}s')
print('err history (every n/10):', [round(o['err']['t'][i], 4) for i in range(0, n_iter + 1, max(1, n_iter // 10))])
np.save(f'uhat_{name}.npy', u)
