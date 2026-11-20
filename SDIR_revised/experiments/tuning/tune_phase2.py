import os; os.environ['OMP_NUM_THREADS']='1'
import itertools, numpy as np
from core import P
import problems
from quick import trial
prob=problems.phase_retrieval()
def setD(tw):
    D=problems.tv_prox(tw); prob.denoiser=lambda u: D(prob.embed(u)).ravel()[prob.idx]
for tw,nu0,nu1 in itertools.product([0.002,0.005],[8,30],[0.1,0.5]):
    setD(tw); par=P(gamma0=1.4,gamma1=12,nu0=nu0,nu1=nu1,nu2=10,s=1.1,tau0=1.02,N0=3000)
    r=[trial(prob,par,'dddir',m)[:3] for m in (16,256)]
    print('dddir',tw,nu0,nu1,[(round(float(a),4),b,c) for a,b,c in r],flush=True)
setD(0.002)
par=P(gamma0=1.4,gamma1=12,tau0=1.02,N0=3000)
print('LW',[(round(float(a),4),b,c) for a,b,c in [trial(prob,par,'landweber',m)[:3] for m in (16,256)]],flush=True)
for om in [0.002,0.01,0.05]:
    par=P(gamma0=1.4,gamma1=12,tau0=1.02,N0=3000,omega=om)
    print('RED',om,[(round(float(a),4),b,c) for a,b,c in [trial(prob,par,'red',m)[:3] for m in (16,256)]],flush=True)
for be in [0.02,0.1,0.3]:
    par=P(gamma0=1.4,gamma1=12,tau0=1.02,N0=3000,beta=be)
    print('PnP',be,[(round(float(a),4),b,c) for a,b,c in [trial(prob,par,'pnp',m)[:3] for m in (16,256)]],flush=True)
