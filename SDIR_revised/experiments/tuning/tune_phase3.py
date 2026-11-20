import os; os.environ['OMP_NUM_THREADS']='1'
import itertools
from core import P
import problems
from quick import trial
prob=problems.phase_retrieval()
def setD(tw):
    D=problems.tv_prox(tw); prob.denoiser=lambda u: D(prob.embed(u)).ravel()[prob.idx]
for tw,nu2,s in itertools.product([0.005,0.01],[0.5,1.0],[0.1,0.5,1.1]):
    setD(tw); par=P(gamma0=1.4,gamma1=12,nu0=8,nu1=0.5,nu2=nu2,s=s,tau0=1.02,N0=3000)
    r=[trial(prob,par,'dddir',m)[:3] for m in (16,256)]
    print('dddir',tw,nu2,s,[(round(float(a),4),b,c) for a,b,c in r],flush=True)
for tw in [0.005,0.01]:
    setD(tw)
    for om in [0.002,0.005]:
        par=P(gamma0=1.4,gamma1=12,tau0=1.02,N0=3000,omega=om)
        print('RED',tw,om,[(round(float(a),4),b,c) for a,b,c in [trial(prob,par,'red',m)[:3] for m in (16,256)]],flush=True)
    for be in [0.02,0.05]:
        par=P(gamma0=1.4,gamma1=12,tau0=1.02,N0=3000,beta=be)
        print('PnP',tw,be,[(round(float(a),4),b,c) for a,b,c in [trial(prob,par,'pnp',m)[:3] for m in (16,256)]],flush=True)
