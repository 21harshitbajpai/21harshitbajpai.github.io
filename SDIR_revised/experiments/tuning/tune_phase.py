import os; os.environ['OMP_NUM_THREADS']='1'
import itertools, numpy as np, json
from core import P
import problems
from quick import trial
prob=problems.phase_retrieval()
res=[]
for tw,nu0,nu1,s,tau0 in itertools.product([0.002,0.005],[0.8,8],[0.1,0.5],[0.1,1.1],[1.05]):
    par=P(gamma0=1.4,gamma1=12,nu0=nu0,nu1=nu1,nu2=10,s=s,tau0=tau0,N0=3000)
    prob.denoiser=(lambda D: (lambda u: D(prob.embed(u)).ravel()[prob.idx]))(problems.tv_prox(tw))
    r=[trial(prob,par,'dddir',m)[:3] for m in (16,256)]
    row=dict(tw=tw,nu0=nu0,nu1=nu1,s=s,tau0=tau0,r=[(float(a),int(b),c) for a,b,c in r]); res.append(row); print(row, flush=True)
json.dump(res,open('tune_phase.json','w'))
for tw in [0.002]:
    par=P(gamma0=1.4,gamma1=12,N0=3000,tau0=1.05)
    r=[trial(prob,par,'landweber',m)[:3] for m in (16,256)]; print('LW',r,flush=True)
