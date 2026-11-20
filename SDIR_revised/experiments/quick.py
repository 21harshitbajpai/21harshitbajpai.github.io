import sys, numpy as np, time
from core import P, Stream, run, sig_entry
import problems
def trial(prob, par, method, m, seed=999, extra=0):
    st=Stream(prob.v_exact, sig_entry(prob), seed); vb,z=st.advance_to(m)
    if hasattr(prob,'c_init'): prob.u0=prob.c_init(vb)*np.ones_like(prob.u_true)
    thr=par.tau(m)*z/np.sqrt(m)
    t=time.time(); o=run(prob, vb, par, method, thr, par.nmax(m), refs={'t':prob.u_true}, n_extra=extra)
    return o['err']['t'][o['n_stop']], o['n_stop'], o['reason'], time.time()-t, o
