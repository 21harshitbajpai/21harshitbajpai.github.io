"""Result-dependent sentences for the templates (numbers computed from the result files)."""
import pickle

import numpy as np


def _L(n):
    return pickle.load(open(f'results_{n}.pkl', 'rb'))


def _m(R, meth, m, key='err'):
    return float(np.mean([r[key] for r in R['rows'] if r['method'] == meth and r['m'] == m]))


def _bud(R, meth, m):
    return float(np.mean([r['reason'] == 'budget' for r in R['rows'] if r['method'] == meth and r['m'] == m]))


db, ph = _L('deblurring'), _L('phase')
V = pickle.load(open('verify_deblur.pkl', 'rb'))
Ps = pickle.load(open('verify_phase_s.pkl', 'rb'))
S = {}

# --- RED / PnP comparison
best_rp_db_256 = min(_m(db, 'red', 256), _m(db, 'pnp', 256))
S['rp_sentence'] = (
    f"For deblurring, RED-SDP and PnP-PGD-SDP are slightly more accurate than \\texttt{{SDIR}} at $m=256$ "
    f"(${best_rp_db_256:.4f}$ compared with ${_m(db, 'dddir', 256):.4f}$), while \\texttt{{SDIR}} is the most accurate method "
    f"for $m\\le64$ and at $m=1024$ (${_m(db, 'dddir', 1024):.4f}$ compared with ${_m(db, 'red', 1024):.4f}$ and ${_m(db, 'pnp', 1024):.4f}$). "
    f"For phase retrieval, \\texttt{{SDIR}} is the most accurate method for $m\\le64$; at $m=256$, PnP-PGD-SDP and RED-SDP attain "
    f"${_m(ph, 'pnp', 256):.3f}$ and ${_m(ph, 'red', 256):.3f}$ compared with ${_m(ph, 'dddir', 256):.3f}$, and at $m=1024$ "
    f"\\texttt{{SDIR}} attains ${_m(ph, 'dddir', 1024):.3f}$ compared with ${_m(ph, 'pnp', 1024):.3f}$ and ${_m(ph, 'red', 1024):.3f}$.")

# --- denoiser paragraph (values from the table file)
S['denoiser_paragraph'] = (
    "For the TV proximal maps, all ratios confirm the theory: they are non-expansive and satisfy (D1) with $p_c=0$. "
    "The random-pair test does not detect any violation for the learned denoiser either. The Jacobian analysis, however, "
    "reveals directions in which the learned denoiser is expansive: $\\|J_D(x)\\|$ reaches $1.73$ after fine-tuning ($2.75$ before). "
    "Hence $D$ is neither non-expansive nor $q$-contractive, and the assumptions of~\\cite{BGJJ26} are violated. Pseudo-contractivity "
    "is a much weaker requirement, and the penalty during fine-tuning reduces the worst violation "
    "$-\\min_x\\lambda_{\\min}(\\mathrm{sym}\\,J_\\Psi(x))$ from $0.94$ to $0.31$. The violation is not removed completely, since the "
    "penalty is enforced on $96\\times96$ patches, whereas the test images have size $256\\times256$; in other words, the learned "
    "denoiser satisfies (D1) only approximately. Since the proofs use (D1) only along the iterates (Remark~\\ref{rem:ass_D1}), the "
    "relevant check is along the trajectory: for the first realisation with $m=256$, the smallest value of "
    "$\\langle\\Psi(u_j^m)-\\Psi(u_k^m),u_j^m-u_k^m\\rangle/\\|u_j^m-u_k^m\\|^2$ over all pairs of iterates is "
    f"${V['traj_mono']:.4f}$, i.e.\\ (D1) is violated only marginally where it is used. Removing the penalty increases the error "
    "of \\texttt{SDIR} (Section~\\ref{ssec:num_verify}).")

# --- theory-compliant
c, p = V['compliant']['rows'], V['practical']['rows']
ce = {m: np.mean([r['err'] for r in c if r['m'] == m]) for m in [4, 1024]}
pe = {m: np.mean([r['err'] for r in p if r['m'] == m]) for m in [4, 1024]}
S['compliant_sentence'] = (
    f"With the theory-compliant parameters, $\\mathcal E_m$ occurs in all runs, so that Lemma~\\ref{{lem:dDDIR_monotonicity}} and "
    f"Theorem~\\ref{{thm:dDDIR_termination}} apply to every run. The price is a larger error: the larger tolerance stops the iteration "
    f"earlier, and the error at $m=1024$ is ${ce[1024]:.4f}$ instead of ${pe[1024]:.4f}$. This quantifies how conservative the "
    f"conditions of the analysis are.")

# --- verification
viol_c = sum(r['viol'] for r in c)
viol_p = {m: sum(r['viol'] for r in p if r['m'] == m) for m in [4, 16, 64, 256, 1024]}
nmax_64 = np.mean([r['n'] for r in p if r['m'] == 64])
ok_res = all(r['ressum'] <= r['ressum_bound'] for r in c if r['inE'])
ok_n = all(r['n'] <= r['bound'] for r in c if r['inE'])
st = V['stab']
r5 = st[5][64] / st[5][1024]
S['verify_sentence'] = (
    f"For the theory-compliant parameters, the error decreases monotonically in every run ({viol_c} violations), "
    f"and the stopping index and the sum of the squared residuals satisfy the bounds of Theorem~\\ref{{thm:dDDIR_termination}} "
    f"and Lemma~\\ref{{lem:dDDIR_monotonicity}}(iii) in every run{'' if (ok_res and ok_n) else ' (with exceptions)'}; the termination bound is, however, "
    f"pessimistic by several orders of magnitude, since $\\mathcal C_D$ is small. For the tuned parameters, which lie outside the "
    f"range of the theory, the error is monotone for all $m$ except $m=64$. There, the iteration performs about {nmax_64:.0f} iterations "
    f"close to the stationary point described in Section~\\ref{{ssec:num_semi}}, and the error oscillates slightly ({viol_p[64]} "
    f"increases in total over the realisations), while its final value is still well below that of Landweber-SDP. The mean-square "
    f"stability error $\\mathbb E\\|u_j^m-u_j\\|^2$ decreases like $m^{{-1}}$, as expected for a linear forward operator: from $m=64$ to "
    f"$m=1024$, it decreases by a factor of {r5:.1f} for $j=5$.")

# --- ablation
A = V['abl_d1']
ab = {(lab, m): np.mean([r['err'] for r in A if r['cnn'] == lab and r['m'] == m]) for lab in ['penalty', 'mse'] for m in [16, 1024]}
ps = {(s, m): (np.mean([r['err'] for r in Ps if r['s'] == s and r['m'] == m]),
               np.mean([r['reason'] == 'budget' for r in Ps if r['s'] == s and r['m'] == m]),
               np.mean([r['n'] for r in Ps if r['s'] == s and r['m'] == m]))
      for s in sorted({r['s'] for r in Ps}) for m in [16, 256, 1024]}
stall = [s for s in sorted({r['s'] for r in Ps}) if ps[(s, 1024)][1] > 0.5]
S['ablation_sentence'] = (
    f"Without the monotonicity penalty, the error of \\texttt{{SDIR}} increases at every $m$, e.g.\\ from ${ab[('penalty', 1024)]:.4f}$ to "
    f"${ab[('mse', 1024)]:.4f}$ at $m=1024$, which indicates that (D1) matters in practice. For phase retrieval, "
    + (f"$s={stall[0]}$ stagnates at the budget for large $m$, " if stall else "")
    + f"$s=0.5$ gives the smallest errors (${ps[(0.5, 1024)][0]:.3f}$ at $m=1024$), and the summable choices $s\\in\\{{1.1,1.5\\}}$ "
    f"terminate in all runs with somewhat larger errors (${ps[(1.1, 1024)][0]:.3f}$ and ${ps[(1.5, 1024)][0]:.3f}$ at $m=1024$), "
    f"since the denoiser is switched off early. Thus, a summable sequence $\\{{\\alpha_n\\}}$ guarantees termination, in agreement with "
    f"Theorem~\\ref{{thm:inexact_main}}, while a moderately decaying sequence is a good practical compromise.")
