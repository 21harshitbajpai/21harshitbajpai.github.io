"""Generate all tables (LaTeX) and figures (PDF) from results_*.pkl."""
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'paper_out'
os.makedirs(f'{OUT}/figures', exist_ok=True)
os.makedirs(f'{OUT}/tables', exist_ok=True)

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8a8a85', 'legend.frameon': False, 'lines.linewidth': 1.6,
                     'pdf.fonttype': 42, 'font.family': 'serif', 'mathtext.fontset': 'cm'})

STYLE = {  # fixed categorical order; marker + dash as secondary encoding
    'dddir': dict(c='#2a78d6', m='o', ls='-', label='SDIR (proposed)'),
    'oracle': dict(c='#1baf7a', m='s', ls='--', label=r'DDIR (oracle $\delta_m$)'),
    'landweber': dict(c='#eda100', m='^', ls='-.', label='Landweber-SDP'),
    'red': dict(c='#eb6834', m='D', ls=':', label='RED-SDP'),
    'pnp': dict(c='#e87ba4', m='v', ls=(0, (5, 1, 1, 1)), label='PnP-PGD-SDP'),
}
NAMES = {'dddir': r'\texttt{SDIR}', 'oracle': r'\texttt{DDIR} (oracle $\delta_m$)',
         'landweber': 'Landweber-SDP', 'red': 'RED-SDP', 'pnp': 'PnP-PGD-SDP', 'ddir1': r'\texttt{DDIR} ($m=1$)'}
METHODS = ['dddir', 'oracle', 'landweber', 'red', 'pnp']
TITLE = {'deblurring': 'Image deblurring', 'phase': 'Phase retrieval', 'eikonal': 'Eikonal equation'}


def load(name):
    p = f'results_{name}.pkl'
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def agg(R, meth, m, key):
    v = [r[key] for r in R['rows'] if r['method'] == meth and r['m'] == m]
    return np.mean(v), np.std(v), len(v)


def budget_frac(R, meth, m):
    v = [r['reason'] == 'budget' for r in R['rows'] if r['method'] == meth and r['m'] == m]
    return np.mean(v)


def nreps(R):
    return len({r['rep'] for r in R['rows']})


# ------------------------------------------------------------------ tables
def block(R, name):
    nd = 4 if name == 'deblurring' else 3
    tol = 0.5 * 10 ** (-nd)
    ms = R['ms']
    lines = [r'\multicolumn{6}{l}{\textbf{' + TITLE[name] + '}' + f' ({nreps(R)} noise realisations)' + r'}\\ \midrule']
    best = {m: min(agg(R, k, m, 'err')[0] for k in METHODS) for m in ms}
    for k in METHODS:
        cells = []
        for m in ms:
            e, _, _ = agg(R, k, m, 'err')
            n, _, _ = agg(R, k, m, 'n')
            star = r'$^{\ast}$' if budget_frac(R, k, m) > 0.5 else ''
            es = f'{e:.{nd}f}'
            if abs(e - best[m]) < tol:
                es = r'\textbf{' + es + '}'
            cells.append(es + r'\,\tiny{(' + f'{n:.0f}' + star + ')}')
        lines.append(NAMES[k] + ' & ' + ' & '.join(cells) + r' \\')
    e, _, _ = agg(R, 'ddir1', 1, 'err')
    n, _, _ = agg(R, 'ddir1', 1, 'n')
    lines.append(NAMES['ddir1'] + r' & \multicolumn{5}{c}{' + f'{e:.{nd}f}' + r'\,\tiny{(' + f'{n:.0f}' + r')} \ (independent of $m$)} \\')
    return lines


def table_main(names, fname):
    L = [r'\begin{tabular}{lccccc}\toprule',
         r'Method & $m=4$ & $m=16$ & $m=64$ & $m=256$ & $m=1024$ \\ \midrule']
    for i, nm in enumerate(names):
        R = load(nm)
        if R is None:
            continue
        if i:
            L.append(r'\midrule')
        L += block(R, nm)
    L.append(r'\bottomrule\end{tabular}')
    open(f'{OUT}/tables/{fname}', 'w').write('\n'.join(L) + '\n')


def table_quality(names):
    L = [r'\begin{tabular}{llcccc}\toprule',
         r' & & \multicolumn{2}{c}{$m=16$} & \multicolumn{2}{c}{$m=256$} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         r'Problem & Method & PSNR & SSIM & PSNR & SSIM \\ \midrule']
    for i, nm in enumerate(names):
        R = load(nm)
        if R is None:
            continue
        if i:
            L.append(r'\midrule')
        for j, k in enumerate(METHODS + ['ddir1']):
            lab = TITLE[nm] if j == 0 else ''
            if k == 'ddir1':
                p, _, _ = agg(R, k, 1, 'psnr'); s, _, _ = agg(R, k, 1, 'ssim')
                L.append(f'{lab} & {NAMES[k]} & {p:.2f} & {s:.3f} & ' + r'\multicolumn{2}{c}{--} \\')
            else:
                c = []
                for m in (16, 256):
                    p, _, _ = agg(R, k, m, 'psnr'); s, _, _ = agg(R, k, m, 'ssim')
                    c += [f'{p:.2f}', f'{s:.3f}']
                L.append(f'{lab} & {NAMES[k]} & ' + ' & '.join(c) + r' \\')
    L.append(r'\bottomrule\end{tabular}')
    open(f'{OUT}/tables/quality.tex', 'w').write('\n'.join(L) + '\n')


# ------------------------------------------------------------------ figures
def ax_err_vs_m(ax, R, name):
    ms = R['ms']
    for k in METHODS:
        st = STYLE[k]
        mu = [agg(R, k, m, 'err')[0] for m in ms]
        sd = [agg(R, k, m, 'err')[1] for m in ms]
        ax.errorbar(ms, mu, sd, color=st['c'], marker=st['m'], ls=st['ls'], ms=5, capsize=2, label=st['label'],
                    mfc='white' if k == 'oracle' else st['c'], lw=1.4)
    e1 = agg(R, 'ddir1', 1, 'err')[0]
    ax.axhline(e1, color='#6b6b66', ls='--', lw=1.0, label='DDIR, single measurement')
    ax.set_xscale('log'); ax.set_xticks(ms, [str(m) for m in ms]); ax.minorticks_off()
    ax.set_xlabel('number of measurements $m$'); ax.set_title(TITLE[name], fontsize=9.5)


def fig_err_vs_m(names, fname):
    Rs = [(n, load(n)) for n in names if load(n) is not None]
    fig, axs = plt.subplots(1, len(Rs), figsize=(3.4 * len(Rs) + 0.4, 2.9))
    axs = np.atleast_1d(axs)
    for ax, (n, R) in zip(axs, Rs):
        ax_err_vs_m(ax, R, n)
    axs[0].set_ylabel(r'relative error $\|u^m_{n_{\rm dp}}-u^\dagger\|/\|u^\dagger\|$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(f'{OUT}/figures/{fname}', bbox_inches='tight'); plt.close(fig)


def fig_mse_stop(names, fname='mse_stop.pdf'):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    slopes = {}
    for i, n in enumerate(names):
        R = load(n)
        if R is None:
            continue
        ms = np.array(R['ms'])
        mse = np.array([np.mean([r['err'] ** 2 for r in R['rows'] if r['method'] == 'dddir' and r['m'] == m]) for m in ms])
        b, a = np.polyfit(np.log(ms), np.log(mse), 1)
        slopes[n] = -b
        c = ['#2a78d6', '#eb6834', '#1baf7a'][i]
        mk = ['o', 's', '^'][i]
        axs[0].loglog(ms, mse, color=c, marker=mk, label=f'{TITLE[n]} (fit $m^{{-{-b:.2f}}}$)')
        axs[0].loglog(ms, np.exp(a) * ms ** b, color=c, ls=':', lw=1)
        nd = [agg(R, 'dddir', m, 'n') for m in ms]
        axs[1].errorbar(ms, [x[0] for x in nd], [x[1] for x in nd], color=c, marker=mk, capsize=2, label=TITLE[n])
    for ax in axs:
        ax.set_xscale('log'); ax.set_xticks(ms, [str(m) for m in ms]); ax.minorticks_off(); ax.set_xlabel('$m$')
    axs[0].set_ylabel('mean squared relative error'); axs[1].set_ylabel(r'stopping index $n_{\rm dp}$')
    axs[1].set_yscale('log')
    axs[0].legend(fontsize=7.5); axs[1].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(f'{OUT}/figures/{fname}', bbox_inches='tight'); plt.close(fig)
    return slopes


def to_img(name, u):
    if name == 'phase':
        import problems
        p = problems.phase_retrieval()
        return p.embed(u)
    return u


def fig_recon(name, fname, vmin=None, vmax=None, cmap='gray', extra=None, crop=None):
    R = load(name)
    if R is None:
        return
    rec = R['recon']
    ut = to_img(name, R['u_true'])
    vmin = ut.min() if vmin is None else vmin
    vmax = ut.max() if vmax is None else vmax
    cols = ['landweber', 'red', 'pnp', 'oracle', 'dddir']
    fig, axs = plt.subplots(2, 6, figsize=(7.2, 2.75))
    for r, m in enumerate((16, 256)):
        first = ('exact', None) if r == 0 else ('ddir1', 1)
        panels = [first] + [(k, m) for k in cols]
        for c, (k, mm) in enumerate(panels):
            ax = axs[r, c]
            u = ut if k == 'exact' else to_img(name, rec[(k, mm)])
            if crop is not None:
                u = u[crop]
            ax.imshow(u, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if k == 'exact':
                ttl = r'exact $u^\dagger$'
                if extra:
                    extra(ax)
            else:
                e = np.linalg.norm(rec[(k, mm)] - R['u_true']) / np.linalg.norm(R['u_true'])
                lab = {'ddir1': 'DDIR, $m=1$', 'landweber': 'Landweber', 'red': 'RED', 'pnp': 'PnP',
                       'oracle': 'DDIR (oracle)', 'dddir': 'SDIR'}[k]
                ttl = f'{lab}\n{e:.3f}'
            ax.set_title(ttl, fontsize=7)
        axs[r, 1].set_ylabel(f'$m={m}$', fontsize=8)
        axs[r, 1].yaxis.set_label_coords(-0.06, 0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/figures/{fname}', bbox_inches='tight', dpi=200); plt.close(fig)


def fig_dddir_vs_m(name='deblurring', fname='dddir_vs_m_deblur.pdf', crop=None):
    R = load(name)
    if R is None:
        return
    ms = R['ms']
    ut = to_img(name, R['u_true'])
    fig, axs = plt.subplots(2, len(ms) + 1, figsize=(7.2, 2.6),
                            gridspec_kw=dict(height_ratios=[1, 1]))
    sl = crop or (slice(None), slice(None))
    for c, k in enumerate(['exact'] + ms):
        u = ut if k == 'exact' else to_img(name, R['recon'][('dddir', k)])
        for r, img in enumerate((u, u[sl])):
            ax = axs[r, c]
            ax.imshow(img, cmap='gray', vmin=ut.min(), vmax=ut.max(), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(r == 1)
                s.set_color('#2a78d6')
        if k == 'exact':
            axs[0, c].set_title('exact', fontsize=7.5)
        else:
            e = np.linalg.norm(R['recon'][('dddir', k)] - R['u_true']) / np.linalg.norm(R['u_true'])
            axs[0, c].set_title(f'$m={k}$\n{e:.3f}', fontsize=7.5)
    axs[1, 0].set_ylabel('zoom', fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/figures/{fname}', bbox_inches='tight', dpi=200); plt.close(fig)


# Side-by-side pair (semi-convergence | step ratio), drawn at final print size:
# text width 6.14 in -> semiconvergence 3.70 in (0.60\\textwidth), lambda 2.30 in (0.375\\textwidth), both 2.55 in tall.
PAIR_H = 2.55
SMALL = {'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
         'legend.fontsize': 6.5, 'lines.linewidth': 1.0}


def fig_semiconv(names, fname='semiconvergence.pdf', size=(3.70, PAIR_H)):
    Rs = [(n, load(n)) for n in names if load(n) is not None]
    with plt.rc_context(SMALL):
        fig, axs = plt.subplots(2, len(Rs), figsize=size, sharex='col', layout='constrained')
        axs = np.array(axs).reshape(2, len(Rs))
        for j, (n, R) in enumerate(Rs):
            for k in ['dddir', 'landweber', 'red', 'pnp']:
                h = R['hist'][k]
                st = STYLE[k]
                lab = st['label'].replace(' (proposed)', '')
                axs[0, j].plot(np.arange(len(h['err'])), h['err'], color=st['c'], ls=st['ls'], label=lab)
                axs[1, j].plot(np.arange(len(h['res'])), h['res'], color=st['c'], ls=st['ls'])
                ns = h['n_stop']
                for a, y in ((axs[0, j], h['err'][ns]), (axs[1, j], h['res'][ns])):
                    a.plot(ns, y, marker=st['m'], color=st['c'], ms=4, mec='white', mew=0.5, ls='none')
            axs[1, j].axhline(R['hist']['dddir']['thr'], color='#6b6b66', ls='--', lw=0.8)
            axs[0, j].set_title(TITLE[n])
            axs[1, j].set_yscale('log'); axs[1, j].set_xlabel('iteration $n$')
            for a in (axs[0, j], axs[1, j]):
                a.set_xscale('symlog', linthresh=10); a.set_xlim(left=0)
                a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            axs[1, j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            lo, hi = axs[1, j].get_ylim()
            if hi / lo < 100:   # less than two decades: label 1-2-5 ticks with plain numbers
                axs[1, j].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
                axs[1, j].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        axs[0, 0].set_ylabel('relative error'); axs[1, 0].set_ylabel('residual')
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc='outside lower center', ncol=4, handlelength=2.2, columnspacing=1.0)
        fig.savefig(f'{OUT}/figures/{fname}'); plt.close(fig)


def fig_lambda(names, fname='lambda.pdf', size=(2.30, PAIR_H)):
    with plt.rc_context(SMALL):
        fig, ax = plt.subplots(figsize=size, layout='constrained')
        for i, n in enumerate(names):
            R = load(n)
            if R is None:
                continue
            h = R['hist']['dddir']
            c = ['#2a78d6', '#eb6834', '#1baf7a'][i]
            r = np.array(h['ratio'])
            ax.plot(np.arange(len(r)), r, color=c, ls=['-', '--'][i % 2], label=TITLE[n])
            ns = min(h['n_stop'], len(r) - 1)
            ax.plot(ns, r[ns], 'o', color=c, ms=4, mec='white', mew=0.5)
        ax.set_xscale('symlog', linthresh=10); ax.set_yscale('log'); ax.set_xlim(left=0)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel('iteration $n$'); ax.set_ylabel('denoiser step / data step')
        ax.legend(loc='lower right')
        fig.savefig(f'{OUT}/figures/{fname}'); plt.close(fig)


def _main():
    main = ['deblurring', 'phase']
    table_main(main, 'main.tex')
    table_main(['eikonal'], 'eikonal.tex')
    table_quality(main)
    fig_err_vs_m(main, 'error_vs_m.pdf')
    sl = fig_mse_stop(main)
    print('MSE slopes', sl)
    fig_recon('deblurring', 'recon_deblur.pdf', crop=(slice(40, 140), slice(70, 170)))
    fig_recon('phase', 'recon_phase.pdf')
    fig_dddir_vs_m(crop=(slice(40, 140), slice(70, 170)))
    fig_semiconv(main)
    fig_lambda(main)
    # eikonal appendix
    if load('eikonal_fine') is not None:
        import problems
        E = problems.eikonal().E
        n = E.n

        def marks(ax):
            ri, rj = E.rec // n, E.rec % n
            ax.plot(rj, ri, '.', color='#3d3d3a', ms=1.4)
            ax.plot(E.src[1], E.src[0], '*', color='#eda100', ms=6)
        fig_recon('eikonal_fine', 'recon_eikonal.pdf', vmin=-0.6, vmax=0.6, cmap='RdBu_r', extra=marks)
        fig_eikonal_summary()


# ------------------------------------------------------------------ verification outputs
def verify_outputs():
    if not os.path.exists('verify_deblur.pkl'):
        return
    V = pickle.load(open('verify_deblur.pkl', 'rb'))
    ms = [4, 16, 64, 256, 1024]
    L = [r'\begin{tabular}{lccccc}\toprule', r' & $m=4$ & $m=16$ & $m=64$ & $m=256$ & $m=1024$ \\ \midrule']
    for name, lab in [('compliant', r'$\tau_0=1.7$, $\vartheta=%.2f$'), ('practical', r'$\tau_0=1.02$ (tuned)')]:
        R = V[name]['rows']
        vt = V[name]['vartheta']
        l = lab % vt if '%' in lab else lab
        fr = [np.mean([r['inE'] for r in R if r['m'] == m]) for m in ms]
        er = [np.mean([r['err'] for r in R if r['m'] == m]) for m in ms]
        nd = [np.mean([r['n'] for r in R if r['m'] == m]) for m in ms]
        L.append(l + r': fraction in $\mathcal{E}_m$ & ' + ' & '.join(f'{x:.2f}' for x in fr) + r' \\')
        L.append(l + r': rel.\ error ($n_{\rm dp}$) & ' + ' & '.join(f'{e:.4f} ({n:.0f})' for e, n in zip(er, nd)) + r' \\')
    L.append(r'\bottomrule\end{tabular}')
    open(f'{OUT}/tables/compliant.tex', 'w').write('\n'.join(L) + '\n')

    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.3))
    for a in axs:
        a.tick_params(labelsize=6.5)
    R = V['compliant']['rows'] + V['practical']['rows']
    c_ = {'compliant': '#2a78d6', 'practical': '#eb6834'}
    for name, mk in [('compliant', 'o'), ('practical', 's')]:
        rr = V[name]['rows']
        vi = [sum(r['viol'] for r in rr if r['m'] == m) for m in ms]
        axs[0].plot(ms, vi, marker=mk, color=c_[name], label={'compliant': r'$\tau_0=1.7$', 'practical': r'$\tau_0=1.02$'}[name])
    axs[0].set_title('(a) monotonicity violations', fontsize=8)
    axs[0].set_ylim(-0.5, max(3, axs[0].get_ylim()[1]))
    rr = V['compliant']['rows']
    b = np.array([r['bound'] for r in rr]); n = np.array([r['n'] for r in rr]); ie = np.array([r['inE'] for r in rr])
    axs[1].scatter(b[ie], n[ie], s=10, color='#2a78d6', label=r'in $\mathcal{E}_m$')
    axs[1].scatter(b[~ie], n[~ie], s=10, color='#eb6834', marker='x', label=r'not in $\mathcal{E}_m$')
    lim = [1, max(b.max(), n.max()) * 1.5]
    axs[1].plot(lim, lim, color='#6b6b66', ls='--', lw=0.9)
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('bound (3.9)'); axs[1].set_ylabel(r'$n_{\rm dp}$')
    axs[1].set_title(r'(b) termination bound, $\tau_0=1.7$', fontsize=8); axs[1].legend(fontsize=6.5)
    for j, (cl, mk) in zip(sorted(V['stab']), [('#2a78d6', 'o'), ('#1baf7a', 's')]):
        y = [V['stab'][j][m] for m in ms]
        axs[2].loglog(ms, y, marker=mk, color=cl, label=f'$j={j}$')
    y0 = V['stab'][sorted(V['stab'])[0]][ms[0]]
    axs[2].loglog(ms, y0 * ms[0] / np.array(ms), color='#6b6b66', ls=':', lw=1, label='slope $-1$')
    axs[2].set_title(r'(c) $\mathbb{E}\|u_j^m-u_j\|^2$', fontsize=8); axs[2].legend(fontsize=6.5)
    A = V['abl_d1']
    for lab, cl, mk in [('penalty', '#2a78d6', 'o'), ('mse', '#eda100', '^')]:
        mm = [16, 256, 1024]
        y = [np.mean([r['err'] for r in A if r['m'] == m and r['cnn'] == lab]) for m in mm]
        axs[3].plot(mm, y, marker=mk, color=cl, label={'penalty': 'with (D1) penalty', 'mse': 'MSE only'}[lab])
    axs[3].set_xscale('log'); axs[3].set_title('(d) (D1) ablation: rel. error', fontsize=8); axs[3].legend(fontsize=6.5)
    for ax in (axs[0], axs[2], axs[3]):
        ax.set_xscale('log'); ax.set_xlabel('$m$'); ax.minorticks_off()
        t = ms if ax is not axs[3] else [16, 256, 1024]
        ax.set_xticks(t, [str(x) for x in t])
    axs[0].legend(fontsize=6.5)
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/figures/verify.pdf', bbox_inches='tight'); plt.close(fig)
    print('traj monotonicity ratio', V.get('traj_mono'))
    for name in ['compliant', 'practical']:
        rr = V[name]['rows']
        print(name, 'vartheta', V[name]['vartheta'], 'C_D', V[name]['C_D'], 'viol', sum(r['viol'] for r in rr),
              'inE', np.mean([r['inE'] for r in rr]),
              'ressum ok', all(r['ressum'] <= r['ressum_bound'] for r in rr if r['inE']) if name == 'compliant' else '',
              'n<=bound', all(r['n'] <= r['bound'] for r in rr if r['inE']) if name == 'compliant' else '')
    print('stab', V['stab'])
    print('abl', {(lab, m): (round(np.mean([r['err'] for r in A if r['m'] == m and r['cnn'] == lab]), 4),
                             np.mean([r['n'] for r in A if r['m'] == m and r['cnn'] == lab]))
                  for lab in ['penalty', 'mse'] for m in [16, 256, 1024]})


def ablation_table():
    L = [r'\begin{tabular}{llccc}\toprule', r'Problem & Variant & $m=16$ & $m=256$ & $m=1024$ \\ \midrule']
    if os.path.exists('verify_deblur.pkl'):
        A = pickle.load(open('verify_deblur.pkl', 'rb'))['abl_d1']
        for i, (lab, txt) in enumerate([('penalty', 'CNN with (D1) penalty'), ('mse', 'CNN, MSE training only')]):
            c = []
            for m in [16, 256, 1024]:
                rr = [r for r in A if r['m'] == m and r['cnn'] == lab]
                st = r'$^{\ast}$' if np.mean([r['reason'] == 'budget' for r in rr]) > 0.5 else ''
                c.append(f"{np.mean([r['err'] for r in rr]):.4f} ({np.mean([r['n'] for r in rr]):.0f}{st})")
            L.append(('Deblurring' if i == 0 else '') + f' & {txt} & ' + ' & '.join(c) + r' \\')
        L.append(r'\midrule')
    if os.path.exists('verify_phase_s.pkl'):
        rows = pickle.load(open('verify_phase_s.pkl', 'rb'))
        for i, s in enumerate(sorted({r['s'] for r in rows})):
            c = []
            for m in [16, 256, 1024]:
                rr = [r for r in rows if r['m'] == m and r['s'] == s]
                st = r'$^{\ast}$' if np.mean([r['reason'] == 'budget' for r in rr]) > 0.5 else ''
                c.append(f"{np.mean([r['err'] for r in rr]):.3f} ({np.mean([r['n'] for r in rr]):.0f}{st})")
            L.append(('Phase retrieval' if i == 0 else '') + f' & $s={s}$ & ' + ' & '.join(c) + r' \\')
    L.append(r'\bottomrule\end{tabular}')
    open(f'{OUT}/tables/ablation.tex', 'w').write('\n'.join(L) + '\n')


def eikonal_table():
    R = load('eikonal_fine'); R2 = load('eikonal')
    if R is None:
        return
    ms = R['ms']
    L = [r'\begin{tabular}{lccccc}\toprule', r'Method & $m=4$ & $m=16$ & $m=64$ & $m=256$ & $m=1024$ \\ \midrule',
         r'\multicolumn{6}{l}{relative error to $u^\dagger$, data from the finer grid (mean stopping index in parentheses)}\\ \midrule']
    L += block(R, 'eikonal')[1:]
    L.append(r'\midrule')
    L.append(r'\multicolumn{6}{l}{relative error of \texttt{SDIR} to the exact-data limit $\hat u$}\\ \midrule')
    eh = [agg(R, 'dddir', m, 'err_uhat')[0] for m in ms]
    L.append(r'\texttt{SDIR}, finer-grid data & ' + ' & '.join(f'{x:.3f}' for x in eh) + r' \\')
    if R2 is not None:
        eh2 = [agg(R2, 'dddir', m, 'err_uhat')[0] for m in ms]
        e2 = [agg(R2, 'dddir', m, 'err')[0] for m in ms]
        L.append(r'\texttt{SDIR}, same-grid data & ' + ' & '.join(f'{x:.3f}' for x in eh2) + r' \\')
        L.append(r'\midrule')
        L.append(r'\multicolumn{6}{l}{relative error to $u^\dagger$, data from the reconstruction grid}\\ \midrule')
        L.append(r'\texttt{SDIR}, same-grid data & ' + ' & '.join(f'{x:.3f}' for x in e2) + r' \\')
    L.append(r'\bottomrule\end{tabular}')
    open(f'{OUT}/tables/eikonal.tex', 'w').write('\n'.join(L) + '\n')


def setup_table():
    import config
    T = config.TUNED
    d, p = T['deblurring'], T['phase']
    L = [r'\begin{tabular}{lcc}\toprule', r' & Deblurring & Phase retrieval \\ \midrule',
         r'Unknown $u$ & $256\times256$ & $64\times64$ (disc, 3096 pixels) \\',
         r'Data $v_i$ & $256\times256$ & $3806$ \\']
    for nm in ['deblurring', 'phase']:
        R = load(nm)
    import problems as pb
    from core import sig_entry
    sd = sig_entry(pb.deblurring(lambda u: u)); sp = sig_entry(pb.phase_retrieval())
    L += [f'Noise std.\\ per entry $\\varsigma$ & {sd:.2e} & {sp:.2e} \\\\',
          r'Denoiser $D$ & CNN & TV prox, $w=' + f"{p['tv']}" + r'$ \\',
          r'Initial guess $u_0$ & $0$ & $c\,\mathbb{1}_{\rm disc}$ \\',
          f"Tolerance $\\tau_0$, $\\varrho$ & {d['base']['tau0']}, {d['base']['rho'] if 'rho' in d['base'] else 0.1} & {p['base']['tau0']}, {p['base']['rho'] if 'rho' in p['base'] else 0.1} \\\\",
          f"Budget $N_0$ & {d['base']['N0']} & {p['base']['N0']} \\\\",
          f"$\\gamma_0$, $\\gamma_1$ & {d['base']['gamma0']}, {d['base']['gamma1']} & {p['base']['gamma0']}, {p['base']['gamma1']} \\\\",
          f"$\\nu_0$, $\\nu_1$, $\\nu_2$ & {d['base']['nu0']}, {d['base']['nu1']}, {d['base']['nu2']} & {p['base']['nu0']}, {p['base']['nu1']}, {p['base']['nu2']} \\\\",
          f"$\\alpha_0$, $s$ in $\\alpha_n=\\alpha_0(n+1)^{{-s}}$ & {d['base']['alpha0']}, {d['base']['s']} & {p['base']['alpha0']}, {p['base']['s']} \\\\",
          f"RED: $\\omega_{{\\rm RED}}$ & {d['red']['omega']} & {p['red']['omega']} \\\\",
          f"PnP: $\\beta$ & {d['pnp']['beta']} & {p['pnp']['beta']} \\\\",
          r'\bottomrule\end{tabular}']
    open(f'{OUT}/tables/setup.tex', 'w').write('\n'.join(L) + '\n')


def fig_eikonal_summary():
    R = load('eikonal_fine')
    ms = R['ms']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for k in METHODS:
        st = STYLE[k]
        mu = [agg(R, k, m, 'err')[0] for m in ms]
        sd = [agg(R, k, m, 'err')[1] for m in ms]
        axs[0].errorbar(ms, mu, sd, color=st['c'], marker=st['m'], ls=st['ls'], ms=4.5, capsize=2, label=st['label'],
                        mfc='white' if k == 'oracle' else st['c'], lw=1.3)
    eh = [agg(R, 'dddir', m, 'err_uhat')[0] for m in ms]
    axs[0].plot(ms, eh, color='#2a78d6', marker='o', mfc='white', ls=(0, (1, 1)), lw=1.3, label=r'SDIR, error to $\hat u$')
    axs[0].axhline(agg(R, 'ddir1', 1, 'err')[0], color='#6b6b66', ls='--', lw=1.0, label='DDIR, single measurement')
    axs[0].set_xscale('log'); axs[0].set_xticks(ms, [str(m) for m in ms]); axs[0].minorticks_off()
    axs[0].set_xlabel('number of measurements $m$'); axs[0].set_ylabel('relative error')
    for k in ['dddir', 'landweber', 'red', 'pnp']:
        h = R['hist'][k]; st = STYLE[k]
        axs[1].plot(np.arange(len(h['err'])), h['err'], color=st['c'], ls=st['ls'], lw=1.3)
        axs[1].plot(h['n_stop'], h['err'][h['n_stop']], marker=st['m'], color=st['c'], ms=6, mec='white', ls='none')
    axs[1].set_xscale('symlog', linthresh=10); axs[1].set_xlim(left=0); axs[1].set_xlabel('iteration $n$'); axs[1].set_ylabel('relative error')
    axs[1].set_title('$m=64$, first realisation', fontsize=8.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f'{OUT}/figures/eikonal_results.pdf', bbox_inches='tight'); plt.close(fig)



if __name__ == '__main__':
    _main()
    verify_outputs()
    ablation_table()
    eikonal_table()
    setup_table()
