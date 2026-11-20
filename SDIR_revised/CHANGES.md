# Revision log (numerical section)

`main_revised.tex` is your uploaded file with (1) the edits from your new PDF ported in and (2) a rewritten Section 4 plus a new Appendix B. `diff_vs_uploaded.pdf` shows every change against the uploaded `.tex`.

## Your PDF edits, ported into the .tex
"This work" in Table 1; shortened κ sentence; extended outline sentence; Rockafellar remark moved into Section 2.3 (Appendix A removed); "see Section 3.4"; removed parenthetical references in the safeguard paragraph; reworded proofs of Lemmas 3.2, 3.8, 3.9 and Theorems 3.3, 3.12; Algorithm 2 caption; Remark 3.7 title removed; "stabilizer"; Proposition 3.14 and Theorem 3.15 wording; Remark after Theorem 3.15 shortened; "No early stopping" lemma merged into Appendix A, with Remark A.6 at its end. Small grammar fixes: "conclusions", "by induction", "Term III is".

## Numerical section
* No division by B̂: γ0, γ1, ν0 are reported directly (Table 3); the text explains why the normalisation is unnecessary.
* Deblurring with φ = 1.5. Eikonal moved to Appendix B; deblurring and phase retrieval stay in Section 4.
* All experiments re-implemented and rerun: new monotone-residual CNN, parameters tuned on a held-out noise realisation (seed 999), nested noise streams.
* All earlier TODOs and placeholders are filled: theory-compliant run (Table 7), verification figure (Figure 8), ablations (Table 8), convergence to û.
* Eikonal: fast marching with exact adjoint, exact source initialisation, data from a finer grid (no inverse crime), comparison with same-grid data, and error to û.

## Reproducing
See `experiments/README.md`. The Radon matrix cache (90 MB) is rebuilt automatically on the first run.

## Rename to SDIR
The method is now called SDIR (statistical denoiser-driven iterative regularization) throughout: title, running head, abstract, contributions, Sections 2–4, Appendix B, all tables, and all figure legends and panel titles. Title: "SDIR: Statistical Denoiser-Driven Iterative Regularization with Repeated Measurements under Pseudo-Contractivity".
