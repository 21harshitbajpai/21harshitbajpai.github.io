# Experiments for "SDIR with repeated measurements" (Section 4 and Appendix B)

Every number, table and figure in Section 4 and Appendix B is produced by the code in this folder.

Requirements: Python 3.11+, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `numba`, and `torch` (CPU is enough).

## Files

| File | Purpose |
| --- | --- |
| `core.py` | SDIR (step sizes (2.9)–(2.10), no rescaling), Landweber, RED, PnP; nested measurement streams (Welford); stopping rules |
| `problems.py` | deblurring (φ = 1.5, periodic FFT), phase retrieval (Radon matrix, 60 angles), eikonal wrapper |
| `eikonal.py` | fast marching with exact source initialisation and exact discrete adjoint (numba) |
| `train_cnn.py`, `finetune.py` | training of the residual CNN: 3000 MSE steps, then 250 steps with the monotonicity penalty |
| `cnn_mse.pt`, `cnn.pt` | trained networks (after MSE training / after fine-tuning) |
| `config.py` | tuned parameters of all methods (selected on noise seed 999, never used for the reported results) |
| `run_all.py` | all methods, all m, nested streams with seeds 1000, 1001, … |
| `compute_uhat.py` | exact-data limit û (Algorithm 2, 3·10⁴ iterations) |
| `verify.py` | theory-compliant run, monotonicity, termination bound, stability, (D1) ablation, αₙ ablation |
| `denoiser_props.py` | Table 6 (empirical properties of the denoisers) |
| `make_outputs.py` | all tables (`paper_out/tables`) and figures (`paper_out/figures`) |
| `fill.py`, `sentences.py`, `*_template.tex` | fill the numbers in Section 4 and Appendix B from the result files |
| `tuning/` | the grid searches used to select the parameters |

## Reproducing

```bash
python train_cnn.py cnn.pt 3000 0 && python finetune.py 250    # or use the provided .pt files
python compute_uhat.py eikonal && python compute_uhat.py eikonal_fine && python compute_uhat.py phase
python run_all.py deblurring 5
python run_all.py phase 10
python run_all.py eikonal 10 && python run_all.py eikonal_fine 10
python verify.py deblur 5 && python verify.py phase_s 5
python denoiser_props.py
python make_outputs.py paper_out && python fill.py
```

Run times on one core of a 2.1 GHz Xeon: deblurring about 20 min per realisation (dominated by RED and PnP running to the budget at m = 1024), phase retrieval about 6 min, eikonal about 4 min.
