# Stage 2 — Controlled DDPM vs stock DDPM on the SAXS benchmark

Prerequisite: Stage 1 verified (stock ensembles reproduce the paper's SAXS
agreement). Stage 2 changes **only the sampler**: `DDPMSampler` →
`ControlledDDPMSampler` with the reduced controller

    nu(t) = alpha * g(t)   on the late-reverse window [0, t_w]

implemented as the two-line DDPM modification (score coefficient
`beta_t -> (beta_t + nu_t^2)/2`, injected noise std `-> nu_t`). Experimental
data, Debye forward model, and metrics are byte-identical to Stage 1.

## Step 1 — Generate controlled (+ paired stock) ensembles

```bash
python generate_stage2.py \
    --fasta data/saxs_scattering/experiment/sequences.fasta \
    --out runs/stage2_alpha2.5_tw0.2 \
    --conformations 400 --alpha 2.5 --window 0.2
```

* Validated protein defaults: `--alpha 2.5 --window 0.2` (boost regime).
  A damp arm (`--alpha 0.5`) is the natural second condition.
* By default the script also regenerates a **paired stock arm** under
  `runs/stage2_.../stock_paired/` using the *same per-protein seed*
  (`seed + crc32(name)`), so both arms consume identical Gaussian draws at
  every reverse step (common random numbers — per-protein differences are
  then nearly free of sampling noise). Use `--no-stock` to skip this and
  compare against the Stage-1 stock statistically instead.
* Layout matches Stage 1: `distance_maps/<name>.npy`, `rg/<name>.npy`.
  Resumable; `--only ash1,p53` for a quick smoke test.

## Step 2 — Curves and metrics (identical commands to Stage 1)

```bash
python batch_curves.py \
    --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
    --dm-dir runs/stage2_alpha2.5_tw0.2/distance_maps \
    --out curves/stage2_alpha2.5_tw0.2

python curve_benchmark.py \
    --exp-dir data/saxs_scattering/experiment \
    --our-curves curves/stage2_alpha2.5_tw0.2 \
    --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
    --out results/curve_benchmark_stage2

# optional: metrics for the paired stock arm as well
python batch_curves.py \
    --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
    --dm-dir runs/stage2_alpha2.5_tw0.2/stock_paired/distance_maps \
    --out curves/stage2_alpha2.5_tw0.2_stock
python curve_benchmark.py \
    --exp-dir data/saxs_scattering/experiment \
    --our-curves curves/stage2_alpha2.5_tw0.2_stock \
    --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
    --out results/curve_benchmark_stage2_stock
```

## Step 3 — Controlled vs stock comparison

```bash
python compare_stage2.py \
    --stock-metrics results/curve_benchmark/curve_metrics.csv \
    --ctrl-metrics  results/curve_benchmark_stage2/curve_metrics.csv \
    --stock-rg runs/stage1_scattering/rg \
    --ctrl-rg  runs/stage2_alpha2.5_tw0.2/rg \
    --mff-csv data/saxs_scattering/experiment/mff_analysis_all.csv \
    --out results/stage2_comparison
```

(use `results/curve_benchmark_stage2_stock/curve_metrics.csv` and
`runs/stage2_.../stock_paired/rg` as `--stock-*` for the CRN-paired analysis)

Outputs:

* `stage2_per_protein.csv` — per-protein χ²ᵣ (both arms), Rg vs experiment
* `stage2_comparison.pdf/.png` — χ²ᵣ scatter (stock vs controlled) and
  |Rg − Rg_exp| scatter
* printed: median Δχ²ᵣ, fraction improved, sign test + Wilcoxon p-values,
  Rg RMSE stock → controlled

## What to look for

* **SAXS χ²ᵣ**: if the controller moves the ensemble toward experiment,
  median Δχ²ᵣ < 0 with a significant Wilcoxon/sign p-value, and the
  improvement concentrates on proteins where stock was already reasonable
  (badly-fitting proteins like `mbp` are model-limitation cases, not
  sampler-limitation cases).
* **Rg**: RMSE stock → controlled against the MFF values; the theory predicts
  the boost regime increases effective diversity (wider Rg distributions,
  shifted means for over-compact ensembles).
* **Rg width**: `rg_std_ctrl` vs stock Rg std per protein — the controller's
  primary predicted effect is on ensemble *spread*, so even where mean Rg is
  unchanged, watch the distribution width.
