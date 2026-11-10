# HBNN_Capstone: Hierarchical Bayesian Neural Networks with Dynamic Price Elasticities

Price and promotion analytics for CPG Client using hierarchical Bayesian models, with a Kalman-filter extension
for time-varying elasticities (HBNN-KF). Four models are compared on identical series and weeks:

| Model | Description |
|---|---|
| M0 | Independent series-level log-log OLS (industry baseline) |
| M1 | Hierarchical Bayes (series → brand → category), Gibbs sampler |
| M2 | HBNN: M1 + Bayesian neural prior on PPG attributes, sign-constrained elasticities (NumPyro, SVI) |
| M3 | HBNN-KF: M2 + differentiable two-state Kalman filter (local level + time-varying elasticity) |
| M3L | M3 with the level state only, used to isolate elasticity dynamics (RQ2) |

**Headline results (52-week rolling-origin backtest):** pooling (M1) cuts sparse-series median WMAPE from
18.9% to 12.3%; M2/M3 do not improve on M1 for sparse series; M2 gives a small significant gain on data-rich
series; M3 is best calibrated; a drifting baseline explains most of the dynamic model's predictive gain, with
modest evidence for time-varying elasticities. See the final report for details.
QM640 Data Analytics Capstone, Walsh College.

## What is in this repository

| Included | Not included (licensed / confidential) |
|---|---|
| All code (`src/`), configuration (`configs/config.yaml`), `Makefile` | The raw syndicated data cube (`DataCube_v6_Q126.xlsx`) |
| Synthetic data with known elasticities (`data/synthetic/`) | The processed weekly panel (`data/processed/panel_clean.csv`) |
| Unit test of the Kalman filter (`tests/`) | All generated figures and result tables (`reports/figures/`, `reports/tables/`) |
| `configs/masking.example.yaml` (template) | `configs/masking.yaml` (real brand and retailer names) |

The data are licensed syndicated retail measurement data supplied by a CPG manufacturer and
cannot be redistributed. All brand and retailer names in the outputs are masked
(Own Brand A/B, Competitor C–F, Private Label; Retailer 1/2). The `.gitignore` blocks the data
files and the masking map from being committed.

## Reproduce the results

Requires Python 3.11 or later.

```
pip install -r requirements.txt
cp configs/masking.example.yaml configs/masking.yaml   # then fill in the raw names
# place the licensed cube at data/raw/DataCube_v6_Q126.xlsx
make all          # prep -> eda -> backtest -> evaluate -> analysis -> hbnn -> final  (under an hour on a CPU)
```

Individual stages: `make prep`, `make eda`, `make backtest`, `make evaluate`, `make analysis`,
`make hbnn` (M2/M3/M3L backtest), `make final` (RQ1–RQ4 tests, elasticities, importance, scenarios),
`make validate` (SVI vs NUTS), `make test` (Kalman filter vs statsmodels).

### Without the licensed data

```
make demo         # simulate a CPG panel with known elasticities and run the recovery study
```
All thresholds, backtest origins and the random seed (640) are set in `configs/config.yaml`.
A different data cube with the same column layout can be passed directly:
`python src/prep.py path/to/cube.xlsx`.

## Layout

```
configs/config.yaml          settings, thresholds, backtest origins, sampler settings, seed
configs/masking.example.yaml template for the private name-masking map
data/raw/                    licensed data cube goes here (git-ignored)
data/processed/              cleaned panel written by prep.py (git-ignored)
src/prep.py                  validation, cleaning, masking, feature engineering
src/eda.py                   exploratory figures
src/models.py                M0 (series OLS) and M1 (hierarchical Bayes, Gibbs sampler)
src/backtest.py              rolling-origin backtest, natural and truncated sparse series
src/evaluate.py              accuracy metrics and RQ1 tests
src/analysis.py              interim elasticities, rolling elasticities, lift, media (M1)
src/hbnn.py                  M2, M3, M3L: neural prior, two-state Kalman layer, RTS smoother
src/backtest_hbnn.py         rolling-origin backtest of M2/M3/M3L, leave-future-out densities
src/evaluate_final.py        metrics for M0–M3, RQ1 tests (Wilcoxon, t, Diebold–Mariano), RQ2 ELPD tests
src/final_analysis.py        full-sample fits: elasticity paths, RQ2 drivers, RQ3 lift, RQ4 media + prior
                             sensitivity, permutation importance, decomposition, scenarios
src/train_metrics.py         in-sample (training) accuracy of M0–M3
src/validate_svi.py          SVI vs NUTS on a 10% subsample
src/synthetic.py, recovery.py  synthetic panel with known elasticities; recovery study
src/diagrams.py              workflow and architecture figures
tests/test_kalman.py         JAX Kalman filter log-likelihood vs statsmodels
data/synthetic/              public synthetic panel and ground truth
reports/figures, reports/tables   outputs
```

## Output map

Outputs are not versioned: every script writes its figures and tables to `reports/` when run.

| Stage | Outputs |
|---|---|
| `prep.py` | `reports/tables/cleaning_log.json` |
| `eda.py` | `reports/figures/fig1`–`fig8`, `reports/tables/eda_stats.json` |
| `backtest.py`, `evaluate.py` | `reports/figures/fig12_backtest_wmape.png`, `reports/tables/series_metrics.csv`, `rq1_results.json` |
| `analysis.py` | `reports/figures/fig9`–`fig11`, `reports/tables/elasticities_full.csv`, `rq2_window_change.csv`, `analysis_results.json` |
| `backtest_hbnn.py` | `reports/tables/lfo_density.csv`, `svi_diagnostics.csv` |
| `evaluate_final.py` | `reports/figures/fig13`, `fig15`, `reports/tables/final_results.json`, `series_metrics_final.csv`, `rq2_elpd_by_series.csv` |
| `final_analysis.py` | `reports/figures/fig14`, `fig16`–`fig18`, `reports/tables/final_analysis.json`, `elasticities_final.csv` |
| `recovery.py`, `validate_svi.py`, `train_metrics.py` | `fig19`, `recovery_results.json`, `svi_vs_nuts.json`, `train_metrics.json` |
