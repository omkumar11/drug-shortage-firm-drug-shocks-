[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23630836.svg)](https://doi.org/10.5281/zenodo.23630836)

# Code for "Manufacturer-Level and Drug-Level Shocks on the U.S. Drug Shortage List: Identification, Persistence and Supplier Redundancy" (Kumar & Salwekar)

Everything in the manuscript is produced by `python run_all.py` from public FDA data.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py
```

| step | script | output |
|---|---|---|
| unit tests (equations and theorems = code) | `tests/test_model.py`, `tests/test_theory.py`, `tests/test_sbc.py`, `tests/test_units.py`, `tests/test_panel.py`, `tests/test_long_panel.py`, `tests/test_inspections.py` | pytest report |
| download raw data | `download_data.py` | `data/raw/*.zip`, `data/raw/download_log.txt` (URL + UTC date of every download) |
| build tables | `pipeline/build_data.py` | `data/processed/{presentations,cells,drugs,firms}.csv`, `build_summary.json` |
| model-free results | `analysis/descriptives.py` | `results/descriptives.json` |
| model-free results for every outcome definition | `analysis/outcome_variants.py` | `results/outcome_variants.json` |
| listing selection: unlisted approved suppliers (Orange Book holders) added to the graph | `analysis/unlisted_suppliers.py` | `results/unlisted_suppliers.json` |
| review-round-2 checks: Schedule II excluded, corporate groups, unlisted holders of listed firms, drug-rate imputation, firm-weighted co-unavailability | `analysis/robustness_v4.py` | `results/robustness_v4.json` |
| two-tier latent-shock model, recovery study, calibration of the ordering, sensitivity fits | `model/twotier.py`, `analysis/fit_model.py` | `results/model_fit.json`, `draws_*.npz`, `firm_posteriors.csv`, `market_posteriors.csv`, `cell_attribution.csv` |
| formal results (Section 2.6): moment estimator, outage and overlap quantities, Corollary 1 (with posterior predictive check) and Corollary 2 on the observed graph | `model/theory.py`, `analysis/theory_analysis.py` | `results/theory.json` |
| simulation-based calibration of the full sampler (Supplementary S1): Geweke kernel test, short run, thinned main run, start-dependence check, every-drug-shocked start on the observed data | `analysis/sbc.py` | `results/sbc.json` |
| independent collapsed-W sampler on the flagged SBC data sets and the observed data | `analysis/sbc_collapsed.py` | `results/sbc_collapsed.json` |
| prior draws of the SBC batches (no sampler run; Supplementary S1) | `analysis/sbc_prior_draws.py` | `results/sbc_prior_draws.json` |
| covariate regressions | `analysis/regressions.py` | `results/regressions.json` |
| FDA inspection classifications linked to firms by company name under a primary and a broad rule (OAI firms; Section 3.6, Supplementary S6) | `analysis/inspections.py` | `results/inspections.json`, `results/firm_inspections_{strict,broad}.csv`, `results/oai_firms_table_{strict,broad}.tex`, `data/raw/inspections_drugs_extract.csv.gz` |
| leave-one-cell-out prediction (firm vs drug record) and the firm tests without demand-increase cells (Sections 3.2, 4.1) | `analysis/predictive_check.py` | `results/predictive.json` |
| three openFDA snapshots (Section 2.4, Supplementary Section S9, Table S10, Proposition 3): the two archived openFDA shortage exports of 2025-06-28 and 2026-01-27 (Internet Archive; `data/raw/archive/`, fetched by `download_data.py`), recurrence of the model-free statistics, persistence of firm and drug covariances across dates, synchronised change, prospective and out-of-sample prediction, and the dynamic simulation that gives the size and power of the change tests | `analysis/panel.py`, `model/theory.py` (`lagged_covariances`) | `results/panel.json` |
| quarterly panel 2019-2026 (Section 2.4, Section 3.3, Figure 3): the 130 Internet Archive captures of the FDA Drug Shortage Database CSV download (`data/raw/archive_csv/`, fetched and checked against `manifest.csv` by `download_archive_csv.py`), free-text availability coding (audit table `results/fda_csv_availability_coding.csv`), drug names linked across renamings, validation against the openFDA exports, recurrence, transition and change tests, first-listing regression, new-cell lag covariance, batched updates | `analysis/fda_csv.py`, `analysis/long_panel.py` | `results/long_panel.json` |
| figures | `analysis/figures.py` | `figures/fig*.pdf`, `figures/fig*.png` (600 dpi) |
| LaTeX macros | `make_numbers.py` | `results/results.json`, `results/numbers.tex` (also copied to `../manuscript/numbers.tex`, overwriting it) |

Settings: `analysis/common.py` (firm unit; `FIRM_UNIT=labeler|marketer|group|raw` switches the primary unit; `load_cells(firm_unit)` for one call). Seeds are fixed in each script.
The sampler runs 4 chains x 40,000 iterations from over-dispersed starting values; the recovery study refits 30 simulated data
sets under four truths and the calibration study 30 data sets under each of 26 equal-contribution nulls (plus 70 at the least favourable one), and the simulation-based calibration refits 200 data sets drawn from the prior twice (8,000 and 110,000 iterations) plus 82 of them from a second start, in parallel over `SBC_WORKERS` processes (default: all but two cores, at most 8), so `run_all.py` takes
about four hours on an 8-core Apple M-series laptop with 7 parallel workers for the calibration steps depending on the machine and on what else is running (the time of the last run is in `results/run_all.log`).

Figure files vs manuscript numbering: `fig6_schematic` = Figure 1, `fig1_incidence` = Figure 2, `fig8_redundancy` = Figure 3;
Supplementary: `fig5_recovery` = Figure S1, `fig7_timeline` = Figure S2, `fig4_validation` = Figure S3, `fig2_dispersion` = Figure S4,
`fig3_posteriors` = Figure S5.

Data sources (all public; see `download_log.txt`):
openFDA drug shortages, NDC Directory and drug enforcement reports (https://open.fda.gov, daily exports), and the
FDA Orange Book data files (https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files).
The FDA Data Dashboard's "Entire Inspections Dataset" (https://datadashboard.fda.gov/oii/cd/inspections.htm, Download Dataset >
Entire Inspections Dataset; the direct link https://datadashboard.fda.gov/InspectionsDataset.xlsx refuses requests made outside a
browser session) is downloaded by hand into `data/raw/InspectionsDataset.xlsx`; `analysis/inspections.py` writes the filtered
extract `data/raw/inspections_drugs_extract.csv.gz` (all drug inspections) from it and reads the extract when the workbook is absent,
so the extract ships with the code and the workbook is archived with the release.
The manuscript's numbers correspond to the export dates recorded in `data/processed/build_summary.json` (shortage export 2026-10-13; NDC and enforcement exports carry the export date 2026-10-14 inside the files and were downloaded on 2026-10-15);
later exports will differ because the shortage list changes daily.
