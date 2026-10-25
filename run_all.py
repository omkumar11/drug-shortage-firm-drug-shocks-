"""
run_all.py -- reproduce every number, table and figure in the manuscript from scratch.

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python run_all.py            # about 4 h on an 8-core Apple M-series laptop (SBC_WORKERS controls the parallel calibration steps) (model fit, sensitivities, recovery, calibration and SBC dominate; SBC uses SBC_WORKERS processes)
    .venv/bin/python run_all.py --quick    # skip the re-download if raw files exist (default) and run tests only

Steps: tests -> download raw data (and the archived CSV captures) -> build processed tables -> descriptives -> outcome variants -> unlisted suppliers -> robustness checks -> model fit -> theory -> SBC -> collapsed sampler -> SBC prior draws -> regressions -> inspections -> predictive check -> three snapshots -> quarterly panel -> leave-one-firm-out -> figures -> numbers.tex
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ("unit tests", [PY, "-m", "pytest", "-q", "tests"]),
    ("download raw data", [PY, "download_data.py"]),
    ("download the archived FDA shortage CSV captures", [PY, "download_archive_csv.py"]),
    ("build processed tables", [PY, "pipeline/build_data.py"]),
    ("model-free descriptives", [PY, "analysis/descriptives.py"]),
    ("model-free statistics for every outcome definition", [PY, "analysis/outcome_variants.py"]),
    ("unlisted suppliers added as available cells (listing selection)", [PY, "analysis/unlisted_suppliers.py"]),
    ("robustness checks of review round 2 (Schedule II, corporate groups, unlisted-supplier variants)", [PY, "analysis/robustness_v4.py"]),
    ("two-tier model fit", [PY, "analysis/fit_model.py"]),
    ("theory: moment estimator, outage and overlap results", [PY, "analysis/theory_analysis.py"]),
    ("simulation-based calibration of the sampler", [PY, "analysis/sbc.py"]),
    ("independent collapsed-W sampler on the flagged calibration data sets", [PY, "analysis/sbc_collapsed.py"]),
    ("prior draws of the calibration batches", [PY, "analysis/sbc_prior_draws.py"]),
    ("regressions", [PY, "analysis/regressions.py"]),
    ("FDA inspection classifications linked to firms", [PY, "analysis/inspections.py"]),
    ("leave-one-cell-out prediction and demand-increase cells dropped", [PY, "analysis/predictive_check.py"]),
    ("three snapshots: recurrence, persistence, synchronised change, prediction", [PY, "analysis/panel.py"]),
    ("quarterly panel 2019-2026 from the archived FDA CSV captures", [PY, "analysis/long_panel.py"]),
    ("leave-one-firm-out checks of the headline statistics", [PY, "analysis/leave_one_out.py"]),
    ("figures", [PY, "analysis/figures.py"]),
    ("numbers.tex", [PY, "make_numbers.py"]),
]


def main():
    t0 = time.time()
    for name, cmd in STEPS:
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"step '{name}' failed with code {r.returncode}"); sys.exit(r.returncode)
    print(f"\nall steps completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
