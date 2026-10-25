"""Tests for analysis/panel.py (three snapshots of the shortage list)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import panel  # noqa: E402
import analysis.descriptives as de  # noqa: E402


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "processed", "presentations.csv")), reason="processed data not built")
def test_current_wave_reproduces_primary_cells():
    from analysis.common import load_cells
    cells, _, _, _ = load_cells("company")
    w, date, _, _ = panel.wave_cells(panel.WAVES[-1][1])
    a = cells[["firm_unit", "drug", "n_presentations", "unavailable"]].sort_values(["firm_unit", "drug"]).reset_index(drop=True)
    b = w[["firm_unit", "drug", "n_presentations", "unavailable"]].sort_values(["firm_unit", "drug"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def toy():
    A = pd.DataFrame(dict(firm_unit=["F", "F", "F", "G", "G"], drug=["a", "b", "c", "a", "d"], n_presentations=1, unavailable=[1, 0, 1, 0, 1]))
    B = pd.DataFrame(dict(firm_unit=["F", "F", "G", "G", "H"], drug=["a", "b", "a", "e", "a"], n_presentations=[1, 2, 1, 1, 1], unavailable=[1, 1, 0, 1, 0]))
    return A, B


def test_transition_leaves_own_drug_out():
    A, B = toy(); M = panel.transition(A, B).set_index(["firm_unit", "drug"])
    # F,a: F's other drugs at A are b (0) and c (1) -> 1/2; a's other firms at A: G (0) -> 0; own state unav
    assert M.loc[("F", "a"), "fs"] == pytest.approx(0.5) and M.loc[("F", "a"), "ds"] == 0 and M.loc[("F", "a"), "state"] == "unav"
    # F,b: others a (1), c (1) -> 1; b has no other firm at A -> NaN; own state avail
    assert M.loc[("F", "b"), "fs"] == 1 and np.isnan(M.loc[("F", "b"), "ds"]) and M.loc[("F", "b"), "state"] == "avail"
    # G,e: new cell; G's drugs at A: a (0), d (1) -> 1/2 (nothing left out)
    assert M.loc[("G", "e"), "fs"] == pytest.approx(0.5) and M.loc[("G", "e"), "state"] == "new"
    # H,a: H not listed at A -> fs NaN; drug a's firms at A: F (1), G (0) -> 1/2
    assert np.isnan(M.loc[("H", "a"), "fs"]) and M.loc[("H", "a"), "ds"] == pytest.approx(0.5)


def test_permutation_recomputation_identity():
    """With the identity permutation, recomputing fs from the lagged firm totals gives the original fs."""
    A, B = toy(); M = panel.transition(A, B)
    sub = M[M["state"] != "new"].dropna(subset=["fs"])
    fs = (sub["f_sum_lag"] - sub["own_lag"]) / (sub["f_size_lag"] - 1)
    np.testing.assert_allclose(fs.values, sub["fs"].values)


def test_fit_logit_matches_closed_form():
    # one binary regressor: the logistic coefficient is the log odds ratio of the 2x2 table
    x = np.array([0] * 10 + [1] * 10); y = np.array([1] * 3 + [0] * 7 + [1] * 6 + [0] * 4)
    b = panel.fit_logit(np.column_stack([np.ones(20), x]), y.astype(float))
    assert b[1] == pytest.approx(np.log((6 / 4) / (3 / 7)), abs=1e-6)


def test_change_covariance_detects_synchronised_firm():
    """A firm whose cells all change together gives a positive firm-sharing covariance of the change."""
    df = pd.DataFrame(dict(firm_unit=["F"] * 4 + ["G"] * 4, drug=list("abcd") * 2, unavailable=[1, 1, 1, 1, 0, 0, 0, 0.0]))
    sc = de.sharing_covariances(df)
    assert sc["cov_shared_firm"] > 0 and sc["cov_shared_drug"] < 0
