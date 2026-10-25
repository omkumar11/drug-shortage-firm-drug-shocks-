"""Tests for analysis/fda_csv.py and analysis/long_panel.py (the archived FDA shortage CSV captures)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import fda_csv as fc  # noqa: E402
from analysis import long_panel as lp  # noqa: E402
from analysis import panel  # noqa: E402


def test_availability_coding():
    c = lambda t, h=False: fc.code_availability(t, harmonise=h)[0]
    for t in ("Available", "Inventory is currently available.", "Stock available. Additional resupply June 2020", "Product is Available",
              "Available and shipping", "Available in stock"):
        assert c(t) == 0, t
    for t in ("Unavailable", "Limited Availability", "Backordered. Next release October 2022.", "Next Delivery: May 2021; Estimated Recovery: TBD",
              "Available. On allocation.", "Limited Supply Available.", "Discontinued", "Supply available; intermittent backorder until June 2021",
              "Depleted", "Inventory depleted.", "Low inventory", "Available in February 2024", "Not currently marketing",
              "Available for primary customers", "Recovery TBD", "Inventory is low", "Back in stock week 1 of June 2020", "Supply Available TBD"):
        assert c(t) == 1, t
    for t in ("Undetermined", "", "Pending information", "TBD", "Launch in June 2020", "Not commercialized, N/A"):
        assert c(t) is None, t
    # allocation-only text counts as available in the harmonised coding; outages do not
    assert c("Available on allocation", True) == 0 and c("Low inventory", True) == 0 and c("Backordered. Next release October 2022.", True) == 1
    # the Unavailable-only outcome exists for the categorical values only
    assert fc.code_availability("Unavailable")[1] == 1 and fc.code_availability("Limited Availability")[1] == 0
    assert fc.code_availability("Backordered.")[1] is None
    # stale: every announced date more than a month before the capture
    assert fc.stale("Backordered. Next release October 2022.", pd.Timestamp("2023-03-01"))
    assert not fc.stale("Next delivery: March 2023", pd.Timestamp("2023-03-01"))


def toy_captures():
    def P(rows):
        return pd.DataFrame(rows, columns=["firm_unit", "drug", "package_ndc", "unavailable"])
    return {"20200101000000": P([("F", "Dextrose 50% Injection", "1-1-1", 1), ("F", "Saline Irrigation", "9-9-9", 0), ("G", "Water Irrigant", "8-8-8", 0),
                                 ("F", "Furosemide Injection, USP", "5-5-5", 1), ("G", "Furosemide Injection", "6-6-6", 0)]),
            "20210101000000": P([("F", "Dextrose Monohydrate 50% Injection", "1-1-1", 1), ("G", "Water Irrigant", "9-9-9", 0), ("G", "Saline Irrigation", "7-7-7", 0),
                                 ("G", "Furosemide Injection", "6-6-6", 0)])}


def test_canonical_drugs_links_renames_only():
    m, merges = fc.canonical_drugs(toy_captures())
    # a renaming that keeps the package NDC is linked, to the later name
    assert m["Dextrose 50% Injection"] == "Dextrose Monohydrate 50% Injection"
    # a style-only renaming is linked even when both names are listed together during the rollout
    assert m["Furosemide Injection, USP"] == "Furosemide Injection"
    # a shared NDC between names listed together in one capture, or without a common ingredient word, is not a renaming
    assert m["Saline Irrigation"] == "Saline Irrigation" and m["Water Irrigant"] == "Water Irrigant"


def three_wave_toy(seed=0):
    rng = np.random.default_rng(seed); W = {}
    firms = [f"F{i}" for i in range(8)]; drugs = [f"D{j}" for j in range(10)]
    base = [(f, d) for f in firms for d in drugs if rng.random() < 0.45]
    for k in ("w1", "w2", "w3"):
        W[k] = pd.DataFrame([(f, d, int(rng.integers(1, 4)), int(rng.random() < 0.5)) for f, d in base],
                            columns=["firm_unit", "drug", "n_presentations", "unavailable"])
    return W


def test_transition_statistic_matches_panel():
    """The generalised transition test computes the same statistic as panel.transition_test on three waves."""
    W = three_wave_toy()
    D = panel.interval_frame(W)
    a = panel.transition_test(W, np.random.default_rng(1), n_perm=50)
    b = lp.transition_test_frame(D.assign(transition=D["transition"]), np.random.default_rng(1), n_perm=50)
    assert b["cross_cov"] == pytest.approx(a["cross_cov"], abs=1e-12)


def test_first_listings_exclude_carryover():
    """A first listing is a cell never listed before on the grid and none of whose package NDCs was listed before."""
    W = three_wave_toy(3)
    for k in W:
        W[k] = W[k].assign(ndcs=[frozenset({f"{f}-{d}"}) for f, d in zip(W[k]["firm_unit"], W[k]["drug"])], injectable=1, categorical=1)
    grid = ["20200101000000", "20200701000000", "20210101000000", "20210701000000"]
    new = pd.DataFrame([("F0", "NEW1", 1, 1, frozenset({"n1"}), 1), ("F1", "NEW2", 2, 0, frozenset({"n2"}), 1), ("F2", "NEW3", 1, 1, frozenset({"F2-D0"}), 1)],
                       columns=["firm_unit", "drug", "n_presentations", "unavailable", "ndcs", "injectable"]).assign(categorical=1)
    WW = {grid[0]: W["w1"], grid[1]: W["w2"], grid[2]: W["w3"], grid[3]: pd.concat([W["w3"], new], ignore_index=True)}
    # captures: every grid date, with the package NDCs of its cells
    P = {g: pd.DataFrame(dict(package_ndc=[k for s_ in WW[g]["ndcs"] for k in s_])) for g in grid}
    P[grid[0]] = pd.concat([P[grid[0]], pd.DataFrame(dict(package_ndc=["F2-D0"]))], ignore_index=True)
    E = lp._first_listing_rows(WW, grid, P, 365, 300)
    # NEW3 carries an NDC listed earlier (under another drug), so it is not a first listing
    assert sorted(E["drug"]) == ["NEW1", "NEW2"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "raw", "archive_csv", "manifest.csv")), reason="captures not downloaded")
def test_csv_capture_agrees_with_openfda_export():
    """The capture nearest the analysis export date reproduces its cells almost exactly."""
    P = {ts: fc.read_capture(p, d).assign(date=d) for ts, d, p in fc.captures() if ts.startswith("202610")}
    W, date, _, _ = panel.wave_cells(panel.WAVES[-1][1])
    ts = min(P, key=lambda t: abs((P[t]["date"].iloc[0] - pd.Timestamp(date)).days))
    C = fc.cells_of(P[ts]); m = W.merge(C, on=["firm_unit", "drug"], suffixes=("_o", "_c"))
    assert len(m) / len(W) > 0.95 and (m["unavailable_o"] == m["unavailable_c"]).mean() > 0.95
