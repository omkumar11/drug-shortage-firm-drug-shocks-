"""Unit tests for the simulation-based calibration helpers (analysis/sbc.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import sbc  # noqa: E402
from model import twotier as tt  # noqa: E402


def test_rank_counts_draws_below_truth():
    assert sbc.rank_of(0.5, [0.1, 0.4, 0.6, 0.9]) == 2
    assert sbc.rank_of(0.0, [0.1, 0.4]) == 0 and sbc.rank_of(1.0, [0.1, 0.4]) == 2


def test_rank_is_uniform_when_draws_come_from_the_same_distribution():
    # if truth and draws are i.i.d. from one distribution, the rank is uniform on {0, ..., L}
    rng = np.random.default_rng(0); L = 20
    ranks = [sbc.rank_of(rng.random(), rng.random(L)) for _ in range(4000)]
    t = sbc.uniformity_test(ranks, L)
    assert t["p"] > 0.01 and abs(t["mean_rank"] - L / 2) < 0.5


def test_uniformity_test_rejects_a_skewed_rank_distribution():
    rng = np.random.default_rng(1); L = 20
    ranks = np.minimum(L, rng.integers(0, 8, 400))      # ranks piled at the bottom
    assert sbc.uniformity_test(ranks, L)["p"] < 1e-6


def test_derived_quantities_match_model_functions():
    th = dict(eps=0.1, a=0.6, b=0.3, phi=0.4, mu=0.2)
    d = sbc.derived(th); aff, afd = tt.attributable_fractions(**th)
    assert d["af_firm"] == pytest.approx(aff) and d["af_drug"] == pytest.approx(afd)
    assert d["phi_a"] == pytest.approx(0.24) and d["mu_b"] == pytest.approx(0.06)


def test_chain_summary_ranks_and_coverage_agree_with_rank_of():
    rng = np.random.default_rng(3)
    d = {k: rng.random(200) for k in tt.PARAMS}
    truth = {k: float(rng.random()) for k in tt.PARAMS}
    out = sbc.chain_summary(d, truth)
    post = sbc.derived(d); tv = sbc.derived(truth)
    for k in sbc.QUANTITIES:
        assert out["ranks"][k] == sbc.rank_of(tv[k], post[k])
        lo, hi = np.percentile(post[k], [5, 95])
        assert out["cover"][k] == int(lo <= tv[k] <= hi)
    assert out["L"] == 200 and 0 < out["ess_mu_b"] <= 200


def test_replicates_are_reproducible_from_the_seed():
    F, M = 4, 3
    fi = np.array([0, 0, 1, 1, 2, 2, 3, 3]); mi = np.array([0, 1, 1, 2, 0, 2, 1, 0])
    a = sbc.draw_replicates(fi, mi, F, M, n_sbc=3, seed=5); b = sbc.draw_replicates(fi, mi, F, M, n_sbc=3, seed=5)
    assert all(x["truth"] == y["truth"] and np.array_equal(x["y"], y["y"]) and x["init"] == y["init"] for x, y in zip(a, b))
