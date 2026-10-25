"""Unit tests: the implementation must agree with the equations written in the manuscript, and satisfy limiting cases."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import twotier as tt  # noqa: E402


def test_eq1_cell_prob_matches_formula():
    eps, a, b = 0.1, 0.6, 0.3
    assert tt.cell_prob(eps, a, b, 0, 0) == pytest.approx(eps)
    assert tt.cell_prob(eps, a, b, 1, 0) == pytest.approx(1 - 0.9 * 0.4)
    assert tt.cell_prob(eps, a, b, 0, 1) == pytest.approx(1 - 0.9 * 0.7)
    assert tt.cell_prob(eps, a, b, 1, 1) == pytest.approx(1 - 0.9 * 0.4 * 0.7)


def test_eq2_marginal_is_expectation_of_eq1():
    eps, a, b, phi, mu = 0.05, 0.7, 0.4, 0.3, 0.2
    exp = sum(phi ** z * (1 - phi) ** (1 - z) * mu ** w * (1 - mu) ** (1 - w) * tt.cell_prob(eps, a, b, z, w)
              for z in (0, 1) for w in (0, 1))
    assert tt.marginal_prob(eps, a, b, phi, mu) == pytest.approx(exp)


def test_eq3_eq4_covariances_by_monte_carlo():
    rng = np.random.default_rng(1)
    eps, a, b, phi, mu = 0.05, 0.6, 0.5, 0.3, 0.25
    n = 400000
    z = rng.random(n) < phi; w1 = rng.random(n) < mu; w2 = rng.random(n) < mu
    y1 = rng.random(n) < tt.cell_prob(eps, a, b, z, w1); y2 = rng.random(n) < tt.cell_prob(eps, a, b, z, w2)
    cov_f = np.cov(y1, y2)[0, 1]
    assert cov_f == pytest.approx(tt.cov_shared_firm(eps, a, b, phi, mu), abs=0.004)
    z1 = rng.random(n) < phi; z2 = rng.random(n) < phi; w = rng.random(n) < mu
    y1 = rng.random(n) < tt.cell_prob(eps, a, b, z1, w); y2 = rng.random(n) < tt.cell_prob(eps, a, b, z2, w)
    cov_m = np.cov(y1, y2)[0, 1]
    assert cov_m == pytest.approx(tt.cov_shared_market(eps, a, b, phi, mu), abs=0.004)


def test_limiting_cases():
    # no transmission -> marginal equals baseline, covariances vanish
    assert tt.marginal_prob(0.2, 0.0, 0.0, 0.5, 0.5) == pytest.approx(0.2)
    assert tt.cov_shared_firm(0.2, 0.0, 0.5, 0.5, 0.5) == 0.0
    assert tt.cov_shared_market(0.2, 0.5, 0.0, 0.5, 0.5) == 0.0
    # certain shocks (phi = 1) -> no firm-level variance
    assert tt.cov_shared_firm(0.1, 0.5, 0.5, 1.0, 0.3) == 0.0
    # eps = 1 -> everything unavailable
    assert tt.marginal_prob(1.0, 0.5, 0.5, 0.3, 0.3) == pytest.approx(1.0)
    af_f, af_m = tt.attributable_fractions(0.1, 0.0, 0.5, 0.3, 0.3)
    assert af_f == pytest.approx(0.0)
    assert 0 < af_m < 1


def test_gibbs_latents_match_exact_posterior_on_tiny_graph():
    # 3 firms x 3 markets, 7 cells; theta fixed so only the latent posterior is compared
    fi = np.array([0, 0, 1, 1, 2, 2, 0]); mi = np.array([0, 1, 1, 2, 0, 2, 2]); y = np.array([1, 1, 0, 1, 0, 0, 1])
    th = dict(eps=0.1, a=0.6, b=0.5, phi=0.3, mu=0.3)
    pz, pw, _ = tt.exact_posterior(y, fi, mi, 3, 3, **th)
    out = tt.gibbs(y, fi, mi, 3, 3, n_iter=40000, burn=2000, thin=1, seed=3, fix_theta=th)
    assert np.allclose(out["z_post"], pz, atol=0.02)
    assert np.allclose(out["w_post"], pw, atol=0.02)


def test_simulation_recovers_parameters_on_moderate_graph():
    rng = np.random.default_rng(7)
    F, M = 120, 80
    fi = rng.integers(0, F, 900); mi = rng.integers(0, M, 900)
    truth = dict(eps=0.05, a=0.7, b=0.5, phi=0.3, mu=0.3)
    y, z, w = tt.simulate(fi, mi, F, M, rng=rng, **truth)
    out = tt.gibbs(y, fi, mi, F, M, n_iter=6000, burn=2000, thin=2, seed=11)
    for k, v in truth.items():
        lo, hi = np.percentile(out[k], [2.5, 97.5])
        assert lo - 0.05 <= v <= hi + 0.05, (k, v, lo, hi)


def test_rhat_near_one_for_identical_chains():
    rng = np.random.default_rng(0)
    c = [rng.standard_normal(2000) for _ in range(4)]
    assert abs(tt.rhat(c) - 1.0) < 0.05


def test_eq6_attribution_rows_sum_to_one_for_unavailable_cells():
    fi = np.array([0, 0, 1, 1]); mi = np.array([0, 1, 0, 1]); y = np.array([1, 1, 0, 1])
    z = np.array([1, 0]); w = np.array([0, 1])
    at = tt.cause_attribution(y, fi, mi, z, w, 0.1, 0.6, 0.5)
    assert np.allclose(at[y == 1].sum(axis=1), 1.0)
    assert np.allclose(at[y == 0], 0.0)
    # cell (0,0): firm shocked, market not -> no market attribution
    assert at[0, 1] == 0 and at[0, 2] == 0 and at[0, 0] == pytest.approx(0.6 / (1 - 0.9 * 0.4))


def test_per_presentation_baseline_reduces_to_eq1_when_n_is_one():
    eps, a, b = 0.1, 0.6, 0.3
    z = np.array([0, 1, 0, 1]); w = np.array([0, 0, 1, 1])
    assert np.allclose(tt.cell_prob(eps, a, b, z, w, n=np.ones(4)), tt.cell_prob(eps, a, b, z, w))
    assert tt.cell_prob(eps, a, b, 0, 0, n=3) == pytest.approx(1 - 0.9 ** 3)
    assert tt.cell_prob(eps, a, b, 1, 1, n=2) == pytest.approx(1 - 0.9 ** 2 * 0.4 * 0.7)


def test_per_presentation_gibbs_matches_exact_latent_posterior_on_small_graph():
    """With theta fixed, the block updates under the per-presentation baseline target the exact posterior of (Z, W)
    computed by enumeration with the n-adjusted cell probabilities."""
    rng = np.random.default_rng(5)
    fi = np.array([0, 0, 1, 1, 2, 2]); mi = np.array([0, 1, 0, 1, 0, 1]); F, M = 3, 2
    n = np.array([1, 3, 2, 1, 4, 2], dtype=float); th = dict(eps=0.08, a=0.6, b=0.4, phi=0.4, mu=0.5)
    z0 = np.array([1, 0, 1]); w0 = np.array([1, 0])
    y = (rng.random(6) < tt.cell_prob(th["eps"], th["a"], th["b"], z0[fi], w0[mi], n)).astype(float)
    # exact posterior mean of each latent by enumeration
    import itertools
    num = np.zeros(F + M); den = 0.0
    for zz in itertools.product([0, 1], repeat=F):
        for ww in itertools.product([0, 1], repeat=M):
            zz_, ww_ = np.array(zz), np.array(ww)
            p = tt.cell_prob(th["eps"], th["a"], th["b"], zz_[fi], ww_[mi], n)
            lik = np.prod(np.where(y == 1, p, 1 - p)) * np.prod(th["phi"] ** zz_ * (1 - th["phi"]) ** (1 - zz_)) * np.prod(th["mu"] ** ww_ * (1 - th["mu"]) ** (1 - ww_))
            num += lik * np.concatenate([zz_, ww_]); den += lik
    exact = num / den
    d = tt.gibbs(y, fi, mi, F, M, n_iter=30000, burn=2000, thin=1, seed=1, fix_theta=th, n_pres=n)
    assert np.allclose(np.concatenate([d["z_post"], d["w_post"]]), exact, atol=0.02)


def test_collapsed_marginal_likelihood_matches_enumeration_over_w():
    """analysis/sbc_collapsed.py: the per-drug log-likelihood with W summed out equals the enumeration over W."""
    import itertools
    from analysis.sbc_collapsed import drug_loglik
    rng = np.random.default_rng(3); F, M = 3, 3
    fi = np.array([0, 0, 1, 1, 2, 2, 0]); mi = np.array([0, 1, 1, 2, 0, 2, 2])
    th = dict(eps=0.2, a=0.6, b=0.5, phi=0.4, mu=0.3)
    y, z, w = tt.simulate(fi, mi, F, M, rng=rng, **th)
    drug_cells = [np.flatnonzero(mi == m) for m in range(M)]
    got = drug_loglik(y, z[fi], th["eps"], th["a"], th["b"], th["mu"], drug_cells).sum()
    tot = 0.0
    for wv in itertools.product([0, 1], repeat=M):
        wv = np.array(wv); p = tt.cell_prob(th["eps"], th["a"], th["b"], z[fi], wv[mi])
        tot += np.prod(np.where(y == 1, p, 1 - p)) * np.prod(np.where(wv == 1, th["mu"], 1 - th["mu"]))
    assert np.isclose(got, np.log(tot))


def test_collapsed_gibbs_recovers_exact_z_posterior():
    """With theta fixed, the collapsed sampler's P(Z_f = 1 | y) matches brute-force enumeration over (Z, W)."""
    import itertools
    from analysis.sbc_collapsed import collapsed_gibbs
    rng = np.random.default_rng(5); F, M = 3, 3
    fi = np.array([0, 0, 1, 1, 2, 2, 0]); mi = np.array([0, 1, 1, 2, 0, 2, 2])
    th = dict(eps=0.15, a=0.7, b=0.5, phi=0.5, mu=0.4)
    y, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **th)
    num = np.zeros(F); den = 0.0
    for zv in itertools.product([0, 1], repeat=F):
        for wv in itertools.product([0, 1], repeat=M):
            zv_, wv_ = np.array(zv), np.array(wv); p = tt.cell_prob(th["eps"], th["a"], th["b"], zv_[fi], wv_[mi])
            pr = np.prod(np.where(y == 1, p, 1 - p)) * np.prod(np.where(zv_ == 1, th["phi"], 1 - th["phi"])) * np.prod(np.where(wv_ == 1, th["mu"], 1 - th["mu"]))
            num += pr * zv_; den += pr
    d = collapsed_gibbs(y, fi, mi, F, M, n_iter=6000, burn=1000, thin=1, seed=1, init=th, fix_theta=th)
    assert np.allclose(d["z_sum"].mean(), (num / den).sum(), atol=0.08)
