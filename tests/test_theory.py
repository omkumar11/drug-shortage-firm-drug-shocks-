"""Unit tests for the formal results of Section 2.5 (model/theory.py): every closed form is checked against brute-force
enumeration of all latent states on small explicit graphs, and every inequality on random parameter draws."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import theory as th  # noqa: E402
from model import twotier as tt  # noqa: E402


def rand_theta(rng):
    return dict(eps=rng.uniform(0, 0.4), a=rng.uniform(0.05, 1), b=rng.uniform(0.05, 0.95),
                phi=rng.uniform(0.05, 0.95), mu=rng.uniform(0.05, 0.95))


def test_lemma1_firm_row_moments_match_enumeration():
    # one firm (0) with j cells in drugs 0..j-1; only that firm's cells enter the event, other firms are integrated out
    rng = np.random.default_rng(1)
    for _ in range(5):
        t = rand_theta(rng)
        for j in (1, 2, 3):
            cells = [(0, m) for m in range(j)]
            assert th.brute_all_unavailable(cells, 3, 3, **t) == pytest.approx(th.firm_moment(j, **t), rel=1e-10)


def test_lemma1_and_theorem2_drug_row_moments_are_outage_probabilities():
    rng = np.random.default_rng(2)
    for _ in range(5):
        t = rand_theta(rng)
        for n in (1, 2, 3, 4):
            cells = [(f, 0) for f in range(n)]
            assert th.brute_all_unavailable(cells, 4, 2, **t) == pytest.approx(th.outage(n, **t), rel=1e-10)


def test_first_moments_agree_with_marginal_eq2():
    rng = np.random.default_rng(3)
    t = rand_theta(rng)
    assert th.firm_moment(1, **t) == pytest.approx(tt.marginal_prob(**t))
    assert th.drug_moment(1, **t) == pytest.approx(tt.marginal_prob(**t))


def test_second_moments_reproduce_covariances_eq3_eq4():
    rng = np.random.default_rng(4)
    t = rand_theta(rng)
    m1 = tt.marginal_prob(**t)
    assert th.firm_moment(2, **t) - m1 ** 2 == pytest.approx(tt.cov_shared_firm(**t), rel=1e-10)
    assert th.drug_moment(2, **t) - m1 ** 2 == pytest.approx(tt.cov_shared_market(**t), rel=1e-10)


def test_theorem1_identification_inverts_population_moments():
    rng = np.random.default_rng(5)
    for _ in range(200):
        t = rand_theta(rng)
        fm = [th.firm_moment(j, **t) for j in (1, 2, 3)]
        dm = [th.drug_moment(j, **t) for j in (1, 2, 3)]
        est = th.identify(fm, dm)
        assert est is not None
        for k in t:
            assert est[k] == pytest.approx(t[k], abs=1e-5), (k, t, est)   # loose: conditioning worsens as the Hankel determinant shrinks


def test_theorem1_degenerates_without_drug_shocks():
    # b = 0: drug rows are i.i.d. Bernoulli, the Hankel determinant is zero and mu is not identified
    t = dict(eps=0.1, a=0.6, b=0.0, phi=0.4, mu=0.3)
    dm = [th.drug_moment(j, **t) for j in (1, 2, 3)]
    assert dm[1] - dm[0] ** 2 == pytest.approx(0.0, abs=1e-15)
    assert th.prony(*dm) is None
    t2 = dict(t, mu=0.8)
    assert np.allclose(dm, [th.drug_moment(j, **t2) for j in (1, 2, 3)])


def test_star_graph_nonidentification_example():
    # every drug has one firm: the likelihood depends on (eps, mu, b) only through (1-eps)(1-mu b)
    t1 = dict(eps=0.10, a=0.6, b=0.5, phi=0.4, mu=0.2)
    keep = (1 - t1["eps"]) * (1 - t1["mu"] * t1["b"])
    t2 = dict(t1, mu=0.6, b=0.25); t2["eps"] = 1 - keep / (1 - t2["mu"] * t2["b"])
    cells = [(0, 0), (0, 1), (1, 2)]
    for sub in ([(0, 0)], [(0, 0), (0, 1)], cells):
        assert th.brute_all_unavailable(sub, 2, 3, **t1) == pytest.approx(th.brute_all_unavailable(sub, 2, 3, **t2))
    af1 = tt.attributable_fractions(**t1); af2 = tt.attributable_fractions(**t2)
    assert af1[0] == pytest.approx(af2[0])          # AF_firm identified
    assert abs(af1[1] - af2[1]) > 0.05               # AF_drug not


def test_proposition1_af_formulas_and_ordering():
    rng = np.random.default_rng(6)
    for _ in range(500):
        t = rand_theta(rng)
        p0, _ = th.firm_row_rates(**t); q0, _ = th.drug_row_rates(**t)
        m1 = tt.marginal_prob(**t)
        aff, afd = tt.attributable_fractions(**t)
        f2, d2 = th.af_from_rates(p0, q0, m1)
        assert f2 == pytest.approx(aff) and d2 == pytest.approx(afd)
        assert (aff > afd) == (t["phi"] * t["a"] > t["mu"] * t["b"])


def test_theorem2_properties():
    rng = np.random.default_rng(7)
    for _ in range(300):
        t = rand_theta(rng)
        O = np.array([th.outage(n, **t) for n in range(1, 25)])
        q0, q1 = th.drug_row_rates(**t)
        assert np.all(np.diff(O) <= 1e-15)                                         # decreasing
        assert np.all(O[1:-1] ** 2 <= O[:-2] * O[2:] * (1 + 1e-12))                 # (a) log-convex
        ind = np.array([th.outage_independent(n, **t) for n in range(1, 25)])
        assert np.all(O >= ind * (1 - 1e-12))                                       # (b) >= independence benchmark
        assert np.all(np.diff(O / ind) >= -1e-9)                                     # (c) O_n / M_1^n non-decreasing
        ratio = O[1:] / O[:-1]
        assert np.all(np.diff(ratio) >= -1e-12) and np.all(ratio <= q1 + 1e-12)    # (c) ratio increases to q_1
        assert ratio[-1] == pytest.approx(q1, rel=0.05) or q1 - q0 < 0.05
        for alpha in (0.05, 0.01):
            if alpha < t["mu"]:
                lo, hi = th.n_star_bounds(alpha, **t)
                ns = th.n_star(alpha, **t)
                assert lo <= ns <= hi                                               # (d)


def test_theorem2_recurrence():
    # O_{n+2} = s O_{n+1} - t O_n with s = q_0 + q_1, t = q_0 q_1: the first three moments determine all others
    t = dict(eps=0.08, a=0.5, b=0.4, phi=0.6, mu=0.3)
    q0, q1 = th.drug_row_rates(**t)
    O = [1.0] + [th.outage(n, **t) for n in range(1, 8)]
    for n in range(0, 6):
        assert O[n + 2] == pytest.approx((q0 + q1) * O[n + 1] - q0 * q1 * O[n], rel=1e-10)


def test_proposition2_joint_outage_formula_and_monotonicity():
    rng = np.random.default_rng(8)
    for _ in range(3):
        t = rand_theta(rng)
        # drugs 0 and 1 with 3 listed firms each, k shared; firms indexed 0..(6-k-1)
        for k in range(0, 4):
            s0 = list(range(3)); s1 = list(range(3 - k, 6 - k))
            cells = [(f, 0) for f in s0] + [(f, 1) for f in s1]
            F = 6 - k
            assert th.brute_all_unavailable(cells, F, 2, **t) == pytest.approx(th.joint_outage(3, 3, k, **t), rel=1e-10)
    for _ in range(300):
        t = rand_theta(rng)
        j = [th.joint_outage(4, 3, k, **t) for k in range(0, 4)]
        assert np.all(np.diff(j) > 0)                                            # strictly increasing on Theta
        assert j[0] == pytest.approx(th.outage(4, **t) * th.outage(3, **t))     # k = 0: independent drugs


def test_mom_estimator_unbiased_moments_and_recovery():
    rng = np.random.default_rng(9)
    F, M = 400, 400
    # each firm 5 drugs, each drug 5 firms: a 5-regular bipartite graph from shifted permutations
    fi = np.repeat(np.arange(F), 5); mi = np.concatenate([(np.arange(5) * 37 + f) % M for f in range(F)])
    assert len(set(zip(fi, mi))) == len(fi)
    truth = dict(eps=0.1, a=0.6, b=0.5, phi=0.4, mu=0.3)
    est = []
    for r in range(30):
        y, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **truth)
        e = th.mom_estimate(y, fi, mi)
        assert e.get("params") is not None
        est.append([e["params"][k] for k in ("eps", "a", "b", "phi", "mu")])
    est = np.median(np.array(est), axis=0)
    assert np.allclose(est, [truth[k] for k in ("eps", "a", "b", "phi", "mu")], atol=0.08), est


def test_proposition1_log_availability_shares_add_up_and_order():
    rng = np.random.default_rng(10)
    for _ in range(200):
        t = rand_theta(rng)
        tot = -np.log(1 - tt.marginal_prob(**t))
        parts = [-np.log(1 - t["eps"]), -np.log(1 - t["phi"] * t["a"]), -np.log(1 - t["mu"] * t["b"])]
        assert sum(parts) == pytest.approx(tot)
        aff, afd = tt.attributable_fractions(**t)
        assert (parts[1] > parts[2]) == (aff > afd)


def test_unbiased_row_moment_estimator():
    # E[C(U,j)/C(k,j)] = M_j for a row of k conditionally i.i.d. cells (exact, by enumeration of U)
    from math import comb
    t = dict(eps=0.1, a=0.6, b=0.5, phi=0.4, mu=0.3)
    p0, p1 = th.firm_row_rates(**t)
    for k in (3, 5):
        for j in (1, 2, 3):
            e = sum((t["phi"] * comb(k, u) * p1 ** u * (1 - p1) ** (k - u) + (1 - t["phi"]) * comb(k, u) * p0 ** u * (1 - p0) ** (k - u))
                    * comb(u, j) / comb(k, j) for u in range(k + 1))
            assert e == pytest.approx(th.firm_moment(j, **t), rel=1e-12)


def test_corollary1_share_variance_matches_enumeration():
    rng = np.random.default_rng(31)
    for _ in range(5):
        t = rand_theta(rng)
        for k in (1, 2, 3, 4):
            e1, e2 = th.brute_share_moments(k, **t)
            assert e1 == pytest.approx(th.firm_moment(1, **t), rel=1e-10)
            assert e2 - e1 ** 2 == pytest.approx(th.share_variance(k, "firm", **t), rel=1e-9)
            m1 = e1
            assert (e2 - e1 ** 2) / (m1 * (1 - m1) / k) == pytest.approx(th.dispersion_ratio_expected(k, "firm", **t), rel=1e-9)


def test_corollary1_hankel_equals_sharing_covariance_and_order_criterion():
    rng = np.random.default_rng(32)
    for _ in range(500):
        t = rand_theta(rng)
        assert th.hankel("firm", **t) == pytest.approx(tt.cov_shared_firm(**t), rel=1e-10)
        assert th.hankel("drug", **t) == pytest.approx(tt.cov_shared_market(**t), rel=1e-10)
        assert th.covariance_order_firm(**t) == (th.hankel("firm", **t) > th.hankel("drug", **t))


def test_corollary1_covariance_order_differs_from_af_order():
    # phi a > mu b but D_F < D_M (firm shocks prevalent), and the mirror case
    t1 = dict(eps=0.1, a=0.5, b=0.6, phi=0.99, mu=0.5)
    assert t1["phi"] * t1["a"] > t1["mu"] * t1["b"] and not th.covariance_order_firm(**t1)
    t2 = dict(eps=0.1, a=0.6, b=0.5, phi=0.5, mu=0.99)
    assert t2["phi"] * t2["a"] < t2["mu"] * t2["b"] and th.covariance_order_firm(**t2)


def test_corollary1d_conditional_co_unavailability_matches_enumeration():
    rng = np.random.default_rng(33)
    for _ in range(5):
        t = rand_theta(rng)
        both = th.brute_all_unavailable([(0, 0), (0, 1)], 2, 2, **t); one = th.brute_all_unavailable([(0, 0)], 2, 2, **t)
        assert both / one == pytest.approx(th.co_unavailability("firm", **t), rel=1e-10)
        bothd = th.brute_all_unavailable([(0, 0), (1, 0)], 2, 2, **t)
        assert bothd / one == pytest.approx(th.co_unavailability("drug", **t), rel=1e-10)
        m1 = th.firm_moment(1, **t)
        assert th.co_unavailability("firm", **t) == pytest.approx(m1 + th.hankel("firm", **t) / m1, rel=1e-10)
        unrel = th.brute_all_unavailable([(0, 0), (1, 1)], 2, 2, **t)
        assert unrel / one == pytest.approx(m1, rel=1e-10)


def test_corollary2_portfolio_outage_variance_matches_enumeration():
    # three drugs on four firms: drug 0 {0,1}, drug 1 {1,2}, drug 2 {3}; drugs 0 and 1 share firm 1
    import itertools
    sets = [[0, 1], [1, 2], [3]]; F, M = 4, 3
    rng = np.random.default_rng(34)
    for _ in range(4):
        t = rand_theta(rng)
        e1 = e2 = 0.0
        for zc in itertools.product([0, 1], repeat=F):
            for wc in itertools.product([0, 1], repeat=M):
                pl = np.prod([t["phi"] if z else 1 - t["phi"] for z in zc]) * np.prod([t["mu"] if w else 1 - t["mu"] for w in wc])
                po = [np.prod([th.pi(t["eps"], t["a"], t["b"], zc[f], wc[m]) for f in fs]) for m, fs in enumerate(sets)]
                # N is a sum of conditionally independent Bernoulli(po[m]) given the latents
                en = sum(po); en2 = sum(po) + sum(po[i] * po[j] for i in range(M) for j in range(M) if i != j)
                e1 += pl * en; e2 += pl * en2
        n = [2, 2, 1]; K = np.zeros((3, 3), int); K[0, 1] = K[1, 0] = 1
        mean, var, var0 = th.portfolio_outage_moments(n, K, **t)
        assert mean == pytest.approx(e1, rel=1e-10)
        assert var == pytest.approx(e2 - e1 ** 2, rel=1e-9)
        assert var > var0


def test_proposition3_lagged_covariances_enumeration():
    """Proposition 3 against exact enumeration of latent paths and outcomes at two times."""
    rng = np.random.default_rng(31)
    for _ in range(25):
        eps, a, b, phi, mu = rng.uniform(0.02, 0.6), rng.uniform(0.1, 1), rng.uniform(0.1, 1), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        # admissible correlations of a stationary binary pair: p11 = p^2 + r p (1-p) must lie in [max(0, 2p-1), p]
        lo = lambda p: max(-p / (1 - p), -(1 - p) / p)
        r_F, r_M = rng.uniform(lo(phi), 1), rng.uniform(lo(mu), 1)
        ex = th.lagged_covariances(eps, a, b, phi, mu, r_F, r_M)
        br = th.brute_lagged_covariances(eps, a, b, phi, mu, r_F, r_M)
        for k in ("cov_firm_lag", "cov_drug_lag", "cov_firm_change", "cov_drug_change"):
            assert br[k] == pytest.approx(ex[k], abs=1e-12)
        assert abs(br["cov_none_lag"]) < 1e-12
        # at r = 1 the lagged covariance is the same-time one (Corollary 1(a)) and changes do not co-vary
        th1 = th.lagged_covariances(eps, a, b, phi, mu, 1.0, 1.0)
        assert th1["cov_firm_lag"] == pytest.approx(th.hankel("firm", eps, a, b, phi, mu)) and th1["cov_firm_change"] == 0


def test_proposition3_with_cell_carryover_enumeration():
    """The carry-over extension of Proposition 3 against exact enumeration; at kappa = 0 it is Proposition 3."""
    rng = np.random.default_rng(37)
    for _ in range(25):
        eps, a, b, phi, mu = rng.uniform(0.02, 0.6), rng.uniform(0.1, 1), rng.uniform(0.1, 1), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        lo = lambda p: max(-p / (1 - p), -(1 - p) / p)
        r_F, r_M, kappa = rng.uniform(lo(phi), 1), rng.uniform(lo(mu), 1), rng.uniform(0, 1)
        ex = th.carryover_covariances(eps, a, b, phi, mu, r_F, kappa)
        br = th.brute_carryover_covariances(eps, a, b, phi, mu, r_F, r_M, kappa)
        assert br["cov_firm_lag"] == pytest.approx(ex["cov_firm_lag"], abs=1e-12)
        assert br["cov_firm_same_t"] == pytest.approx(ex["cov_firm_same_t"], abs=1e-12)
        assert br["cov_firm_change"] == pytest.approx(ex["cov_firm_change"], abs=1e-12)
        k0 = th.carryover_covariances(eps, a, b, phi, mu, r_F, 0.0)
        assert k0["cov_firm_lag"] == pytest.approx(th.lagged_covariances(eps, a, b, phi, mu, r_F, r_M)["cov_firm_lag"])
        # the ratio is at least r_F for r_F in [0, 1] and equals one at r_F = 1
        if r_F >= 0:
            assert ex["ratio"] >= r_F - 1e-12
    assert th.carryover_covariances(0.1, 0.5, 0.5, 0.5, 0.5, 1.0, 0.4)["ratio"] == pytest.approx(1.0)
