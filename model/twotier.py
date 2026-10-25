"""
twotier.py -- the two-tier latent-shock model used in the manuscript (Section 2.4), implemented exactly as written.

Notation (manuscript eqs. (1)-(4)):
  firms f = 1..F  (tier 1), drug markets m = 1..M (tier 2), observed cells e = (f, m) in E, outcome y_e in {0,1}
  (1 = the firm's presentations of the drug are unavailable or on limited availability).
  Latent firm shock      Z_f ~ Bernoulli(phi)       independent over f
  Latent market shock    W_m ~ Bernoulli(mu)        independent over m, independent of Z
  Eq. (1)  P(y_e = 1 | Z_f, W_m) = 1 - (1 - eps) (1 - a)^{Z_f} (1 - b)^{W_m}          (noisy-OR)
  Eq. (2)  P(y_e = 1) = 1 - (1 - eps)(1 - phi a)(1 - mu b)                               (marginal)
  Eq. (3)  Cov(y_e, y_e') for e, e' sharing a firm (different markets)
                        = (1 - eps)^2 (1 - mu b)^2 phi (1 - phi) a^2
  Eq. (4)  Cov(y_e, y_e') for e, e' sharing a market (different firms)
                        = (1 - eps)^2 (1 - phi a)^2 mu (1 - mu) b^2
  Attributable fractions (eq. (5)): AF_firm = 1 - P(y=1 | a=0)/P(y=1), AF_market = 1 - P(y=1 | b=0)/P(y=1).

Estimation: Bayesian, blocked Gibbs sampling. Given W, the Z_f are conditionally independent (and vice versa),
so each block is updated in one vectorised step; (eps, a, b) are updated by a random-walk Metropolis step on the
logit scale; phi and mu have Beta conjugate updates. Priors: Beta(1,1) on every parameter.
A brute-force exact posterior for tiny graphs (used by the unit tests) is in `exact_posterior`.
"""
import itertools

import numpy as np

PARAMS = ("eps", "a", "b", "phi", "mu")


def cell_prob(eps, a, b, z, w, n=None):
    """Eq. (1). z, w are 0/1 arrays aligned with cells. n (optional): presentations per cell for the per-presentation
    baseline sensitivity, in which the baseline cause fires independently for each presentation, 1 - (1 - eps)^n."""
    base = (1.0 - eps) if n is None else (1.0 - eps) ** np.asarray(n)
    return 1.0 - base * (1.0 - a) ** z * (1.0 - b) ** w


def marginal_prob(eps, a, b, phi, mu):
    """Eq. (2)."""
    return 1.0 - (1.0 - eps) * (1.0 - phi * a) * (1.0 - mu * b)


def cov_shared_firm(eps, a, b, phi, mu):
    """Eq. (3)."""
    return (1.0 - eps) ** 2 * (1.0 - mu * b) ** 2 * phi * (1.0 - phi) * a ** 2


def cov_shared_market(eps, a, b, phi, mu):
    """Eq. (4)."""
    return (1.0 - eps) ** 2 * (1.0 - phi * a) ** 2 * mu * (1.0 - mu) * b ** 2


def attributable_fractions(eps, a, b, phi, mu):
    """Eq. (5): share of the marginal unavailability rate removed by switching off firm (market) shocks."""
    p = marginal_prob(eps, a, b, phi, mu)
    p_no_firm = marginal_prob(eps, 0.0, b, phi, mu)
    p_no_market = marginal_prob(eps, a, 0.0, phi, mu)
    return 1.0 - p_no_firm / p, 1.0 - p_no_market / p


def loglik_given_latents(y, fi, mi, z, w, eps, a, b, n=None):
    """log P(y | Z, W, theta) summed over cells. fi, mi: firm / market index per cell."""
    p = cell_prob(eps, a, b, z[fi], w[mi], n)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))


def simulate(fi, mi, F, M, eps, a, b, phi, mu, rng):
    """Draw latents and outcomes on a given bipartite graph."""
    z = (rng.random(F) < phi).astype(int)
    w = (rng.random(M) < mu).astype(int)
    y = (rng.random(len(fi)) < cell_prob(eps, a, b, z[fi], w[mi])).astype(int)
    return y, z, w


def _block_update(y, idx_self, idx_other, state_other, prior_p, eps, trans_self, trans_other, n_self, rng, n=None):
    """Vectorised conditional update of one block of latents (firms given markets, or markets given firms).
    trans_self is the transmission probability of this tier (a for firms, b for markets)."""
    other_factor = (1.0 - trans_other) ** state_other[idx_other]
    base = (1.0 - eps) if n is None else (1.0 - eps) ** np.asarray(n)
    p1 = 1.0 - base * (1.0 - trans_self) * other_factor      # cell prob if this latent = 1
    p0 = 1.0 - base * other_factor                             # cell prob if this latent = 0
    p1 = np.clip(p1, 1e-12, 1 - 1e-12); p0 = np.clip(p0, 1e-12, 1 - 1e-12)
    ll1 = y * np.log(p1) + (1 - y) * np.log1p(-p1)
    ll0 = y * np.log(p0) + (1 - y) * np.log1p(-p0)
    s1 = np.bincount(idx_self, weights=ll1, minlength=n_self) + np.log(prior_p)
    s0 = np.bincount(idx_self, weights=ll0, minlength=n_self) + np.log1p(-prior_p)
    post = 1.0 / (1.0 + np.exp(s0 - s1))
    return (rng.random(n_self) < post).astype(int), post


def gibbs_step(y, fi, mi, F, M, th, z, w, rng, step=0.35, fix_theta=None, prior=(1.0, 1.0), n_pres=None):
    """One sweep of the blocked Gibbs / Metropolis kernel: Z | W, theta; W | Z, theta; phi, mu (Beta conjugate);
    (eps, a, b) by a random-walk Metropolis step on the logit scale. Mutates th in place; returns (z, w, accepted)."""
    alpha0, beta0 = prior
    z, _ = _block_update(y, fi, mi, w, th["phi"], th["eps"], th["a"], th["b"], F, rng, n_pres)
    w, _ = _block_update(y, mi, fi, z, th["mu"], th["eps"], th["b"], th["a"], M, rng, n_pres)
    if not (fix_theta and "phi" in fix_theta):
        th["phi"] = rng.beta(alpha0 + z.sum(), beta0 + F - z.sum())
    if not (fix_theta and "mu" in fix_theta):
        th["mu"] = rng.beta(alpha0 + w.sum(), beta0 + M - w.sum())
    free = [k for k in ("eps", "a", "b") if not (fix_theta and k in fix_theta)]
    accepted = None
    if free:
        cur = np.array([th[k] for k in free]); lcur = np.log(cur / (1 - cur))
        lprop = lcur + step * rng.standard_normal(len(free)); prop = 1 / (1 + np.exp(-lprop))
        thp = dict(th); thp.update(dict(zip(free, prop)))
        ll_cur = loglik_given_latents(y, fi, mi, z, w, th["eps"], th["a"], th["b"], n_pres)
        ll_prop = loglik_given_latents(y, fi, mi, z, w, thp["eps"], thp["a"], thp["b"], n_pres)
        # Beta(alpha0,beta0) prior on the probability scale + logit Jacobian
        lp_cur = np.sum((alpha0 - 1) * np.log(cur) + (beta0 - 1) * np.log1p(-cur) + np.log(cur) + np.log1p(-cur))
        lp_prop = np.sum((alpha0 - 1) * np.log(prop) + (beta0 - 1) * np.log1p(-prop) + np.log(prop) + np.log1p(-prop))
        accepted = bool(np.log(rng.random()) < (ll_prop + lp_prop) - (ll_cur + lp_cur))
        if accepted:
            th.update(dict(zip(free, prop)))
    return z, w, accepted


def gibbs(y, fi, mi, F, M, n_iter=20000, burn=5000, thin=5, seed=0, step=0.35, init=None,
          fix_theta=None, prior=(1.0, 1.0), init_latent=None, n_pres=None):
    """Blocked Gibbs / Metropolis sampler (repeated gibbs_step). Returns dict of posterior draws and latent posterior means.
    fix_theta: optional dict of parameters held fixed (used in tests against the exact posterior).
    init_latent: optional (z, w) starting states of the latents; by default they are drawn from the starting phi and mu.
    n_pres: optional presentations per cell; if given, the baseline is 1 - (1 - eps)^n per cell (per-presentation baseline)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float); fi = np.asarray(fi); mi = np.asarray(mi)
    th = dict(eps=0.1, a=0.5, b=0.5, phi=0.3, mu=0.3)
    if init: th.update(init)
    if fix_theta: th.update(fix_theta)
    z = (rng.random(F) < th["phi"]).astype(int); w = (rng.random(M) < th["mu"]).astype(int)
    if init_latent is not None:
        z = np.asarray(init_latent[0], dtype=int).copy(); w = np.asarray(init_latent[1], dtype=int).copy()
    draws = {k: [] for k in PARAMS}; zsum = np.zeros(F); wsum = np.zeros(M); n_keep = 0
    n_cells = len(y); attr = np.zeros((n_cells, 4))   # posterior mean of (firm-only, market-only, both, baseline-only) attribution
    acc = 0; tries = 0
    for it in range(n_iter):
        z, w, accepted = gibbs_step(y, fi, mi, F, M, th, z, w, rng, step=step, fix_theta=fix_theta, prior=prior, n_pres=n_pres)
        if accepted is not None:
            tries += 1; acc += int(accepted)
        if it >= burn and (it - burn) % thin == 0:
            for k in PARAMS: draws[k].append(th[k])
            zsum += z; wsum += w; n_keep += 1
            attr += cause_attribution(y, fi, mi, z, w, th["eps"], th["a"], th["b"])
    out = {k: np.array(v) for k, v in draws.items()}
    out["z_post"] = zsum / n_keep; out["w_post"] = wsum / n_keep
    out["accept_rate"] = acc / max(tries, 1)
    out["attribution"] = attr / n_keep
    return out


def cause_attribution(y, fi, mi, z, w, eps, a, b):
    """Eq. (6): for each cell, the conditional probability (given y_e = 1 and the latents) that the unavailability
    involves the firm cause (and not the market cause), the market cause (and not the firm cause), both, or only the
    baseline cause. The four terms sum to one for every cell with y_e = 1.
    In the noisy-OR each active cause 'fires' independently: firm with prob a*Z_f, market with prob b*W_m,
    baseline with prob eps; y_e = 1 iff at least one fires. Rows for y_e = 0 are zero."""
    pf = a * z[fi]; pm = b * w[mi]
    p = 1.0 - (1.0 - eps) * (1.0 - pf) * (1.0 - pm)
    p = np.clip(p, 1e-12, 1.0)
    firm_only = pf * (1 - pm) / p           # firm cause fires, market cause does not (baseline either way)
    market_only = pm * (1 - pf) / p         # market cause fires, firm cause does not
    both = pf * pm / p                      # firm and market both fire
    base_only = eps * (1 - pf) * (1 - pm) / p   # only the baseline cause fires
    out = np.stack([firm_only, market_only, both, base_only], axis=1)
    out[y == 0] = 0.0
    return out


def exact_posterior(y, fi, mi, F, M, eps, a, b, phi, mu):
    """Brute-force posterior P(Z_f = 1 | y), P(W_m = 1 | y) and the marginal likelihood by enumerating all
    2^(F+M) latent configurations. Only for tiny graphs (tests)."""
    y = np.asarray(y, dtype=float); fi = np.asarray(fi); mi = np.asarray(mi)
    total = 0.0; pz = np.zeros(F); pw = np.zeros(M)
    for zc in itertools.product([0, 1], repeat=F):
        z = np.array(zc)
        for wc in itertools.product([0, 1], repeat=M):
            w = np.array(wc)
            prior = (phi ** z.sum()) * ((1 - phi) ** (F - z.sum())) * (mu ** w.sum()) * ((1 - mu) ** (M - w.sum()))
            lik = np.exp(loglik_given_latents(y, fi, mi, z, w, eps, a, b))
            joint = prior * lik
            total += joint; pz += joint * z; pw += joint * w
    return pz / total, pw / total, total


def rhat(chains):
    """Gelman-Rubin split-R-hat for a list of 1-D chains of equal length."""
    chains = [np.asarray(c) for c in chains]
    n = min(len(c) for c in chains); chains = np.array([c[:n] for c in chains])
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m, n2 = split.shape
    means = split.mean(axis=1); var_within = split.var(axis=1, ddof=1).mean()
    var_between = n2 * means.var(ddof=1)
    var_hat = (n2 - 1) / n2 * var_within + var_between / n2
    return float(np.sqrt(var_hat / var_within))
