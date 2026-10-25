"""
theory_analysis.py -- empirical counterparts of the formal results of Section 2.5 (model/theory.py).

(1) Method-of-moments estimates (Theorem 1 applied to the unbiased row moments, Section 2.6), as a check on the
    Bayesian fit that uses no prior and no MCMC, with row-bootstrap intervals.
(2) Theorem 2 on the posterior draws: probability that every listed firm of a drug is unavailable, O_n, against the
    independence benchmark, with the observed share by number of listed firms and its posterior predictive interval;
    smallest number of listed firms n*(alpha) that brings O_n below alpha.
(3) Proposition 2 on the posterior draws: joint outage lift of two drugs as a function of the number of shared firms,
    and a posterior predictive check on the observed pairs of drugs.
Writes results/theory.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, RESULTS  # noqa: E402
from analysis.fit_model import index_graph  # noqa: E402
from model import theory as th  # noqa: E402
from model import twotier as tt  # noqa: E402

SEED = 20261015
N_BOOT = 2000
N_POST = 4000          # posterior draws used for the closed forms
N_PPC = 1000           # posterior draws used for the predictive checks
N_MAX = 8              # O_n reported for n = 1..N_MAX
ALPHAS = (0.10, 0.05)


def q(x, p):
    return float(np.percentile(x, p))


def summ(x):
    x = np.asarray(x, dtype=float)
    return dict(median=float(np.median(x)), lo=q(x, 2.5), hi=q(x, 97.5))


def mom_with_bootstrap(y, fi, mi, rng):
    est = th.mom_estimate(y, fi, mi)
    F = fi.max() + 1; M = mi.max() + 1
    rows_f = [np.flatnonzero(fi == f) for f in range(F)]; rows_m = [np.flatnonzero(mi == m) for m in range(M)]
    keys = ("phi", "p0", "p1", "af_firm"), ("mu", "q0", "q1", "af_drug")
    m1_diff = []        # over-identifying restriction of Lemma 1: M1 is the same from firm rows and from drug rows
    bs = {k: [] for k in keys[0] + keys[1]}; hank_f = []; hank_d = []; deg_f = 0; deg_d = 0; order = []; eps_b = []
    for _ in range(N_BOOT):
        # one-way row bootstraps: firms resampled for the firm side, drugs for the drug side (each estimator is an
        # average over rows of one tier; dependence between rows through the other tier is ignored)
        sf = rng.integers(0, F, F); sd = rng.integers(0, M, M)
        yf = np.concatenate([y[rows_f[f]] for f in sf]); ff = np.concatenate([np.full(len(rows_f[f]), i) for i, f in enumerate(sf)])
        yd = np.concatenate([y[rows_m[m]] for m in sd]); dd = np.concatenate([np.full(len(rows_m[m]), i) for i, m in enumerate(sd)])
        mf, _ = th.row_moments(yf, ff); md, _ = th.row_moments(yd, dd)
        hank_f.append(mf[1] - mf[0] ** 2); hank_d.append(md[1] - md[0] ** 2); m1_diff.append(md[0] - mf[0])
        f = th.prony(*mf); d = th.prony(*md)
        if f is None:
            deg_f += 1
        else:
            for k, v in zip(keys[0], (f[0], f[1], f[2], 1 - f[1] / mf[0])): bs[k].append(v)
        if d is None:
            deg_d += 1
        else:
            for k, v in zip(keys[1], (d[0], d[1], d[2], 1 - d[1] / md[0])): bs[k].append(v)
        if f is not None and d is not None:
            order.append(d[1] > f[1])
            eps_b.append(th.params_from_rates(f[0], f[1], f[2], d[0], d[1], d[2])["eps"])
    est["boot"] = {k: summ(v) for k, v in bs.items() if len(v)}
    est["boot_hankel_firm"] = summ(hank_f); est["boot_hankel_drug"] = summ(hank_d)
    est["boot_share_degenerate_firm"] = deg_f / N_BOOT; est["boot_share_degenerate_drug"] = deg_d / N_BOOT
    est["boot_share_hankel_drug_le0"] = float(np.mean(np.array(hank_d) <= 0))
    est["boot_p_order"] = float(np.mean(order)) if order else None                 # among replicates where both sides invert
    est["boot_share_both_valid"] = len(order) / N_BOOT
    est["boot_share_order_all"] = float(np.sum(order)) / N_BOOT                     # unconditional share with q0 > p0
    est["m1_firm_rows"] = est["firm_moments"][0]; est["m1_drug_rows"] = est["drug_moments"][0]
    est["boot_m1_diff"] = summ(m1_diff)
    est["boot_eps"] = summ(eps_b) if eps_b else None
    est["boot_share_eps_negative"] = float(np.mean(np.array(eps_b) < 0)) if eps_b else None
    est["n_boot"] = N_BOOT
    return est


def outage_analysis(draws, y, fi, mi, F, M, rng, label):
    n_draw = len(draws["eps"]); idx = rng.choice(n_draw, min(N_POST, n_draw), replace=False)
    T = [{k: float(draws[k][i]) for k in tt.PARAMS} for i in idx]
    out = dict(label=label)
    # Proposition 1: log-availability shares -log(1-x)/-log(1-M_1) of baseline, firm tier and drug tier
    tot = np.array([-np.log(1 - tt.marginal_prob(**t)) for t in T])
    lf = np.array([-np.log(1 - t["phi"] * t["a"]) for t in T]) / tot
    ld = np.array([-np.log(1 - t["mu"] * t["b"]) for t in T]) / tot
    out["logshare_firm"] = summ(lf); out["logshare_drug"] = summ(ld); out["logshare_base"] = summ(1 - lf - ld)
    out["p_phia_gt_mub"] = float(np.mean(draws["phi"] * draws["a"] > draws["mu"] * draws["b"]))   # all draws; equals P(AF_firm > AF_drug)
    O = np.array([[th.outage(n, **t) for n in range(1, N_MAX + 1)] for t in T])
    I = np.array([[th.outage_independent(n, **t) for n in range(1, N_MAX + 1)] for t in T])
    out["O"] = [summ(O[:, j]) for j in range(N_MAX)]; out["O_indep"] = [summ(I[:, j]) for j in range(N_MAX)]
    out["O_ratio_to_indep"] = [summ(O[:, j] / I[:, j]) for j in range(N_MAX)]
    out["second_source_reduction"] = summ(1 - O[:, 1] / O[:, 0])            # relative reduction from 1 to 2 firms
    out["fifth_source_reduction"] = summ(1 - O[:, 4] / O[:, 3])             # from 4 to 5
    q1 = np.array([th.drug_row_rates(**t)[1] for t in T]); q0 = np.array([th.drug_row_rates(**t)[0] for t in T])
    out["q0"] = summ(q0); out["q1"] = summ(q1); out["floor_reduction"] = summ(1 - q1)
    for a in ALPHAS:
        ns = np.array([th.n_star(a, **t, n_max=200) for t in T], dtype=float)
        ni = np.array([np.ceil(np.log(a) / np.log(th.drug_moment(1, **t))) for t in T])
        out[f"n_star_{a}"] = dict(summ(np.minimum(ns, 200)), share_inf=float(np.mean(~np.isfinite(ns))), indep=summ(ni))
    # observed share of drugs with n listed firms that have every firm unavailable, and posterior predictive interval
    k = np.bincount(mi, minlength=M); u = np.bincount(mi, weights=y, minlength=M)
    bins = [1, 2, 3, 4, 5, 6]                                      # 6 = six or more
    nb = np.minimum(k, 6)
    obs = []
    for b in bins:
        sel = nb == b
        obs.append(dict(n=b, drugs=int(sel.sum()), all_unavailable=int((u[sel] == k[sel]).sum())))
    pp = np.zeros((N_PPC, len(bins))); pp_pairs = []
    pidx = rng.choice(n_draw, min(N_PPC, n_draw), replace=False)
    pairs = pair_table(fi, mi, M)
    for r, i in enumerate(pidx):
        t = {kk: float(draws[kk][i]) for kk in tt.PARAMS}
        yr, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **t)
        ur = np.bincount(mi, weights=yr, minlength=M); full = ur == k
        for j, b in enumerate(bins):
            pp[r, j] = full[nb == b].sum()
        pp_pairs.append(pair_counts(full, pairs))
    for j, o in enumerate(obs):
        o["ppc_mean"] = float(pp[:, j].mean()); o["ppc_lo"] = q(pp[:, j], 2.5); o["ppc_hi"] = q(pp[:, j], 97.5)
        o["ppc_p_ge"] = float(np.mean(pp[:, j] >= o["all_unavailable"]))
    out["observed_by_n"] = obs
    # Proposition 2: lift of joint outage over independence for two drugs with three listed firms each
    L = np.array([[th.joint_outage(3, 3, kk, **t) / th.outage(3, **t) ** 2 for kk in range(4)] for t in T])
    out["lift_33"] = [summ(L[:, kk]) for kk in range(4)]
    obs_full = (u == k)
    oc = pair_counts(obs_full, pairs); ppa = np.array(pp_pairs)
    out["pairs"] = [dict(shared=s, pairs=int(pairs["n"][j]), mean_listed=float((k[pairs["groups"][j][:, 0]] + k[pairs["groups"][j][:, 1]]).mean() / 2), both_out=int(oc[j]), ppc_mean=float(ppa[:, j].mean()),
                         ppc_lo=q(ppa[:, j], 2.5), ppc_hi=q(ppa[:, j], 97.5), ppc_p_ge=float(np.mean(ppa[:, j] >= oc[j])))
                    for j, s in enumerate(pairs["labels"])]
    return out


def pair_table(fi, mi, M):
    """Pairs of drugs with at least two listed firms each, grouped by the number of listed firms they share
    (0, 1, 2 or more)."""
    firms_of = [set(fi[mi == m]) for m in range(M)]
    multi = [m for m in range(M) if len(firms_of[m]) >= 2]
    groups = {0: [], 1: [], 2: []}
    for i, m in enumerate(multi):
        for m2 in multi[i + 1:]:
            groups[min(2, len(firms_of[m] & firms_of[m2]))].append((m, m2))
    return dict(labels=["0", "1", "2+"], groups=[np.array(groups[g]) for g in (0, 1, 2)], n=[len(groups[g]) for g in (0, 1, 2)])


def pair_counts(full, pairs):
    return [int(np.sum(full[g[:, 0]] & full[g[:, 1]])) if len(g) else 0 for g in pairs["groups"]]


def fullout_power(cells, fi, mi, F, M, draws_main, draws_strict, rng, n_sim=150, n_perm=400):
    """Power of the model-free fully-out check (outcome_variants.fully_out_check) against data simulated on the observed
    graph from the two-tier model: share of simulated data sets with p < 0.05, for the posterior medians of the main and
    strict fits and for three truths with larger drug components."""
    from analysis.outcome_variants import fully_out_check
    med = {k: float(np.median(draws_main[k])) for k in tt.PARAMS}
    meds = {k: float(np.median(draws_strict[k])) for k in tt.PARAMS}
    truths = {"main_fit": med, "strict_fit": meds, "mub_015": dict(med, b=0.5, mu=0.3),
              "swapped": dict(eps=med["eps"], a=med["b"], b=med["a"], phi=med["mu"], mu=med["phi"]), "near_super_event": dict(med, b=0.8, mu=0.3)}
    out = {}
    for name, t in truths.items():
        rej = 0
        for _ in range(n_sim):
            y, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **t)
            r = fully_out_check(cells, y, rng, n_perm=n_perm)
            rej += r["p_total"] < 0.05
        out[name] = dict(truth=t, power=rej / n_sim, mub=t["mu"] * t["b"])
        print("power", name, out[name]["power"])
    out["n_sim"] = n_sim; out["n_perm"] = n_perm
    return out


def derived_diagnostics(draws):
    """Split-R-hat and effective sample sizes (per chain, summed over chains) of the derived quantities of the main fit:
    mu*b, AF_drug, AF_firm, AF_firm - AF_drug, and b and mu themselves. The stored draws are the N_CHAINS chains of
    fit_model.py concatenated in order, so they are split back into chains here."""
    from analysis.fit_model import N_CHAINS, ess
    n = len(draws["eps"]); per = n // N_CHAINS
    aff, afd = tt.attributable_fractions(draws["eps"], draws["a"], draws["b"], draws["phi"], draws["mu"])
    q = dict(mu_b=draws["mu"] * draws["b"], af_drug=afd, af_firm=aff, af_diff=aff - afd, b=draws["b"], mu=draws["mu"])
    out = dict(n_chains=N_CHAINS, draws_per_chain=per)
    for k, x in q.items():
        chains = [x[c * per:(c + 1) * per] for c in range(N_CHAINS)]
        e = [ess(c) for c in chains]
        out[k] = dict(ess_sum=float(sum(e)), ess_chain_min=float(min(e)), ess_chain_max=float(max(e)), rhat=float(tt.rhat(chains)), ess_pooled_naive=float(ess(x)))
    return out


def corollary2(d, fi, mi, n_draws=1000):
    """Corollary 2 on the observed graph at the posterior of the strict-outcome fit (complete outages): expected number of
    drugs with no listed firm available, its variance, the variance if no two drugs shared a listed firm, and their ratio
    (the variance inflation due to shared suppliers). Pairwise joint outages use Proposition 2, vectorised over draws."""
    M = int(mi.max()) + 1
    firms_of = [set(fi[mi == m].tolist()) for m in range(M)]; n = np.array([len(x) for x in firms_of])
    pairs = [(i, j, len(firms_of[i] & firms_of[j])) for i in range(M) for j in range(i + 1, M) if firms_of[i] & firms_of[j]]
    idx = np.linspace(0, len(d["eps"]) - 1, n_draws).astype(int)
    t = {k: np.asarray(d[k])[idx] for k in ("eps", "a", "b", "phi", "mu")}
    O = np.array([th.outage(int(x), **t) for x in n])                     # drugs x draws
    var0 = (O * (1 - O)).sum(axis=0)
    cov = sum(th.joint_outage(int(n[i]), int(n[j]), k, **t) - O[i] * O[j] for i, j, k in pairs)
    var = var0 + 2 * cov; ratio = var / var0
    q = lambda x: dict(median=float(np.median(x)), lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)))
    return dict(n_drugs=M, n_pairs_sharing=len(pairs), n_pairs=M * (M - 1) // 2, mean=q(O.sum(axis=0)), sd=q(np.sqrt(var)),
                sd_disjoint=q(np.sqrt(var0)), var_ratio=q(ratio))


def corollary1_ppc(d, cells, fi, mi, F, M, n_rep=600, seed=SEED + 41):
    """Posterior predictive distribution of the four observed statistics that Corollary 1 explains (firm and drug dispersion
    ratios, same-firm and unrelated conditional co-unavailability), computed exactly as for the data (analysis/descriptives.py)
    on data sets simulated from posterior draws on the observed graph; returns 95% intervals and the upper-tail probability
    P(simulated >= observed)."""
    from analysis import descriptives as de
    rng = np.random.default_rng(seed); idx = rng.choice(len(d["eps"]), n_rep, replace=False)
    base = cells[["firm_unit", "drug"]].reset_index(drop=True)
    def stats_of(y):
        df = base.assign(unavailable=np.asarray(y).astype(int)); out = {}
        for grp, key in (("firm_unit", "firm"), ("drug", "drug")):
            v, e, _ = de.dispersion_stat(df, grp); out[f"ratio_{key}"] = v / e
        co = de.conditional_co_unavailability(df)
        out["co_same_firm"] = co["same_firm"]["p_second_given_first"]; out["co_unrelated"] = co["unrelated"]["p_second_given_first"]
        return out
    obs = stats_of(cells["unavailable"].values)
    sims = []
    for i in idx:
        y, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **{k: float(d[k][i]) for k in ("eps", "a", "b", "phi", "mu")})
        sims.append(stats_of(y))
    res = {}
    for k in obs:
        x = np.array([s_[k] for s_ in sims]); x = x[np.isfinite(x)]
        res[k] = dict(observed=float(obs[k]), lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)), p_upper=float(np.mean(x >= obs[k])))
    res["n_rep"] = n_rep
    return res


def corollary1(d, deg, ppc=None):
    """Corollary 1 at the posterior of the main fit: the probability that the covariance ordering D_F > D_M holds, its
    agreement with the attributable ordering phi a > mu b, and the model-implied value of the dispersion ratio of Table 2
    on the observed row sizes. The statistic is the variance of the shares across rows with k_g >= 2 cells divided by
    the mean of M_1 (1 - M_1)/k_g; by Corollary 1 its expectation is, ignoring the estimation of the pooled rate and the
    dependence between rows, mean_g Var(S_g) / mean_g (M_1 (1 - M_1)/k_g) = 1 + D mean_g(1 - 1/k_g) / (M_1 (1 - M_1) mean_g(1/k_g))."""
    n = len(d["eps"]); P = [{k: float(d[k][i]) for k in ("eps", "a", "b", "phi", "mu")} for i in range(n)]
    cov_order = np.array([th.covariance_order_firm(**t) for t in P]); af_order = np.array([t["phi"] * t["a"] > t["mu"] * t["b"] for t in P])
    kf = deg["firm"][deg["firm"] >= 2].astype(float); kd = deg["drug"][deg["drug"] >= 2].astype(float)

    def ratio(ks, side, t):
        m1 = th.firm_moment(1, **t)
        return float(np.mean([th.share_variance(k, side, **t) for k in ks]) / np.mean(m1 * (1 - m1) / ks))
    rf = np.array([ratio(kf, "firm", t) for t in P]); rd = np.array([ratio(kd, "drug", t) for t in P])
    kf, kd = float(kf.mean()), float(kd.mean())
    q = lambda x: dict(median=float(np.median(x)), lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)))
    cof = np.array([th.co_unavailability("firm", **t) for t in P]); cod = np.array([th.co_unavailability("drug", **t) for t in P])
    m1 = np.array([th.firm_moment(1, **t) for t in P])
    return dict(co_same_firm=q(cof), co_same_drug=q(cod), co_unrelated=q(m1), ppc=ppc,
                p_cov_order_firm=float(cov_order.mean()), p_af_order_firm=float(af_order.mean()), p_orders_agree=float((cov_order == af_order).mean()),
                mean_row_firm=kf, mean_row_drug=kd, ratio_firm=q(rf), ratio_drug=q(rd))


def main():
    rng = np.random.default_rng(SEED)
    cells, pres, drugs, firms = load_cells()
    fi, mi, F, M, _, _ = index_graph(cells)
    res = dict()
    outcomes = dict(main=cells["unavailable"].values.astype(int), strict=cells["all_unavailable"].values.astype(int),
                    unavailable_only=cells["unavailable_only"].values.astype(int),
                    majority=(cells["n_unavailable"] / cells["n_presentations"] > 0.5).astype(int).values)
    res["mom"] = {k: mom_with_bootstrap(v, fi, mi, rng) for k, v in outcomes.items()}
    deg = dict(firm=np.bincount(fi), drug=np.bincount(mi))
    res["rows_ge3"] = dict(firms=int((deg["firm"] >= 3).sum()), drugs=int((deg["drug"] >= 3).sum()))
    for label in ("main", "strict", "unavailable_only"):
        d = dict(np.load(os.path.join(RESULTS, f"draws_{label}.npz")))
        res[f"outage_{label}"] = outage_analysis(d, outcomes[label], fi, mi, F, M, rng, label)
    res["ess_derived"] = derived_diagnostics(dict(np.load(os.path.join(RESULTS, "draws_main.npz"))))
    res["fullout_power"] = fullout_power(cells, fi, mi, F, M, dict(np.load(os.path.join(RESULTS, "draws_main.npz"))),
                                         dict(np.load(os.path.join(RESULTS, "draws_strict.npz"))), rng)
    res["corollary2"] = corollary2(dict(np.load(os.path.join(RESULTS, "draws_strict.npz"))), fi, mi)
    dm = dict(np.load(os.path.join(RESULTS, "draws_main.npz")))
    res["corollary1"] = corollary1(dm, deg, ppc=corollary1_ppc(dm, cells, fi, mi, F, M))
    with open(os.path.join(RESULTS, "theory.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["mom"]["main"], indent=1, default=float))
    for label in ("main", "strict"):
        o = res[f"outage_{label}"]
        print(label, "O", [round(x["median"], 3) for x in o["O"]], "indep", [round(x["median"], 3) for x in o["O_indep"]])
        print(" obs", o["observed_by_n"]); print(" pairs", o["pairs"]); print(" lift", o["lift_33"])
        print(" n*", {a: o[f"n_star_{a}"] for a in ALPHAS}, "2nd", o["second_source_reduction"], "floor", o["floor_reduction"])


if __name__ == "__main__":
    main()
