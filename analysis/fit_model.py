"""
fit_model.py -- Bayesian estimation of the two-tier latent-shock model (Section 2.4) on the cell data,
posterior predictive checks, attributable fractions (eq. 5), validation against manufacturer-reported reasons
(not used in fitting), a parameter-recovery study on the observed graph, and sensitivity analyses.
Writes results/model_fit.json and results/firm_market_posteriors.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, FIRM_UNIT, RESULTS  # noqa: E402
from model import twotier as tt  # noqa: E402

N_CHAINS = 4; N_ITER = 40000; BURN = 10000; THIN = 5
N_RECOVERY = 30
N_CALIB = 30                       # replicates per null configuration in calibrate_ordering
CALIB_PHI = (0.30, 0.45, 0.65, 0.85)   # firm-shock prevalences of the null grid (phi >= phi*a = c, about 0.27)
CALIB_MU = (0.35, 0.55, 0.75, 0.95, 0.99)    # drug-shock prevalences of the null grid
N_CALIB_WORST = 70                 # extra replicates at the least favourable null
CALIB_EPS_POINTS = {(0.30, 0.95), (0.30, 0.75), (0.45, 0.95), (0.45, 0.99)}   # (phi, mu) also run at the eps quantiles
SEED = 20261014


def index_graph(cells, firm_col="firm_unit"):
    fcat = cells[firm_col].astype("category"); mcat = cells["drug"].astype("category")
    return (fcat.cat.codes.values, mcat.cat.codes.values, len(fcat.cat.categories), len(mcat.cat.categories),
            list(fcat.cat.categories), list(mcat.cat.categories))


def overdispersed_init(seed):
    """Chain-specific starting values spread over the parameter space, so that split-R-hat is informative."""
    r = np.random.default_rng(seed + 99991)
    return dict(eps=r.uniform(0.02, 0.3), a=r.uniform(0.15, 0.95), b=r.uniform(0.15, 0.95), phi=r.uniform(0.1, 0.9), mu=r.uniform(0.1, 0.9))


def fit(y, fi, mi, F, M, seed, n_iter=N_ITER, burn=BURN, thin=THIN, n_chains=N_CHAINS, prior=(1.0, 1.0), n_pres=None):
    chains = [tt.gibbs(y, fi, mi, F, M, n_iter=n_iter, burn=burn, thin=thin, seed=seed + c, init=overdispersed_init(seed + c), prior=prior, n_pres=n_pres)
              for c in range(n_chains)]
    out = {k: np.concatenate([c[k] for c in chains]) for k in tt.PARAMS}
    out["rhat"] = {k: tt.rhat([c[k] for c in chains]) for k in tt.PARAMS}
    out["accept_rate"] = float(np.mean([c["accept_rate"] for c in chains]))
    out["z_post"] = np.mean([c["z_post"] for c in chains], axis=0); out["w_post"] = np.mean([c["w_post"] for c in chains], axis=0)
    out["attribution"] = np.mean([c["attribution"] for c in chains], axis=0)
    return out


def ess(x):
    """Effective sample size of one 1-D chain of draws (initial positive sequence estimator, Geyer 1992)."""
    x = np.asarray(x, dtype=float); n = len(x); x = x - x.mean()
    if n < 10 or x.std() == 0: return float(n)
    acf = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    s = 0.0; k = 1
    while k + 1 < n:
        pair = acf[k] + acf[k + 1]
        if pair <= 0: break
        s += pair; k += 2
    return float(n / (1 + 2 * s))


def summarise(draws):
    s = {}
    for k in tt.PARAMS:
        d = draws[k]; s[k] = dict(median=float(np.median(d)), mean=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
    af = np.array([tt.attributable_fractions(*[draws[k][i] for k in tt.PARAMS]) for i in range(len(draws["eps"]))])
    s["af_firm"] = dict(median=float(np.median(af[:, 0])), lo=float(np.percentile(af[:, 0], 2.5)), hi=float(np.percentile(af[:, 0], 97.5)))
    s["af_market"] = dict(median=float(np.median(af[:, 1])), lo=float(np.percentile(af[:, 1], 2.5)), hi=float(np.percentile(af[:, 1], 97.5)))
    s["p_firm_gt_market"] = float(np.mean(af[:, 0] > af[:, 1]))
    s["nonfirm"] = dict(median=float(np.median(1 - af[:, 0])), lo=float(np.percentile(1 - af[:, 0], 2.5)), hi=float(np.percentile(1 - af[:, 0], 97.5)))
    unsh = 1 - (1 - draws["eps"]) * (1 - draws["mu"] * draws["b"])     # unavailability rate of a cell whose firm is not shocked
    s["unshocked_rate"] = dict(median=float(np.median(unsh)), lo=float(np.percentile(unsh, 2.5)), hi=float(np.percentile(unsh, 97.5)))
    mub = draws["mu"] * draws["b"]; s["mu_b"] = dict(median=float(np.median(mub)), lo=float(np.percentile(mub, 2.5)), hi=float(np.percentile(mub, 97.5)))
    phia = draws["phi"] * draws["a"]; s["phi_a"] = dict(median=float(np.median(phia)), lo=float(np.percentile(phia, 2.5)), hi=float(np.percentile(phia, 97.5)))
    marg = np.array([tt.marginal_prob(*[draws[k][i] for k in tt.PARAMS]) for i in range(len(draws["eps"]))])
    s["marginal_rate"] = dict(median=float(np.median(marg)), lo=float(np.percentile(marg, 2.5)), hi=float(np.percentile(marg, 97.5)))
    cf = np.array([tt.cov_shared_firm(*[draws[k][i] for k in tt.PARAMS]) for i in range(len(draws["eps"]))])
    cm = np.array([tt.cov_shared_market(*[draws[k][i] for k in tt.PARAMS]) for i in range(len(draws["eps"]))])
    s["cov_shared_firm"] = dict(median=float(np.median(cf)), lo=float(np.percentile(cf, 2.5)), hi=float(np.percentile(cf, 97.5)))
    s["cov_shared_market"] = dict(median=float(np.median(cm)), lo=float(np.percentile(cm, 2.5)), hi=float(np.percentile(cm, 97.5)))
    s["rhat"] = draws["rhat"]; s["accept_rate"] = draws["accept_rate"]
    s["ess"] = {k: ess(draws[k]) for k in tt.PARAMS}; s["ess_af_diff"] = ess(af[:, 0] - af[:, 1]); s["n_draws"] = int(len(draws["eps"]))
    return s


N_PPC = 400          # posterior predictive replicate data sets


def posterior_predictive(draws, y, fi, mi, F, M, rng, n_rep=N_PPC):
    """Replicate data from the posterior and compare firm- and drug-level dispersion with the observed."""
    from analysis.descriptives import dispersion_stat
    df = pd.DataFrame(dict(firm_unit=fi, drug=mi, unavailable=y))

    def untargeted(d):
        g = d.groupby("drug")["unavailable"].agg(["size", "sum"]); g = g[g["size"] >= 2]; sh = g["sum"] / g["size"]
        gf = d.groupby("firm_unit")["unavailable"].agg(["size", "sum"]); gf = gf[gf["size"] >= 2]
        return (int((sh == 1).sum()), int((sh == 0).sum()), float(((gf["sum"] == gf["size"]).mean())))   # drugs all unavailable, drugs all available, share of multi-drug firms fully unavailable
    obs_f = dispersion_stat(df, "firm_unit")[0]; obs_d = dispersion_stat(df, "drug")[0]; obs_u = untargeted(df)
    idx = rng.choice(len(draws["eps"]), n_rep, replace=False)
    rep_f = []; rep_d = []; rep_rate = []; rep_u = []
    for i in idx:
        th = {k: draws[k][i] for k in tt.PARAMS}
        yr, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **th)
        df["unavailable"] = yr
        rep_f.append(dispersion_stat(df, "firm_unit")[0]); rep_d.append(dispersion_stat(df, "drug")[0]); rep_rate.append(yr.mean()); rep_u.append(untargeted(df))
    ru = np.array(rep_u)
    return dict(obs_firm_var=float(obs_f), ppc_firm_var_mean=float(np.mean(rep_f)), ppc_firm_p=float(np.mean(np.array(rep_f) >= obs_f)),
                obs_drug_var=float(obs_d), ppc_drug_var_mean=float(np.mean(rep_d)), ppc_drug_p=float(np.mean(np.array(rep_d) >= obs_d)),
                obs_rate=float(y.mean()), ppc_rate_mean=float(np.mean(rep_rate)),
                obs_drugs_all_unavail=obs_u[0], ppc_drugs_all_unavail_mean=float(ru[:, 0].mean()), ppc_drugs_all_unavail_p=float(np.mean(ru[:, 0] >= obs_u[0])),
                obs_drugs_all_avail=obs_u[1], ppc_drugs_all_avail_mean=float(ru[:, 1].mean()), ppc_drugs_all_avail_p=float(np.mean(ru[:, 1] >= obs_u[1])),
                obs_firms_all_unavail_share=float(obs_u[2]), ppc_firms_all_unavail_share_mean=float(ru[:, 2].mean()), ppc_firms_all_unavail_p=float(np.mean(ru[:, 2] >= obs_u[2])))


def recovery_study(truth, fi, mi, F, M, rng, n=N_RECOVERY, seed0=1000):
    """Simulate from `truth` on the observed graph, refit with shorter chains, record coverage, width, bias and the
    distribution of P(AF_firm > AF_market) and of the AF estimates."""
    cover = {k: 0 for k in tt.PARAMS}; width = {k: [] for k in tt.PARAMS}; bias = {k: [] for k in tt.PARAMS}
    af_cover = 0; af_truth = tt.attributable_fractions(**truth); p_list = []; aff_list = []; afm_list = []
    for r in range(n):
        y, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **truth)
        d = fit(y, fi, mi, F, M, seed=seed0 + 17 * r, n_iter=12000, burn=4000, thin=4, n_chains=2)
        for k in tt.PARAMS:
            lo, hi = np.percentile(d[k], [2.5, 97.5]); cover[k] += int(lo <= truth[k] <= hi); width[k].append(hi - lo); bias[k].append(np.median(d[k]) - truth[k])
        af = np.array([tt.attributable_fractions(*[d[k][i] for k in tt.PARAMS]) for i in range(len(d["eps"]))])
        lo, hi = np.percentile(af[:, 0] - af[:, 1], [2.5, 97.5]); af_cover += int(lo <= af_truth[0] - af_truth[1] <= hi)
        p_list.append(float(np.mean(af[:, 0] > af[:, 1]))); aff_list.append(float(np.median(af[:, 0]))); afm_list.append(float(np.median(af[:, 1])))
    return dict(truth=truth, af_truth=dict(firm=float(af_truth[0]), market=float(af_truth[1])), n=n,
                coverage={k: cover[k] / n for k in tt.PARAMS}, mean_width={k: float(np.mean(width[k])) for k in tt.PARAMS},
                mean_bias={k: float(np.mean(bias[k])) for k in tt.PARAMS}, af_diff_coverage=af_cover / n,
                p_firm_gt_market=dict(mean=float(np.mean(p_list)), min=float(np.min(p_list)), max=float(np.max(p_list)), share_above_0_95=float(np.mean(np.array(p_list) > 0.95))),
                af_firm_median_mean=float(np.mean(aff_list)), af_market_median_mean=float(np.mean(afm_list)))


def calibrate_ordering(fitted, draws, y, fi, mi, F, M, rng, p_obs, af_drug_obs, n_rep=N_CALIB, seed0=9000):
    """Frequentist calibration of P(AF_firm > AF_drug). The null phi*a = mu*b (AF_firm = AF_drug, Proposition 1) is
    composite, so data are simulated on the observed graph from a grid of null configurations: shock prevalences
    (phi, mu) on CALIB_PHI x CALIB_MU, the baseline eps at its posterior median and at its 2.5% and 97.5% quantiles,
    and the common value phi*a = mu*b chosen to match the observed cell rate (configurations with a > 1 or b > 1 are
    skipped). Each simulated data set is refitted and P recorded. The valid p-value for a composite null is the
    supremum over the null of P(P_null >= P_obs); the maximum over the grid is reported as a lower bound on it, with
    per-replicate values saved. Refits use 2 chains x 12,000 iterations."""
    r = float(np.mean(y)); out = dict(rate=r, p_obs=p_obs, af_drug_obs=af_drug_obs, nulls=[])
    eps_grid = [("median", fitted["eps"]), ("q025", float(np.percentile(draws["eps"], 2.5))), ("q975", float(np.percentile(draws["eps"], 97.5)))]
    allp = []; i = 0
    for elab, eps in eps_grid:
        c = 1.0 - np.sqrt((1.0 - r) / (1.0 - eps))          # common phi*a = mu*b matching the marginal rate
        for phi in CALIB_PHI:
            for mu in CALIB_MU:
                if elab != "median" and not (phi, mu) in CALIB_EPS_POINTS:
                    continue
                if c / phi > 1 or c / mu > 1:
                    continue
                truth = dict(eps=eps, a=c / phi, b=c / mu, phi=phi, mu=mu)
                ps = []; afd = []
                for rr in range(n_rep):
                    ys, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **truth)
                    d = fit(ys, fi, mi, F, M, seed=seed0 + 1000 * i + 17 * rr, n_iter=12000, burn=4000, thin=4, n_chains=2)
                    af = np.array([tt.attributable_fractions(*[d[k][j] for k in tt.PARAMS]) for j in range(len(d["eps"]))])
                    ps.append(float(np.mean(af[:, 0] > af[:, 1]))); afd.append(float(np.median(af[:, 1])))
                ps = np.array(ps); afd = np.array(afd); allp.extend(ps); i += 1
                k = int(np.sum(ps >= p_obs))
                out["nulls"].append(dict(eps_label=elab, truth=truth, c=float(c), af_true=float(tt.attributable_fractions(**truth)[0]),
                                         p_values=ps.tolist(), af_drug_medians=afd.tolist(), p_mean=float(ps.mean()), p_max=float(ps.max()),
                                         n_ge_obs=k, share_ge_obs=k / len(ps), share_gt_095=float(np.mean(ps > 0.95)),
                                         share_afd_le_obs=float(np.mean(afd <= af_drug_obs)), af_drug_median_mean=float(afd.mean())))
    grid_p = np.array(allp)                              # grid replicates only, before the extra worst-point replicates
    # extra replicates at the least favourable grid point, so that its exceedance share is estimated more precisely
    worst = max(out["nulls"], key=lambda n: (n["share_ge_obs"], n["p_mean"]))
    for rr in range(N_CALIB_WORST):
        ys, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **worst["truth"])
        d = fit(ys, fi, mi, F, M, seed=seed0 + 999000 + 17 * rr, n_iter=12000, burn=4000, thin=4, n_chains=2)
        af = np.array([tt.attributable_fractions(*[d[k][j] for k in tt.PARAMS]) for j in range(len(d["eps"]))])
        worst["p_values"].append(float(np.mean(af[:, 0] > af[:, 1]))); worst["af_drug_medians"].append(float(np.median(af[:, 1])))
        allp.append(worst["p_values"][-1])
    ps = np.array(worst["p_values"]); worst["n_ge_obs"] = int(np.sum(ps >= p_obs)); worst["share_ge_obs"] = worst["n_ge_obs"] / len(ps)
    worst["p_mean"] = float(ps.mean()); worst["p_max"] = float(ps.max()); worst["share_gt_095"] = float(np.mean(ps > 0.95))
    allp = np.array(allp)
    from scipy.stats import beta as _beta
    k, n = worst["n_ge_obs"], len(worst["p_values"])
    out.update(n_nulls=len(out["nulls"]), n_rep=n_rep, pooled_share_ge_obs=float(np.mean(grid_p >= p_obs)), pooled_share_gt_095=float(np.mean(grid_p > 0.95)),
               max_share_ge_obs=worst["share_ge_obs"], worst_truth=worst["truth"], worst_eps_label=worst["eps_label"],
               worst_ci=[float(_beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(_beta.ppf(0.975, k + 1, n - k))],
               share_afd_le_obs_all=float(np.mean([v <= af_drug_obs for nn in out["nulls"] for v in nn["af_drug_medians"]])))
    return out


def validate_reasons(cells, draws, fi, mi):
    """Cell-level cause attribution (eq. 6) vs the manufacturer-reported reason type (not used in fitting)."""
    at = draws["attribution"]
    c = cells.assign(attr_firm=at[:, 0] + at[:, 2], attr_market=at[:, 1] + at[:, 2], attr_firm_only=at[:, 0],
                     attr_market_only=at[:, 1], attr_both=at[:, 2], attr_base=at[:, 3], z_post=draws["z_post"][fi], w_post=draws["w_post"][mi])
    un = c[c["unavailable"] == 1]
    out = dict(cell_attribution_means=dict(firm_only=float(un["attr_firm_only"].mean()), market_only=float(un["attr_market_only"].mean()),
                                           both=float(un["attr_both"].mean()), baseline_only=float(un["attr_base"].mean())))
    g = un.groupby("reason_type").agg(n=("attr_firm", "size"), firm=("attr_firm", "mean"), market=("attr_market", "mean"), z=("z_post", "mean"), w=("w_post", "mean"))
    out["by_reason"] = {k: dict(n=int(r.n), attr_firm=float(r.firm), attr_market=float(r.market), z_post=float(r.z), w_post=float(r.w)) for k, r in g.iterrows()}
    a = un[un["reason_type"] == "firm"]; b = un[un["reason_type"] == "market"]
    if len(a) > 1 and len(b) > 1:
        u, p = stats.mannwhitneyu(a["attr_firm"], b["attr_firm"], alternative="greater")
        out["mwu_attr_firm"] = dict(U=float(u), p=float(p), n_firm=int(len(a)), n_market=int(len(b)), auc=float(u / (len(a) * len(b))))
        u2, p2 = stats.mannwhitneyu(b["attr_market"], a["attr_market"], alternative="greater")
        out["mwu_attr_market"] = dict(U=float(u2), p=float(p2), auc=float(u2 / (len(a) * len(b))))
        d = a["attr_firm"].mean() - b["attr_firm"].mean()
        # bootstrap CI for the difference in mean firm attribution
        rng = np.random.default_rng(5); bs = [rng.choice(a["attr_firm"], len(a)).mean() - rng.choice(b["attr_firm"], len(b)).mean() for _ in range(5000)]
        out["diff_attr_firm"] = dict(diff=float(d), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    # alternative precedence: market-type reason wins when a cell reports both
    rt2 = cells["reason_type_market_first"] if "reason_type_market_first" in cells else cells["reason_type"]
    a2 = un[rt2[un.index] == "firm"]; b2 = un[rt2[un.index] == "market"]
    if len(a2) > 1 and len(b2) > 1:
        u3, p3 = stats.mannwhitneyu(a2["attr_firm"], b2["attr_firm"], alternative="greater")
        out["mwu_attr_firm_market_first"] = dict(p=float(p3), auc=float(u3 / (len(a2) * len(b2))), n_firm=int(len(a2)), n_market=int(len(b2)))
    c.to_csv(os.path.join(RESULTS, "cell_attribution.csv"), index=False)
    return out


def main():
    rng = np.random.default_rng(SEED)
    cells, pres, drugs, firms = load_cells()
    fi, mi, F, M, fnames, mnames = index_graph(cells)
    y = cells["unavailable"].values.astype(int)
    res = dict(firm_unit=FIRM_UNIT, n_cells=int(len(y)), F=F, M=M, n_chains=N_CHAINS, n_iter=N_ITER, burn=BURN, thin=THIN)
    draws = fit(y, fi, mi, F, M, seed=SEED)
    np.savez(os.path.join(RESULTS, "draws_main.npz"), **{k: draws[k] for k in tt.PARAMS})   # used by theory_analysis.py
    res["main"] = summarise(draws)
    res["ppc"] = posterior_predictive(draws, y, fi, mi, F, M, rng)
    res["ppc"]["n_rep"] = N_PPC
    res["reason_validation"] = validate_reasons(cells, draws, fi, mi)
    # firm and market posteriors
    fpost = pd.DataFrame(dict(firm=fnames, z_post=draws["z_post"])).merge(
        cells.groupby("firm_unit").agg(n_drugs=("drug", "size"), n_unavailable=("unavailable", "sum")).reset_index().rename(columns={"firm_unit": "firm"}))
    mpost = pd.DataFrame(dict(drug=mnames, w_post=draws["w_post"])).merge(
        cells.groupby("drug").agg(n_firms=("firm_unit", "size"), n_unavailable=("unavailable", "sum")).reset_index())
    fpost.sort_values("z_post", ascending=False).to_csv(os.path.join(RESULTS, "firm_posteriors.csv"), index=False)
    mpost.sort_values("w_post", ascending=False).to_csv(os.path.join(RESULTS, "market_posteriors.csv"), index=False)
    res["n_firms_zpost_gt_0.5"] = int((draws["z_post"] > 0.5).sum()); res["n_markets_wpost_gt_0.5"] = int((draws["w_post"] > 0.5).sum())
    res["top_firm_posteriors"] = fpost.sort_values("z_post", ascending=False).head(8).to_dict("records")
    res["top_market_posteriors"] = mpost.sort_values("w_post", ascending=False).head(8).to_dict("records")
    # recovery study at the fitted point and under alternative truths on the same graph
    fitted = {k: float(np.median(draws[k])) for k in tt.PARAMS}
    res["recovery"] = recovery_study(fitted, fi, mi, F, M, rng, seed0=1000)
    swapped = dict(eps=fitted["eps"], a=fitted["b"], b=fitted["a"], phi=fitted["mu"], mu=fitted["phi"])
    res["recovery_swapped"] = recovery_study(swapped, fi, mi, F, M, rng, seed0=3000)
    symmetric = dict(eps=fitted["eps"], a=0.45, b=0.45, phi=0.5, mu=0.5)
    res["recovery_symmetric"] = recovery_study(symmetric, fi, mi, F, M, rng, seed0=5000)
    nofirm = dict(eps=fitted["eps"], a=0.0, b=0.6, phi=0.5, mu=0.5)
    res["recovery_nofirm"] = recovery_study(nofirm, fi, mi, F, M, rng, seed0=7000)
    res["calibration"] = calibrate_ordering(fitted, draws, y, fi, mi, F, M, rng, res["main"]["p_firm_gt_market"], res["main"]["af_market"]["median"])
    # prior probability of the ordering (Beta(1,1) and Beta(2,2) priors are symmetric in the two tiers)
    pr = np.random.default_rng(1)
    for nm, (al, be) in (("prior_p_order_beta11", (1, 1)), ("prior_p_order_beta22", (2, 2))):
        res[nm] = float(np.mean(pr.beta(al, be, 200000) * pr.beta(al, be, 200000) > pr.beta(al, be, 200000) * pr.beta(al, be, 200000)))
    # sensitivity analyses
    sens = {}
    # (a) strict outcome: all of the firm's presentations unavailable
    ys = cells["all_unavailable"].values.astype(int)
    d_strict = fit(ys, fi, mi, F, M, seed=SEED + 100)
    np.savez(os.path.join(RESULTS, "draws_strict.npz"), **{k: d_strict[k] for k in tt.PARAMS})
    sens["strict_outcome"] = summarise(d_strict)
    sens["strict_outcome"]["rate"] = float(ys.mean())
    # (b) alternative firm units, each re-aggregated from the presentation level:
    #     NDC labeler code (finer), corporate group (coarser), raw FDA company string (no normalisation)
    #     and the marketer of record (normalised NDC labeler name, the unit of the purchaser-facing firm table)
    for key, col, sd in (("labeler_unit", "firm", 200), ("marketer_unit", "marketer_norm", 210), ("corporate_group", "company_group", 250), ("raw_company", "company_name", 260)):
        cu = pres.assign(firm_unit=pres[col].astype(str)).groupby(["firm_unit", "drug"], as_index=False)["unavailable"].max()
        fiu, miu, Fu, Mu, _, _ = index_graph(cu, "firm_unit")
        sens[key] = summarise(fit(cu["unavailable"].values.astype(int), fiu, miu, Fu, Mu, seed=SEED + sd)); sens[key]["n_cells"] = int(len(cu)); sens[key]["F"] = Fu
    # (b3b) only "Unavailable" counts; Limited Availability treated as available
    yu = cells["unavailable_only"].values.astype(int)
    d_uo = fit(yu, fi, mi, F, M, seed=SEED + 265)
    np.savez(os.path.join(RESULTS, "draws_unavailable_only.npz"), **{k: d_uo[k] for k in tt.PARAMS})
    sens["unavailable_only"] = summarise(d_uo); sens["unavailable_only"]["rate"] = float(yu.mean())
    # (b3c) drop presentations whose NDC labeler code is reported under more than one firm unit (contract manufacturers
    #       and marketers sharing a code), so that no two firms share a labeler
    shared_lab = pres.groupby("firm")["firm_unit"].nunique(); shared_lab = set(shared_lab[shared_lab > 1].index)
    pc = pres[~pres["firm"].isin(shared_lab)].groupby(["firm_unit", "drug"], as_index=False)["unavailable"].max()
    fic, mic, Fc, Mc, _, _ = index_graph(pc, "firm_unit")
    sens["no_shared_labelers"] = summarise(fit(pc["unavailable"].values.astype(int), fic, mic, Fc, Mc, seed=SEED + 268))
    sens["no_shared_labelers"]["n_cells"] = int(len(pc)); sens["no_shared_labelers"]["F"] = Fc; sens["no_shared_labelers"]["labelers_dropped"] = len(shared_lab)
    # the same two firm-unit checks under the strict outcome, where firm dependence through shared production would load
    # on the drug tier
    pcs = pres[~pres["firm"].isin(shared_lab)].groupby(["firm_unit", "drug"], as_index=False)["unavailable"].min()
    f2, m2, F2, M2, _, _ = index_graph(pcs, "firm_unit")
    sens["strict_no_shared_labelers"] = summarise(fit(pcs["unavailable"].values.astype(int), f2, m2, F2, M2, seed=SEED + 269)); sens["strict_no_shared_labelers"]["n_cells"] = int(len(pcs))
    cg = pres.assign(firm_unit=pres["company_group"].astype(str)).groupby(["firm_unit", "drug"], as_index=False)["unavailable"].min()
    f3, m3, F3, M3, _, _ = index_graph(cg, "firm_unit")
    sens["strict_corporate_group"] = summarise(fit(cg["unavailable"].values.astype(int), f3, m3, F3, M3, seed=SEED + 251)); sens["strict_corporate_group"]["n_cells"] = int(len(cg))
    # (b4) allocation notes counted as limited availability
    ya = cells["unavailable_alloc"].values.astype(int)
    sens["allocation_outcome"] = summarise(fit(ya, fi, mi, F, M, seed=SEED + 270)); sens["allocation_outcome"]["rate"] = float(ya.mean())
    # (b5) any constraining note (allocation, "additional lots ... as released", "limited") counted as limited availability
    yn = cells["unavailable_note"].values.astype(int)
    sens["note_outcome"] = summarise(fit(yn, fi, mi, F, M, seed=SEED + 280)); sens["note_outcome"]["rate"] = float(yn.mean()); sens["note_outcome"]["cells_changed"] = int((yn != y).sum())
    # (b5b) discontinued presentations treated as not listed: a cell is unavailable only through its other presentations
    yd = cells["unavailable_nodisc"].values.astype(int)
    sens["no_discontinued"] = summarise(fit(yd, fi, mi, F, M, seed=SEED + 285)); sens["no_discontinued"]["rate"] = float(yd.mean()); sens["no_discontinued"]["cells_changed"] = int((yd != y).sum())
    # (b6) size-neutral outcome: majority (cell unavailable if more than half of its presentations are). A fit on
    #      presentation-level rows was dropped: presentations of one cell share both latents, so treating them as
    #      conditionally independent misspecifies the model, and its chains did not converge from over-dispersed starts.
    ym = (cells["n_unavailable"] / cells["n_presentations"] > 0.5).astype(int).values
    sens["majority_outcome"] = summarise(fit(ym, fi, mi, F, M, seed=SEED + 295)); sens["majority_outcome"]["rate"] = float(ym.mean())
    # (c) injectables only
    inj = cells[cells["injectable"] == 1]
    fi3, mi3, F3, M3, _, _ = index_graph(inj)
    sens["injectables_only"] = summarise(fit(inj["unavailable"].values.astype(int), fi3, mi3, F3, M3, seed=SEED + 300)); sens["injectables_only"]["n_cells"] = int(len(inj))
    # (d) drop single-firm drugs and single-drug firms (cells that carry no sharing information)
    keep = cells.groupby("drug")["drug"].transform("size") >= 2
    keep &= cells.groupby("firm_unit")["firm_unit"].transform("size") >= 2
    sub = cells[keep]; fi4, mi4, F4, M4, _, _ = index_graph(sub)
    sens["shared_cells_only"] = summarise(fit(sub["unavailable"].values.astype(int), fi4, mi4, F4, M4, seed=SEED + 400)); sens["shared_cells_only"]["n_cells"] = int(len(sub))
    # (e) prior sensitivity: Beta(2,2)
    sens["prior_beta22"] = summarise(fit(y, fi, mi, F, M, seed=SEED + 500, prior=(2.0, 2.0)))
    # (f) per-presentation baseline: the baseline cause fires independently for each of a cell's n presentations, so the
    #     cell baseline is 1 - (1 - eps)^n and the mechanical size effect of the main outcome is in the model, not in Z.
    #     The attributable fractions are evaluated at the mean cell size (eps replaced by 1 - (1 - eps)^{mean n}).
    npres = cells["n_presentations"].values.astype(float)
    d_pp = fit(y, fi, mi, F, M, seed=SEED + 600, n_pres=npres)
    eps_bar = 1 - (1 - d_pp["eps"]) ** npres.mean()
    sens["per_presentation_baseline"] = summarise(dict(d_pp, eps=eps_bar))
    sens["per_presentation_baseline"]["eps_per_presentation"] = dict(median=float(np.median(d_pp["eps"])), lo=float(np.percentile(d_pp["eps"], 2.5)), hi=float(np.percentile(d_pp["eps"], 97.5)))
    sens["per_presentation_baseline"]["mean_cell_size"] = float(npres.mean()); sens["per_presentation_baseline"]["rhat_max"] = float(max(d_pp["rhat"].values()))
    sens["per_presentation_baseline"]["n_firms_zpost_gt_0.5"] = int((d_pp["z_post"] > 0.5).sum())
    res["sensitivity_rhat_max"] = float(max(max(v["rhat"].values()) for v in sens.values()))
    res["sensitivity"] = sens
    with open(os.path.join(RESULTS, "model_fit.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["main"], indent=1)); print(json.dumps(res["ppc"], indent=1)); print(json.dumps(res["reason_validation"], indent=1)); print(json.dumps(res["recovery"], indent=1))


if __name__ == "__main__":
    main()
