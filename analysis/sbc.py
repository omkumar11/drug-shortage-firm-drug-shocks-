"""
sbc.py -- simulation-based calibration of the full sampler (Talts et al. 2018, arXiv:1804.06788). For each replicate: draw
theta from the prior (Beta(1,1) on every parameter, as in the main fit), simulate outcomes on the graph, run the sampler (one
chain, thinned), and record the rank of the true value among the kept posterior draws. If the sampler targets the correct
posterior, each rank is uniform on {0, ..., L}. Uniformity is tested with a chi-square test on equal-count bins for the five
parameters, the two attributable fractions and the two per-cell shock probabilities phi*a and mu*b.
SBC needs approximately independent posterior draws (Talts et al. 2018, Sect. 5.1), so the main run thins each chain far
beyond the integrated autocorrelation time of the slowest quantity (mu*b, about 225 iterations on this graph) and records
per-replicate split-R-hat, effective sample size and lag-1 autocorrelation of the kept draws. A short run with the chain
length first used (8,000 iterations thinned to 200, i.e. about 20-60 effective draws) is reported alongside to show what
autocorrelated draws do to the rank histogram. The transition kernel itself is checked by Geweke's (2004) joint-distribution
test: alternating one kernel sweep with a redraw of the data from p(y | theta, Z, W) is a Markov chain on (theta, Z, W, y)
whose stationary marginal for theta must be the prior; the thinned chain is compared with the prior by Kolmogorov-Smirnov
distances adjusted for its effective sample size. Because a stationary kernel can still fail to reach the posterior's
mass from a given start within the run, a start-dependence check refits the replicates whose true mu*b ranked in the top
tenth, and as many unflagged replicates, from a second start (the true parameters and latent states) and compares the two
chains; the same question is asked of the observed data with a chain started with every drug shocked. Replicates are
independent and are fitted in parallel (SBC_WORKERS processes). The main run is repeated on N_BATCHES independent batches of
prior draws (different seeds) and the ranks are also tested pooled. Finally, on the drug-only submodel (a = phi = 0), whose
posterior of (eps, b, mu) is exact on a grid (a sum over each drug's two latent states), the sampler with a and phi fixed is
compared with exact posterior draws replicate by replicate. Writes results/sbc.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, RESULTS  # noqa: E402
from analysis.fit_model import index_graph  # noqa: E402
from model import twotier as tt  # noqa: E402
from analysis.fit_model import ess  # noqa: E402

SEED = 20261017
N_SBC = int(os.environ.get("SBC_N", 200))            # prior draws
# short run: the chain length first used (kept draws are autocorrelated; reported to show the effect)
SHORT_N_ITER, SHORT_BURN, SHORT_THIN = 8000, 3000, 25                      # L = 200
# main run: draws thinned to approximate independence (thin >> integrated autocorrelation time of mu*b, ~225 iterations)
N_ITER = int(os.environ.get("SBC_N_ITER", 110000)); BURN = int(os.environ.get("SBC_BURN", 10000)); THIN = int(os.environ.get("SBC_THIN", 500))   # L = 200
GEWEKE_N_ITER = int(os.environ.get("SBC_GEWEKE_N_ITER", 3000000)); GEWEKE_THIN = 50; GEWEKE_PRIOR_N = 2000000
GEWEKE_REGION = dict(mu_min=0.8, eps_max=0.4)       # the region of the prior where the first batch's flagged replicates lie
N_BATCHES = int(os.environ.get("SBC_N_BATCHES", 4)); BATCH_SEEDS = [SEED, SEED + 5000, SEED + 6000, SEED + 7000]
N_RESTRICTED = int(os.environ.get("SBC_N_RESTRICTED", 200)); RESTRICTED_GRID = int(os.environ.get("SBC_RESTRICTED_GRID", 200))
OUT_NAME = os.environ.get("SBC_OUT", "sbc.json")
WORKERS = int(os.environ.get("SBC_WORKERS", max(1, min(8, (os.cpu_count() or 2) - 2))))
RHAT_FLAG = 1.1                                     # two starts are called disagreeing when their split-R-hat exceeds this
OBS_HIGH_START = dict(eps=0.02, a=0.5, b=0.9, phi=0.3, mu=0.99)   # observed-data chain started with every drug shocked
N_BINS = 10
QUANTITIES = ("eps", "a", "b", "phi", "mu", "af_firm", "af_drug", "phi_a", "mu_b")


def derived(th):
    """The nine quantities whose ranks are recorded, from a dict (or dict of arrays) of the five parameters."""
    aff, afd = tt.attributable_fractions(th["eps"], th["a"], th["b"], th["phi"], th["mu"])
    return dict(eps=th["eps"], a=th["a"], b=th["b"], phi=th["phi"], mu=th["mu"], af_firm=aff, af_drug=afd,
                phi_a=th["phi"] * th["a"], mu_b=th["mu"] * th["b"])


def rank_of(true_value, draws):
    """Number of posterior draws strictly below the true value: uniform on {0, ..., L} under a correct sampler."""
    return int(np.sum(np.asarray(draws) < true_value))


def uniformity_test(ranks, L, n_bins=N_BINS):
    """Chi-square test of uniformity of ranks in {0, ..., L}: the L + 1 integer values are cut into n_bins consecutive
    groups of (nearly) equal size, and the observed counts are compared with the expected counts under uniformity."""
    ranks = np.asarray(ranks); n_vals = L + 1
    cuts = [int(np.floor(j * n_vals / n_bins)) for j in range(n_bins + 1)]      # integer cut points on {0, ..., L + 1}
    counts = np.array([np.sum((ranks >= cuts[j]) & (ranks < cuts[j + 1])) for j in range(n_bins)])
    expected = np.array([(cuts[j + 1] - cuts[j]) / n_vals * len(ranks) for j in range(n_bins)])
    chi2, p = stats.chisquare(counts, f_exp=expected)
    return dict(chi2=float(chi2), p=float(p), counts=counts.tolist(), expected=expected.tolist(), mean_rank=float(ranks.mean()), L=L)


def draw_replicates(fi, mi, F, M, n_sbc=N_SBC, seed=SEED):
    """The prior draws, simulated data sets, true latent states and starting values of the n_sbc replicates. They depend
    only on `seed`, so runs with different chain settings refit the same data sets."""
    rng = np.random.default_rng(seed); reps = []
    for r in range(n_sbc):
        truth = {k: float(rng.beta(1.0, 1.0)) for k in tt.PARAMS}
        y, z, w = tt.simulate(fi, mi, F, M, rng=rng, **truth)
        init = {k: float(v) for k, v in zip(tt.PARAMS, rng.uniform(0.05, 0.95, 5))}
        reps.append(dict(r=r, truth=truth, y=y, z=z, w=w, init=init, seed=seed + 7 * r))
    return reps


def chain_summary(d, truth):
    """Ranks, 90% coverage and diagnostics of one chain's kept draws against the truth."""
    post = derived({k: d[k] for k in tt.PARAMS}); tv = derived(truth); L = len(d["eps"])
    out = dict(L=L, ranks={}, cover={}, post_mean_mu_b=float(np.mean(post["mu_b"])), post_mean_eps=float(np.mean(post["eps"])))
    for k in QUANTITIES:
        out["ranks"][k] = rank_of(tv[k], post[k])
        lo, hi = np.percentile(post[k], [5, 95]); out["cover"][k] = int(lo <= tv[k] <= hi)
    mb = np.asarray(post["mu_b"]); half = L // 2
    out["rhat_mu_b"] = tt.rhat([mb[:half], mb[half:2 * half]])
    out["ess_mu_b"] = ess(mb); out["ess_af_drug"] = ess(post["af_drug"]); out["ess_eps"] = ess(post["eps"])
    out["lag1_mu_b"] = float(np.corrcoef(mb[:-1], mb[1:])[0, 1]) if mb.std() > 0 else 0.0
    out["mu_b_draws"] = mb
    return out


def fit_replicate(args):
    """Worker: fit one replicate from its stored start (and, if asked, a second chain from the truth)."""
    rep, n_iter, burn, thin, from_truth = args
    fi, mi, F, M = _GRAPH["fi"], _GRAPH["mi"], _GRAPH["F"], _GRAPH["M"]
    d = tt.gibbs(rep["y"], fi, mi, F, M, n_iter=n_iter, burn=burn, thin=thin, seed=rep["seed"], init=rep["init"])
    out = dict(r=rep["r"], cold=chain_summary(d, rep["truth"]))
    if from_truth:
        d2 = tt.gibbs(rep["y"], fi, mi, F, M, n_iter=n_iter, burn=burn, thin=thin, seed=rep["seed"] + 3,
                      init=dict(rep["truth"]), init_latent=(rep["z"], rep["w"]))
        out["truth_start"] = chain_summary(d2, rep["truth"])
        out["rhat_between_starts_mu_b"] = tt.rhat([out["cold"]["mu_b_draws"], out["truth_start"]["mu_b_draws"]])
        out["rhat_between_starts_eps"] = tt.rhat([_eps_draws(d), _eps_draws(d2)])
    for key in ("cold", "truth_start"):
        if key in out: out[key].pop("mu_b_draws")
    return out


def _eps_draws(d):
    return np.asarray(d["eps"])


_GRAPH = {}


def _init_worker(fi, mi, F, M):
    _GRAPH.update(fi=fi, mi=mi, F=F, M=M)


def run_parallel(fi, mi, F, M, jobs, label):
    """Fit the jobs (tuples for fit_replicate) over WORKERS processes, in replicate order."""
    _init_worker(fi, mi, F, M); results = []
    if WORKERS <= 1:
        for j, job in enumerate(jobs):
            results.append(fit_replicate(job))
            if (j + 1) % 25 == 0: print(f"sbc{label} {j + 1}/{len(jobs)}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(fi, mi, F, M)) as pool:
        for j, res in enumerate(pool.map(fit_replicate, jobs, chunksize=1)):
            results.append(res)
            if (j + 1) % 25 == 0: print(f"sbc{label} {j + 1}/{len(jobs)}", flush=True)
    return results


def run_sbc(fi, mi, F, M, n_iter, burn, thin, n_sbc=N_SBC, seed=SEED, label=""):
    """One SBC run on the graph (fi, mi): n_sbc prior draws, one chain each, ranks and 90% coverage of the nine quantities,
    with per-replicate diagnostics of the kept draws of mu*b (split-R-hat of the two halves, effective sample size, lag-1
    autocorrelation)."""
    reps = draw_replicates(fi, mi, F, M, n_sbc, seed)
    fits = run_parallel(fi, mi, F, M, [(rep, n_iter, burn, thin, False) for rep in reps], label)
    ranks = {k: [f["cold"]["ranks"][k] for f in fits] for k in QUANTITIES}
    cover90 = {k: sum(f["cold"]["cover"][k] for f in fits) for k in QUANTITIES}
    diag = {k: [f["cold"][k] for f in fits] for k in ("rhat_mu_b", "ess_mu_b", "ess_af_drug", "ess_eps", "lag1_mu_b")}
    L = fits[0]["cold"]["L"]; degenerate = int(sum(rep["y"].min() == rep["y"].max() for rep in reps))
    truths = {k: [rep["truth"][k] for rep in reps] for k in tt.PARAMS}; y_means = [float(rep["y"].mean()) for rep in reps]
    out = dict(n_sbc=n_sbc, L=L, n_iter=n_iter, burn=burn, thin=thin, n_bins=N_BINS, n_degenerate_datasets=degenerate,
               F=int(F), M=int(M), n_cells=int(len(fi)),
               ranks={k: ranks[k] for k in QUANTITIES},
               tests={k: uniformity_test(ranks[k], L) for k in QUANTITIES},
               coverage90={k: cover90[k] / n_sbc for k in QUANTITIES})
    out["expected_per_bin"] = n_sbc / N_BINS
    out["expected_top_bin"] = out["tests"]["eps"]["expected"][-1]; out["expected_bottom_bin"] = out["tests"]["eps"]["expected"][0]
    out["top_bin"] = {k: out["tests"][k]["counts"][-1] for k in QUANTITIES}
    out["bottom_bin"] = {k: out["tests"][k]["counts"][0] for k in QUANTITIES}
    out["min_p"] = float(min(t["p"] for t in out["tests"].values())); out["max_p"] = float(max(t["p"] for t in out["tests"].values()))
    out["n_p_below_005"] = int(sum(t["p"] < 0.05 for t in out["tests"].values()))
    out["truths"] = truths; out["y_means"] = y_means
    out["flagged"] = flagged_summary(out)
    out["diagnostics"] = {k: dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in diag.items()}
    out["diagnostics"]["n_rhat_above_105"] = int(np.sum(np.asarray(diag["rhat_mu_b"]) > 1.05))
    out["diagnostics"]["n_ess_mu_b_below_100"] = int(np.sum(np.asarray(diag["ess_mu_b"]) < 100))
    out["per_replicate"] = {k: [float(x) for x in v] for k, v in diag.items()}      # so flagged and unflagged replicates can be compared
    out["_reps"] = reps                                   # kept in memory for the start-dependence check, not written
    return out


def restart_check(fi, mi, F, M, main_run, seed=SEED + 23):
    """Start-dependence check. The replicates whose true mu*b ranked in the top tenth of the cold-start chain (flagged) and
    an equal number of the other replicates (controls, drawn at random) are refitted with the same chain settings from a
    second start: the true parameters and the true latent states. For each replicate: the rank of the true mu*b in the
    second chain, and the split-R-hat of mu*b and of eps across the two starts. If the cold chain had reached the posterior,
    the two starts agree (R-hat near 1) and the flagged ranks mostly stay in the top tenth; if it had not, the starts
    disagree and the truth-started chain places the true mu*b in the body of its draws."""
    reps = main_run["_reps"]; L = main_run["L"]; top_cut = int(np.floor(9 * (L + 1) / 10))
    mb_ranks = np.asarray(main_run["ranks"]["mu_b"])
    flagged = [r for r in range(len(reps)) if mb_ranks[r] >= top_cut]
    rng = np.random.default_rng(seed)
    others = [r for r in range(len(reps)) if mb_ranks[r] < top_cut]
    controls = sorted(rng.choice(others, size=min(len(flagged), len(others)), replace=False).tolist())
    jobs = [(reps[r], main_run["n_iter"], main_run["burn"], main_run["thin"], True) for r in flagged + controls]
    fits = run_parallel(fi, mi, F, M, jobs, "-restart")
    out = dict(n_flagged=len(flagged), n_control=len(controls), rhat_flag=RHAT_FLAG, top_cut=top_cut, replicates=[])
    for f, r in zip(fits, flagged + controls):
        out["replicates"].append(dict(r=r, flagged=r in flagged, truth=reps[r]["truth"],
                                      rank_cold=f["cold"]["ranks"]["mu_b"], rank_truth_start=f["truth_start"]["ranks"]["mu_b"],
                                      rank_af_drug_cold=f["cold"]["ranks"]["af_drug"], rank_af_drug_truth_start=f["truth_start"]["ranks"]["af_drug"],
                                      rank_eps_cold=f["cold"]["ranks"]["eps"], rank_eps_truth_start=f["truth_start"]["ranks"]["eps"],
                                      post_mean_mu_b_cold=f["cold"]["post_mean_mu_b"], post_mean_mu_b_truth_start=f["truth_start"]["post_mean_mu_b"],
                                      post_mean_eps_cold=f["cold"]["post_mean_eps"], post_mean_eps_truth_start=f["truth_start"]["post_mean_eps"],
                                      rhat_between_starts_mu_b=f["rhat_between_starts_mu_b"], rhat_between_starts_eps=f["rhat_between_starts_eps"]))
    for grp in ("flagged", "control"):
        rows = [x for x in out["replicates"] if x["flagged"] == (grp == "flagged")]
        if not rows:
            out[grp] = dict(n=0); continue
        rh = np.array([x["rhat_between_starts_mu_b"] for x in rows]); rhe = np.array([x["rhat_between_starts_eps"] for x in rows])
        out[grp] = dict(n=len(rows),
                        n_cold_top=int(sum(x["rank_cold"] >= top_cut for x in rows)),
                        n_truth_start_top=int(sum(x["rank_truth_start"] >= top_cut for x in rows)),
                        n_truth_start_af_drug_top=int(sum(x["rank_af_drug_truth_start"] >= top_cut for x in rows)),
                        n_rhat_above=int(np.sum(rh > RHAT_FLAG)), rhat_median=float(np.median(rh)), rhat_max=float(rh.max()),
                        n_rhat_eps_above=int(np.sum(rhe > RHAT_FLAG)), rhat_eps_median=float(np.median(rhe)),
                        median_rank_cold=float(np.median([x["rank_cold"] for x in rows])),
                        median_rank_truth_start=float(np.median([x["rank_truth_start"] for x in rows])),
                        mean_true_mu_b=float(np.mean([x["truth"]["mu"] * x["truth"]["b"] for x in rows])),
                        mean_post_mu_b_cold=float(np.mean([x["post_mean_mu_b_cold"] for x in rows])),
                        mean_post_mu_b_truth_start=float(np.mean([x["post_mean_mu_b_truth_start"] for x in rows])),
                        mean_true_eps=float(np.mean([x["truth"]["eps"] for x in rows])),
                        mean_post_eps_cold=float(np.mean([x["post_mean_eps_cold"] for x in rows])),
                        mean_post_eps_truth_start=float(np.mean([x["post_mean_eps_truth_start"] for x in rows])))
    return out


def observed_high_start(y, fi, mi, F, M, n_iter, burn, thin, seed=SEED + 29):
    """On the observed data: one chain started with every drug shocked (W = 1 for every drug, mu = 0.99, b = 0.9, eps = 0.02)
    and one chain from an ordinary start, same length as the SBC main run; the posterior summaries of mu*b and AF_drug from
    each and the split-R-hat across the two starts."""
    rng = np.random.default_rng(seed)
    init_cold = {k: float(v) for k, v in zip(tt.PARAMS, rng.uniform(0.05, 0.95, 5))}
    d_cold = tt.gibbs(y, fi, mi, F, M, n_iter=n_iter, burn=burn, thin=thin, seed=seed, init=init_cold)
    z1 = (rng.random(F) < OBS_HIGH_START["phi"]).astype(int); w1 = np.ones(M, dtype=int)
    d_high = tt.gibbs(y, fi, mi, F, M, n_iter=n_iter, burn=burn, thin=thin, seed=seed + 1, init=dict(OBS_HIGH_START), init_latent=(z1, w1))
    out = dict(n_iter=n_iter, burn=burn, thin=thin, start=OBS_HIGH_START, init_cold=init_cold)
    for name, d in (("cold", d_cold), ("high", d_high)):
        post = derived({k: d[k] for k in tt.PARAMS})
        out[name] = {k: dict(median=float(np.median(post[k])), lo=float(np.percentile(post[k], 2.5)), hi=float(np.percentile(post[k], 97.5)))
                     for k in ("mu_b", "af_drug", "eps", "mu", "b")}
    pc = derived({k: d_cold[k] for k in tt.PARAMS}); ph = derived({k: d_high[k] for k in tt.PARAMS})
    out["rhat_between_starts"] = {k: tt.rhat([pc[k], ph[k]]) for k in ("mu_b", "af_drug", "eps", "mu", "b")}
    out["rhat_between_starts_max"] = float(max(out["rhat_between_starts"].values()))
    return out


def flagged_summary(out):
    """The prior draws whose true mu*b ranks in the top tenth of the posterior draws, against the other draws: how many,
    their mean true mu and b, the share with true mu above 0.9, and the share whose true eps ranks in the bottom tenth."""
    L = out["L"]; mb = np.asarray(out["ranks"]["mu_b"]); ep = np.asarray(out["ranks"]["eps"])
    flag = mb >= np.floor(9 * (L + 1) / 10)
    tr = {k: np.asarray(v) for k, v in out["truths"].items()}
    if flag.sum() == 0:
        return dict(n=0)
    return dict(n=int(flag.sum()), mu_mean=float(tr["mu"][flag].mean()), other_mu_mean=float(tr["mu"][~flag].mean()),
                b_mean=float(tr["b"][flag].mean()), other_b_mean=float(tr["b"][~flag].mean()),
                share_mu_above_0_9=float(np.mean(tr["mu"][flag] > 0.9)), other_share_mu_above_0_9=float(np.mean(tr["mu"][~flag] > 0.9)),
                share_eps_rank_bottom=float(np.mean(ep[flag] < np.ceil((L + 1) / 10))),
                eps_mean=float(tr["eps"][flag].mean()), other_eps_mean=float(tr["eps"][~flag].mean()))


def geweke_test(fi, mi, F, M, n_iter=GEWEKE_N_ITER, thin=GEWEKE_THIN, seed=SEED + 11):
    """Geweke (2004) joint-distribution test of the transition kernel on the observed graph: alternate one kernel sweep with
    a redraw of y from p(y | theta, Z, W). The stationary marginal of theta is the prior (Beta(1,1) on every parameter), so the
    thinned chain is compared with the prior: means (0.5 expected), Kolmogorov-Smirnov distance against Uniform(0,1) for the
    five parameters and against prior draws for the four derived quantities, each with an ESS-adjusted 5% critical value
    (1.36 / sqrt(ESS) one-sample; 1.36 * sqrt((ESS + n_prior) / (ESS * n_prior)) two-sample), and the shares of the chain above
    the prior 90th percentile and below the 10th (0.10 expected)."""
    rng = np.random.default_rng(seed)
    th = {k: float(rng.beta(1.0, 1.0)) for k in tt.PARAMS}
    z = (rng.random(F) < th["phi"]).astype(int); w = (rng.random(M) < th["mu"]).astype(int)
    y = (rng.random(len(fi)) < tt.cell_prob(th["eps"], th["a"], th["b"], z[fi], w[mi])).astype(float)
    kept = {k: [] for k in tt.PARAMS}
    for it in range(n_iter):
        z, w, _ = tt.gibbs_step(y, fi, mi, F, M, th, z, w, rng)
        y = (rng.random(len(fi)) < tt.cell_prob(th["eps"], th["a"], th["b"], z[fi], w[mi])).astype(float)
        if it % thin == 0:
            for k in tt.PARAMS: kept[k].append(th[k])
    kept = {k: np.asarray(v) for k, v in kept.items()}
    chain = derived(kept)
    prior_rng = np.random.default_rng(seed + 1)
    prior = derived({k: prior_rng.beta(1.0, 1.0, GEWEKE_PRIOR_N) for k in tt.PARAMS})
    out = dict(n_iter=n_iter, thin=thin, n_kept=int(len(kept["eps"])), n_prior=GEWEKE_PRIOR_N, results={})
    for k in QUANTITIES:
        x = np.asarray(chain[k]); e = ess(x)
        if k in tt.PARAMS:
            D = float(stats.kstest(x, "uniform").statistic); crit = 1.36 / np.sqrt(e)
        else:
            D = float(stats.ks_2samp(x, prior[k]).statistic); crit = 1.36 * np.sqrt((e + GEWEKE_PRIOR_N) / (e * GEWEKE_PRIOR_N))
        q10, q90 = np.percentile(prior[k], [10, 90])
        out["results"][k] = dict(mean=float(x.mean()), prior_mean=float(np.mean(prior[k])), ks_D=D, ks_crit_005=float(crit), ess=float(e),
                                 share_above_prior_q90=float(np.mean(x > q90)), share_below_prior_q10=float(np.mean(x < q10)))
    reg = (kept["mu"] > GEWEKE_REGION["mu_min"]) & (kept["eps"] < GEWEKE_REGION["eps_max"])
    pr = {k: prior_rng.beta(1.0, 1.0, GEWEKE_PRIOR_N) for k in ("mu", "eps", "b")}
    preg = (pr["mu"] > GEWEKE_REGION["mu_min"]) & (pr["eps"] < GEWEKE_REGION["eps_max"])
    out["region"] = dict(mu_min=GEWEKE_REGION["mu_min"], eps_max=GEWEKE_REGION["eps_max"], n=int(reg.sum()),
                         share=float(reg.mean()), prior_share=float(preg.mean()),
                         eps_mean=float(kept["eps"][reg].mean()), prior_eps_mean=float(pr["eps"][preg].mean()),
                         b_mean=float(kept["b"][reg].mean()), prior_b_mean=float(pr["b"][preg].mean()),
                         mu_b_mean=float((kept["mu"] * kept["b"])[reg].mean()), prior_mu_b_mean=float((pr["mu"] * pr["b"])[preg].mean()),
                         ess_b=float(ess(kept["b"][reg])) if reg.sum() > 10 else None)
    out["max_D_over_crit"] = float(max(v["ks_D"] / v["ks_crit_005"] for v in out["results"].values()))
    out["n_fail"] = int(sum(v["ks_D"] > v["ks_crit_005"] for v in out["results"].values()))
    out["min_ess"] = float(min(v["ess"] for v in out["results"].values()))
    return out


def restricted_exact_posterior(y, mi, M, G=RESTRICTED_GRID):
    """Drug-only submodel (a = phi = 0): exact posterior of (eps, b, mu) on a G^3 grid of cell midpoints under the uniform
    prior, from the marginal likelihood prod_m [(1 - mu) P(y_m | eps) + mu P(y_m | 1 - (1 - eps)(1 - b))]."""
    grid = (np.arange(G) + 0.5) / G
    eps = grid[:, None, None]; b = grid[None, :, None]; mu = grid[None, None, :]
    p1 = 1 - (1 - eps) * (1 - b)
    logp = np.zeros((G, G, G))
    for m in range(M):
        cells = np.where(mi == m)[0]; k = int(y[cells].sum()); n = len(cells)
        l0 = k * np.log(eps) + (n - k) * np.log1p(-eps); l1 = k * np.log(p1) + (n - k) * np.log1p(-p1)
        top = np.maximum(l0, l1)
        logp += top + np.log((1 - mu) * np.exp(l0 - top) + mu * np.exp(l1 - top))
    logp -= logp.max(); w = np.exp(logp); w /= w.sum()
    return grid, w


def restricted_one(args):
    """Worker: one replicate of the drug-only check (sampler with a, phi fixed at 0 against exact grid draws)."""
    r, truth, y, seed, n_iter, burn, thin = args
    fi, mi, F, M = _GRAPH["fi"], _GRAPH["mi"], _GRAPH["F"], _GRAPH["M"]
    rng = np.random.default_rng(seed + 1)
    with np.errstate(divide="ignore"):
        d = tt.gibbs(y, fi, mi, F, M, n_iter=n_iter, burn=burn, thin=thin, seed=seed,
                     init=dict(eps=float(rng.uniform(0.05, 0.95)), b=float(rng.uniform(0.05, 0.95)), mu=float(rng.uniform(0.05, 0.95))),
                     fix_theta=dict(a=0.0, phi=0.0))
    G = RESTRICTED_GRID; grid, w = restricted_exact_posterior(y, mi, M, G)
    L = len(d["eps"]); idx = rng.choice(G ** 3, size=L, p=w.ravel()); ie, ib, im = np.unravel_index(idx, (G, G, G))
    jit = lambda i: (i + rng.random(len(i))) / G
    e_x, b_x, m_x = jit(ie), jit(ib), jit(im)
    mb_s = d["mu"] * d["b"]; tv = truth["mu"] * truth["b"]
    return dict(r=r, L=L, rank_mu_b_sampler=int(np.sum(mb_s < tv)), rank_mu_b_exact=int(np.sum(m_x * b_x < tv)),
                rank_eps_sampler=int(np.sum(d["eps"] < truth["eps"])), rank_eps_exact=int(np.sum(e_x < truth["eps"])),
                mean_mu_b_sampler=float(mb_s.mean()), mean_mu_b_exact=float(np.sum(w * (grid[None, :, None] * grid[None, None, :]))),
                mean_eps_sampler=float(d["eps"].mean()), mean_eps_exact=float(np.sum(w * grid[:, None, None])),
                true_mu_b=tv, true_eps=truth["eps"])


def restricted_check(fi, mi, F, M, n_iter, burn, thin, n=N_RESTRICTED, seed=SEED + 41):
    """SBC of the drug-only submodel against its exact posterior: for n prior draws (a = phi = 0), the rank of the true mu*b
    and eps among the sampler's draws and among exact draws from the grid posterior, and the agreement of posterior means."""
    rng = np.random.default_rng(seed); jobs = []
    for r in range(n):
        truth = dict(eps=float(rng.beta(1.0, 1.0)), a=0.0, b=float(rng.beta(1.0, 1.0)), phi=0.0, mu=float(rng.beta(1.0, 1.0)))
        y, _, _ = tt.simulate(fi, mi, F, M, rng=rng, **truth)
        jobs.append((r, truth, y, seed + 7 * r, n_iter, burn, thin))
    _init_worker(fi, mi, F, M)
    if WORKERS <= 1:
        res = [restricted_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(fi, mi, F, M)) as pool:
            res = list(pool.map(restricted_one, jobs, chunksize=1))
    L = res[0]["L"]
    out = dict(n=n, grid=RESTRICTED_GRID, n_iter=n_iter, burn=burn, thin=thin, L=L, tests={}, top_bin={})
    for key in ("mu_b_sampler", "mu_b_exact", "eps_sampler", "eps_exact"):
        t = uniformity_test([x[f"rank_{key}"] for x in res], L); out["tests"][key] = t; out["top_bin"][key] = t["counts"][-1]
    diff_mb = np.array([x["mean_mu_b_sampler"] - x["mean_mu_b_exact"] for x in res]); diff_eps = np.array([x["mean_eps_sampler"] - x["mean_eps_exact"] for x in res])
    out["mean_abs_diff_mu_b"] = float(np.abs(diff_mb).mean()); out["max_abs_diff_mu_b"] = float(np.abs(diff_mb).max()); out["mean_diff_mu_b"] = float(diff_mb.mean())
    out["mean_abs_diff_eps"] = float(np.abs(diff_eps).mean()); out["max_abs_diff_eps"] = float(np.abs(diff_eps).max())
    out["corr_mean_mu_b"] = float(np.corrcoef([x["mean_mu_b_sampler"] for x in res], [x["mean_mu_b_exact"] for x in res])[0, 1])
    out["rank_corr_mu_b"] = float(np.corrcoef([x["rank_mu_b_sampler"] for x in res], [x["rank_mu_b_exact"] for x in res])[0, 1])
    out["expected_top_bin"] = out["tests"]["mu_b_sampler"]["expected"][-1]
    out["replicates"] = res
    return out


def batch_summary(run):
    """What is kept of each extra batch of the main run."""
    return dict(seed=run.get("seed"), n_sbc=run["n_sbc"], tests=run["tests"], top_bin=run["top_bin"], bottom_bin=run["bottom_bin"],
                coverage90=run["coverage90"], diagnostics=run["diagnostics"], per_replicate=run["per_replicate"], flagged=run["flagged"], ranks=run["ranks"],
                min_p=run["min_p"], n_p_below_005=run["n_p_below_005"], n_degenerate_datasets=run["n_degenerate_datasets"])


def pooled_summary(batches, L):
    """Rank-uniformity tests and coverage pooled over the batches."""
    ranks = {k: sum((b["ranks"][k] for b in batches), []) for k in QUANTITIES}
    n = len(ranks["eps"])
    out = dict(n=n, n_batches=len(batches), tests={k: uniformity_test(ranks[k], L) for k in QUANTITIES},
               coverage90={k: float(np.mean([b["coverage90"][k] * b["n_sbc"] for b in batches]) * len(batches) / n) for k in QUANTITIES})
    out["top_bin"] = {k: out["tests"][k]["counts"][-1] for k in QUANTITIES}; out["bottom_bin"] = {k: out["tests"][k]["counts"][0] for k in QUANTITIES}
    out["expected_top_bin"] = out["tests"]["eps"]["expected"][-1]; out["expected_per_bin"] = n / N_BINS
    out["min_p"] = float(min(t["p"] for t in out["tests"].values())); out["n_p_below_005"] = int(sum(t["p"] < 0.05 for t in out["tests"].values()))
    other = [k for k in QUANTITIES if k not in ("af_drug", "mu_b")]; out["min_p_other"] = float(min(out["tests"][k]["p"] for k in other))
    out["batch_top_mu_b"] = [b["top_bin"]["mu_b"] for b in batches]; out["batch_top_af_drug"] = [b["top_bin"]["af_drug"] for b in batches]
    out["batch_p_mu_b"] = [b["tests"]["mu_b"]["p"] for b in batches]; out["batch_p_af_drug"] = [b["tests"]["af_drug"]["p"] for b in batches]
    out["batch_min_p_other"] = [min(b["tests"][k]["p"] for k in other) for b in batches]
    return out


def main():
    cells, _, _, _ = load_cells()
    fi, mi, F, M, _, _ = index_graph(cells)
    y_obs = cells["unavailable"].values.astype(int)
    out = dict(geweke=geweke_test(fi, mi, F, M))
    print("geweke", json.dumps({k: (round(v["mean"], 3), round(v["ks_D"], 3), round(v["ks_crit_005"], 3), round(v["ess"])) for k, v in out["geweke"]["results"].items()}), flush=True)
    short = run_sbc(fi, mi, F, M, SHORT_N_ITER, SHORT_BURN, SHORT_THIN, label="-short"); short.pop("_reps")
    out["short"] = short
    print("short", json.dumps({k: (round(v["p"], 3), v["counts"]) for k, v in out["short"]["tests"].items()}), out["short"]["diagnostics"], flush=True)
    main_run = run_sbc(fi, mi, F, M, N_ITER, BURN, THIN, label="-main"); main_run["seed"] = SEED
    print("main", json.dumps({k: (round(v["p"], 3), v["counts"]) for k, v in main_run["tests"].items()}), main_run["diagnostics"], flush=True)
    out["restart"] = restart_check(fi, mi, F, M, main_run)
    print("restart", json.dumps({g: out["restart"][g] for g in ("flagged", "control")}), flush=True)
    out["observed_high_start"] = observed_high_start(y_obs, fi, mi, F, M, N_ITER, BURN, THIN)
    print("observed_high_start", json.dumps(out["observed_high_start"]["rhat_between_starts"]), json.dumps(out["observed_high_start"]["high"]["mu_b"]), json.dumps(out["observed_high_start"]["cold"]["mu_b"]), flush=True)
    main_run.pop("_reps")
    batches = [batch_summary(main_run)]
    for i, seed in enumerate(BATCH_SEEDS[1:N_BATCHES], start=2):
        b = run_sbc(fi, mi, F, M, N_ITER, BURN, THIN, seed=seed, label=f"-batch{i}"); b.pop("_reps"); b["seed"] = seed
        batches.append(batch_summary(b))
        print(f"batch{i}", json.dumps({k: (round(v["p"], 3), v["counts"]) for k, v in b["tests"].items()}), b["diagnostics"], flush=True)
    out["batches"] = batches; out["pooled"] = pooled_summary(batches, main_run["L"])
    print("pooled", json.dumps({k: (round(v["p"], 3), v["counts"]) for k, v in out["pooled"]["tests"].items()}), flush=True)
    out["restricted"] = restricted_check(fi, mi, F, M, N_ITER, BURN, THIN)
    print("restricted", json.dumps({k: (round(v["p"], 3), v["counts"]) for k, v in out["restricted"]["tests"].items()}),
          {k: out["restricted"][k] for k in ("mean_abs_diff_mu_b", "max_abs_diff_mu_b", "corr_mean_mu_b", "rank_corr_mu_b")}, flush=True)
    out.update(main_run)
    out["workers"] = WORKERS; out["n_batches"] = len(batches)
    with open(os.path.join(RESULTS, OUT_NAME), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
