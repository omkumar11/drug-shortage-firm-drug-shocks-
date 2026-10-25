"""Independent implementation of the two-tier posterior with the drug states W summed out exactly (the check the round-3
methods referee required). Given theta and Z, each drug m contributes
    L_m = (1 - mu) prod_{cells of m} Bern(y | pi(Z_f, 0)) + mu prod_{cells of m} Bern(y | pi(Z_f, 1)),
so a sampler over (Z, eps, a, b, mu, phi) with this marginal likelihood targets the same posterior as the blocked Gibbs
sampler of model/twotier.py but shares none of its latent-W machinery: Z_f by single-site Gibbs on the marginal
likelihood, (eps, a, b, mu) by a joint random-walk Metropolis step on the logit scale, phi by its conjugate Beta update.
It is run on the SBC replicates of the first batch whose true mu*b ranked in the top tenth (the flagged data sets), on
as many unflagged controls, and on the observed data; the ranks, posterior means and intervals of mu*b and AF_drug are
compared with the original sampler's (results/sbc.json, results/model_fit.json). Writes results/sbc_collapsed.json."""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from analysis.common import load_cells, RESULTS  # noqa: E402
from analysis.fit_model import index_graph, ess  # noqa: E402
from analysis import sbc  # noqa: E402
from model import twotier as tt  # noqa: E402

SEED = sbc.SEED + 97
N_ITER = int(os.environ.get("COLL_N_ITER", 60000)); BURN = int(os.environ.get("COLL_BURN", 10000)); THIN = int(os.environ.get("COLL_THIN", 250))
N_ITER_OBS = int(os.environ.get("COLL_N_ITER_OBS", 200000)); THIN_OBS = int(os.environ.get("COLL_THIN_OBS", 50))
WORKERS = int(os.environ.get("SBC_WORKERS", max(1, min(8, (os.cpu_count() or 2) - 2))))
STEP = 0.5


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def drug_loglik(y, z_cells, eps, a, b, mu, drug_cells):
    """log L_m for every drug: the exact sum over W_m of the cell likelihoods. drug_cells: list of index arrays."""
    base = (1.0 - eps) * (1.0 - a) ** z_cells
    p0 = 1.0 - base; p1 = 1.0 - base * (1.0 - b)
    with np.errstate(divide="ignore"):
        l0 = np.where(y == 1, np.log(p0), np.log1p(-p0)); l1 = np.where(y == 1, np.log(p1), np.log1p(-p1))
    out = np.empty(len(drug_cells))
    for m, idx in enumerate(drug_cells):
        s0 = l0[idx].sum(); s1 = l1[idx].sum(); hi = max(s0, s1)
        out[m] = hi + np.log((1.0 - mu) * np.exp(s0 - hi) + mu * np.exp(s1 - hi))
    return out


def marginal_loglik(y, fi, z, eps, a, b, mu, drug_cells):
    return float(drug_loglik(y, z[fi], eps, a, b, mu, drug_cells).sum())


def _colour_firms(fi, mi, F):
    """Greedy colouring of the firm graph (firms adjacent if they share a drug): firms of one colour can be updated at once."""
    drugs_of = [set(mi[fi == f]) for f in range(F)]
    colour = -np.ones(F, int)
    for f in np.argsort([-len(d) for d in drugs_of]):
        used = {colour[g] for g in range(F) if g != f and colour[g] >= 0 and drugs_of[f] & drugs_of[g]}
        c = 0
        while c in used: c += 1
        colour[f] = c
    return [np.flatnonzero(colour == c) for c in range(colour.max() + 1)]


def _cell_terms(y, zc, eps, a, b):
    """Per-cell log Bern(y | pi(z, w)) for w = 0 and w = 1."""
    base = (1.0 - eps) * (1.0 - a) ** zc
    p0 = 1.0 - base; p1 = 1.0 - base * (1.0 - b)
    with np.errstate(divide="ignore"):
        return np.where(y == 1, np.log(p0), np.log1p(-p0)), np.where(y == 1, np.log(p1), np.log1p(-p1))


def collapsed_gibbs(y, fi, mi, F, M, n_iter, burn, thin, seed, init=None, fix_theta=None, prior=(1.0, 1.0)):
    """Sampler over (Z, eps, a, b, mu, phi) with W summed out. Returns kept draws of the five parameters and of sum(Z).
    Per-drug sums S0[m], S1[m] of the cell terms under W_m = 0, 1 are maintained; firms of one colour class (no shared
    drug) are updated together, since each drug then has at most one cell whose term changes."""
    rng = np.random.default_rng(seed); y = np.asarray(y, int); fi = np.asarray(fi); mi = np.asarray(mi)
    classes = _colour_firms(fi, mi, F); cells_of_class = [np.flatnonzero(np.isin(fi, cl)) for cl in classes]
    th = dict(init) if init else {k: float(rng.uniform(0.05, 0.95)) for k in tt.PARAMS}
    if fix_theta: th.update(fix_theta)
    z = rng.integers(0, 2, F); a0, b0 = prior
    free = [k for k in ("eps", "a", "b", "mu") if not (fix_theta and k in fix_theta)]
    keep = {k: [] for k in tt.PARAMS}; keep["z_sum"] = []; acc = 0; n_theta = 0

    def drug_sums(zc, eps, a, b):
        l0, l1 = _cell_terms(y, zc, eps, a, b)
        return l0, l1, np.bincount(mi, weights=l0, minlength=M), np.bincount(mi, weights=l1, minlength=M)

    def total(S0, S1, mu):
        return float(np.logaddexp(np.log1p(-mu) + S0, np.log(mu) + S1).sum())

    c0, c1, S0, S1 = drug_sums(z[fi], th["eps"], th["a"], th["b"])
    for it in range(n_iter):
        eps, a, b, mu, phi = th["eps"], th["a"], th["b"], th["mu"], th["phi"]
        lmu0, lmu1 = np.log1p(-mu), np.log(mu)
        for cl, idx in zip(classes, cells_of_class):
            f_idx = fi[idx]; m_idx = mi[idx]; yc = y[idx]
            ll = np.empty((2, F))
            cand = {}
            for v in (0, 1):
                l0v, l1v = _cell_terms(yc, np.full(len(idx), v), eps, a, b)
                s0 = S0[m_idx] - c0[idx] + l0v; s1 = S1[m_idx] - c1[idx] + l1v
                ll[v] = np.bincount(f_idx, weights=np.logaddexp(lmu0 + s0, lmu1 + s1), minlength=F)
                cand[v] = (l0v, l1v)
            diff = (np.log(phi) + ll[1, cl]) - (np.log1p(-phi) + ll[0, cl])
            znew = (rng.uniform(size=len(cl)) < 1.0 / (1.0 + np.exp(-diff))).astype(int)
            z[cl] = znew
            zc_cells = z[f_idx]
            n0 = np.where(zc_cells == 1, cand[1][0], cand[0][0]); n1 = np.where(zc_cells == 1, cand[1][1], cand[0][1])
            np.add.at(S0, m_idx, n0 - c0[idx]); np.add.at(S1, m_idx, n1 - c1[idx]); c0[idx] = n0; c1[idx] = n1
        if not (fix_theta and "phi" in fix_theta):
            th["phi"] = float(rng.beta(a0 + z.sum(), b0 + F - z.sum()))
        if free:
            prop = dict(th); x = np.array([_logit(th[k]) for k in free]) + STEP * rng.standard_normal(len(free))
            for k, v in zip(free, _expit(x)): prop[k] = float(v)
            pc0, pc1, pS0, pS1 = drug_sums(z[fi], prop["eps"], prop["a"], prop["b"])
            lprior = lambda d: sum(a0 * np.log(d[k]) + b0 * np.log1p(-d[k]) for k in free)   # Beta(a0, b0) plus the logit Jacobian
            if np.log(rng.uniform()) < total(pS0, pS1, prop["mu"]) - total(S0, S1, th["mu"]) + lprior(prop) - lprior(th):
                th = prop; c0, c1, S0, S1 = pc0, pc1, pS0, pS1; acc += 1
            n_theta += 1
        if it >= burn and (it - burn) % thin == 0:
            for k in tt.PARAMS: keep[k].append(th[k])
            keep["z_sum"].append(int(z.sum()))
    out = {k: np.asarray(v) for k, v in keep.items()}; out["accept"] = acc / max(n_theta, 1)
    return out


def summarise(d, truth=None):
    post = sbc.derived({k: d[k] for k in tt.PARAMS}); L = len(d["eps"])
    out = dict(L=L, accept=float(d["accept"]), ess_mu_b=float(ess(post["mu_b"])), ess_af_drug=float(ess(post["af_drug"])), ess_eps=float(ess(d["eps"])),
               lag1_mu_b=float(np.corrcoef(post["mu_b"][:-1], post["mu_b"][1:])[0, 1]) if post["mu_b"].std() > 0 else 0.0)
    for k in ("mu_b", "af_drug", "eps", "b", "mu"):
        v = np.asarray(post[k]); lo, hi = np.percentile(v, [5, 95])
        out[k] = dict(mean=float(v.mean()), median=float(np.median(v)), lo90=float(lo), hi90=float(hi), lo95=float(np.percentile(v, 2.5)), hi95=float(np.percentile(v, 97.5)))
        if truth is not None:
            tv = sbc.derived(truth)[k]; out[k]["rank"] = sbc.rank_of(tv, v); out[k]["cover90"] = int(lo <= tv <= hi); out[k]["truth"] = float(tv)
    return out


_G = {}


def _init(fi, mi, F, M):
    _G.update(fi=fi, mi=mi, F=F, M=M)


def _one(args):
    rep, n_iter, burn, thin = args
    d = collapsed_gibbs(rep["y"], _G["fi"], _G["mi"], _G["F"], _G["M"], n_iter, burn, thin, seed=rep["seed"] + 101, init=rep["init"])
    return dict(r=rep["r"], **summarise(d, rep["truth"]))


def _one_obs(args):
    y, n_iter, burn, thin, seed, init = args
    return collapsed_gibbs(y, _G["fi"], _G["mi"], _G["F"], _G["M"], n_iter, burn, thin, seed=seed, init=init)


def main():
    t0 = time.time()
    cells, _, _, _ = load_cells(); fi, mi, F, M, _, _ = index_graph(cells); y_obs = cells["unavailable"].values.astype(int)
    prev = json.load(open(os.path.join(RESULTS, "sbc.json"))); rs = prev["restart"]
    b0 = prev["batches"][0]; n_sbc = len(b0["ranks"]["mu_b"]); top_cut = rs["top_cut"]
    L_prev = b0.get("L", prev["short"]["L"]); seed0 = b0.get("seed", sbc.SEED)
    reps = sbc.draw_replicates(fi, mi, F, M, n_sbc, seed0)
    byr = {x["r"]: x for x in rs["replicates"]}; idx = sorted(byr)
    with Pool(WORKERS, initializer=_init, initargs=(fi, mi, F, M)) as pool:
        fits = pool.map(_one, [(reps[r], N_ITER, BURN, THIN) for r in idx])
    rows = []
    for f in fits:
        o = byr[f["r"]]
        rows.append(dict(r=f["r"], flagged=bool(o["flagged"]), rank_prev=o["rank_cold"], rank_collapsed=f["mu_b"]["rank"],
                         rank_af_drug_prev=o["rank_af_drug_cold"], rank_af_drug_collapsed=f["af_drug"]["rank"],
                         post_mean_mu_b_prev=o["post_mean_mu_b_cold"], post_mean_mu_b_collapsed=f["mu_b"]["mean"],
                         post_mean_eps_prev=o["post_mean_eps_cold"], post_mean_eps_collapsed=f["eps"]["mean"],
                         ess_mu_b=f["ess_mu_b"], lag1_mu_b=f["lag1_mu_b"], accept=f["accept"], cover90_mu_b=f["mu_b"]["cover90"], L=f["L"]))
    out = dict(n_iter=N_ITER, burn=BURN, thin=THIN, L=fits[0]["L"], L_prev=L_prev, top_cut=top_cut, workers=WORKERS, replicates=rows)
    for grp in ("flagged", "control"):
        g = [x for x in rows if x["flagged"] == (grp == "flagged")]
        scale = out["L"] / L_prev
        d_mb = np.array([x["post_mean_mu_b_collapsed"] - x["post_mean_mu_b_prev"] for x in g])
        d_eps = np.array([x["post_mean_eps_collapsed"] - x["post_mean_eps_prev"] for x in g])
        rp = np.array([x["rank_prev"] * scale for x in g]); rc = np.array([x["rank_collapsed"] for x in g])
        out[grp] = dict(n=len(g), top_tenth_prev=int(sum(x["rank_prev"] >= top_cut for x in g)),
                        top_tenth_collapsed=int(sum(x["rank_collapsed"] >= round(0.9 * (out["L"] + 1)) for x in g)),
                        top_tenth_af_drug_collapsed=int(sum(x["rank_af_drug_collapsed"] >= round(0.9 * (out["L"] + 1)) for x in g)),
                        mean_abs_diff_mu_b=float(np.abs(d_mb).mean()), max_abs_diff_mu_b=float(np.abs(d_mb).max()), mean_diff_mu_b=float(d_mb.mean()),
                        mean_abs_diff_eps=float(np.abs(d_eps).mean()), max_abs_diff_eps=float(np.abs(d_eps).max()),
                        corr_post_mean_mu_b=float(np.corrcoef([x["post_mean_mu_b_prev"] for x in g], [x["post_mean_mu_b_collapsed"] for x in g])[0, 1]),
                        rank_corr_mu_b=float(np.corrcoef(rp, rc)[0, 1]), mean_abs_rank_diff=float(np.abs(rp - rc).mean()),
                        ess_mu_b_median=float(np.median([x["ess_mu_b"] for x in g])), ess_mu_b_min=float(np.min([x["ess_mu_b"] for x in g])),
                        lag1_mu_b_median=float(np.median([x["lag1_mu_b"] for x in g])), accept_median=float(np.median([x["accept"] for x in g])),
                        cover90_mu_b=int(sum(x["cover90_mu_b"] for x in g)))
    # the observed data: four collapsed chains from over-dispersed starts against the main fit
    rng = np.random.default_rng(SEED)
    inits = [{k: float(v) for k, v in zip(tt.PARAMS, rng.uniform(0.05, 0.95, 5))} for _ in range(4)]
    with Pool(min(WORKERS, 4), initializer=_init, initargs=(fi, mi, F, M)) as pool:
        chains = pool.map(_one_obs, [(y_obs, N_ITER_OBS, BURN, THIN_OBS, SEED + 11 * c, inits[c]) for c in range(4)])
    pooled = {k: np.concatenate([c[k] for c in chains]) for k in tt.PARAMS}; pooled["accept"] = float(np.mean([c["accept"] for c in chains]))
    obs = summarise(pooled); posts = [sbc.derived({k: c[k] for k in tt.PARAMS}) for c in chains]
    obs["rhat"] = {k: float(tt.rhat([p[k] for p in posts])) for k in ("mu_b", "af_drug", "eps", "b", "mu", "af_firm")}
    obs["ess_pooled"] = {k: float(sum(ess(p[k]) for p in posts)) for k in ("mu_b", "af_drug", "eps", "b", "mu")}
    mf = json.load(open(os.path.join(RESULTS, "model_fit.json")))["main"]
    obs["main_fit"] = {k: mf[k] for k in ("af_market", "b", "mu", "eps") if k in mf}
    out["observed"] = obs; out["n_iter_obs"] = N_ITER_OBS; out["thin_obs"] = THIN_OBS; out["n_chains_obs"] = 4; out["minutes"] = (time.time() - t0) / 60
    json.dump(out, open(os.path.join(RESULTS, "sbc_collapsed.json"), "w"), indent=1)
    print("flagged", json.dumps(out["flagged"])); print("control", json.dumps(out["control"]))
    print("observed", json.dumps({k: obs[k] for k in ("mu_b", "af_drug", "rhat", "ess_pooled", "main_fit")}))
    print(f"done in {out['minutes']:.1f} min")


if __name__ == "__main__":
    main()
