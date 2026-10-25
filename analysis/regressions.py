"""
regressions.py -- covariate associations with cell unavailability (Section 3.4): logistic regressions with
standard errors clustered two ways (firm and drug) following Cameron, Gelbach and Miller (2011):
V = V_firm + V_drug - V_(firm x drug). Writes results/regressions.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, RESULTS  # noqa: E402


def twoway_cluster_logit(y, X, g1, g2):
    m = sm.Logit(y, X)
    r0 = m.fit(disp=0, maxiter=200)
    V1 = m.fit(disp=0, cov_type="cluster", cov_kwds=dict(groups=g1)).cov_params()
    V2 = m.fit(disp=0, cov_type="cluster", cov_kwds=dict(groups=g2)).cov_params()
    g12 = pd.factorize(pd.Series(g1).astype(str) + "|" + pd.Series(g2).astype(str))[0]
    V12 = m.fit(disp=0, cov_type="cluster", cov_kwds=dict(groups=g12)).cov_params()
    V = V1 + V2 - V12
    min_eig = float(np.linalg.eigvalsh(np.asarray(V)).min())
    if min_eig <= 0:
        raise RuntimeError(f"two-way cluster covariance not positive definite (min eigenvalue {min_eig:g})")
    se = np.sqrt(np.diag(V))
    return r0, se, min_eig


def main():
    cells, pres, drugs, firms = load_cells()
    d = cells.copy()
    d["log_apps"] = np.log(d["n_apps_in_market"].clip(lower=1))
    d["log_labelers"] = np.log(d["n_firms_in_market"].clip(lower=1))
    d["log_pres"] = np.log(d["n_presentations"])
    d["recall_pre_any"] = (d["firm_recall_events_pre_posting"] > 0).astype(int)
    d["log_listed"] = np.log(d.groupby("drug")["drug"].transform("size"))
    d["recall_any"] = (d["firm_recall_events_5y"] > 0).astype(int)
    d["log_recalls1"] = np.log1p(d["firm_recall_events_5y"])
    d["log_firm_markets"] = np.log(d["firm_n_markets"].clip(lower=1))
    d["market_age"] = (2026 - d["first_approval_year"])
    d["recall_ext_any"] = (d["recall_ext_events"] > 0).astype(int)
    d["recall_ext_other_any"] = (d["recall_ext_other_events"] > 0).astype(int)
    d = d.dropna(subset=["log_apps"])
    specs = {
        "m1_structure": ["injectable", "controlled", "log_apps", "log_pres"],
        "m2_firm": ["injectable", "controlled", "log_apps", "log_pres", "recall_any", "log_firm_markets"],
        "m3_firm_count": ["injectable", "controlled", "log_apps", "log_pres", "log_recalls1", "log_firm_markets"],
        "m5_recall_pre": ["injectable", "controlled", "log_apps", "log_pres", "recall_pre_any", "log_firm_markets"],
        "m6_labelers": ["injectable", "controlled", "log_labelers", "log_pres", "recall_any", "log_firm_markets"],
        "m7_listed": ["injectable", "controlled", "log_apps", "log_listed", "log_pres", "recall_any", "log_firm_markets"],
        "m8_nopres": ["injectable", "controlled", "log_apps", "recall_any", "log_firm_markets"],
        "m9_recall_ext": ["injectable", "controlled", "log_apps", "log_pres", "recall_ext_any", "log_firm_markets"],
        "m10_recall_other": ["injectable", "controlled", "log_apps", "log_pres", "recall_ext_other_any", "log_firm_markets"],
    }
    out = dict(n=int(len(d)), n_unavailable=int(d["unavailable"].sum()))
    for name, cols in specs.items():
        X = sm.add_constant(d[cols].astype(float)); y = d["unavailable"].astype(float)
        r, se, min_eig = twoway_cluster_logit(y, X, d["firm_unit"].values, d["drug"].values)
        res = {}
        for i, c in enumerate(X.columns):
            b = float(r.params.iloc[i]); s = float(se[i]); z = b / s
            res[c] = dict(coef=b, se=s, or_=float(np.exp(b)), or_lo=float(np.exp(b - 1.96 * s)), or_hi=float(np.exp(b + 1.96 * s)),
                          p=float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(z)))))
        out[name] = dict(coefs=res, llf=float(r.llf), n=int(r.nobs), pseudo_r2=float(r.prsquared), min_eigenvalue=min_eig,
                         n_firms=int(d["firm_unit"].nunique()), n_drugs=int(d["drug"].nunique()))
    # age model on the subset with Orange Book match
    da = d.dropna(subset=["market_age"])
    X = sm.add_constant(da[["injectable", "controlled", "log_apps", "log_pres", "recall_any", "market_age"]].astype(float))
    r, se, min_eig4 = twoway_cluster_logit(da["unavailable"].astype(float), X, da["firm_unit"].values, da["drug"].values)
    out["m4_age"] = dict(n=int(r.nobs), min_eigenvalue=min_eig4, coefs={c: dict(coef=float(r.params.iloc[i]), se=float(se[i]), or_=float(np.exp(r.params.iloc[i])),
                                                      or_lo=float(np.exp(r.params.iloc[i] - 1.96 * se[i])), or_hi=float(np.exp(r.params.iloc[i] + 1.96 * se[i])))
                                              for i, c in enumerate(X.columns)})
    # sensitivity: the competition count keys markets by ingredient set, dosage-form family and route, which merges the
    # dextrose strengths into one market; the firm model is refitted without the dextrose cells
    dx = d[~d["drug"].str.contains("Dextrose", case=False)]
    X = sm.add_constant(dx[specs["m2_firm"]].astype(float))
    r, se, min_eigx = twoway_cluster_logit(dx["unavailable"].astype(float), X, dx["firm_unit"].values, dx["drug"].values)
    out["m2_no_dextrose"] = dict(n=int(r.nobs), cells_dropped=int(len(d) - len(dx)), min_eigenvalue=min_eigx,
                                 coefs={c: dict(coef=float(r.params.iloc[i]), se=float(se[i]), or_=float(np.exp(r.params.iloc[i])),
                                                or_lo=float(np.exp(r.params.iloc[i] - 1.96 * se[i])), or_hi=float(np.exp(r.params.iloc[i] + 1.96 * se[i])))
                                        for i, c in enumerate(X.columns)})
    out["min_eigenvalue_all"] = float(min(v["min_eigenvalue"] for k, v in out.items() if isinstance(v, dict) and "min_eigenvalue" in v))
    out["recall_share_firms"] = float((firms["firm_recall_events_5y"] > 0).mean())
    out["recall_pre_share_cells"] = float(d["recall_pre_any"].mean())
    out["cells_left_censored"] = int(cells["left_censored"].sum())
    diff = d["recall_ext_any"] != d["recall_ext_other_any"]
    out["n_cells_recall_ext_vs_other_differ"] = int(diff.sum()); out["n_cells_recall_differ_unavailable"] = int(d.loc[diff, "unavailable"].sum())
    out["n_reference_firms_no_recall_ext"] = int(d.loc[d["recall_ext_any"] == 0, "firm_unit"].nunique())
    out["n_reference_firms_no_recall_other"] = int(d.loc[d["recall_ext_other_any"] == 0, "firm_unit"].nunique())
    out["n_reference_firms_no_recall"] = int(d.loc[d["recall_any"] == 0, "firm_unit"].nunique()); out["n_reference_cells_no_recall"] = int((d["recall_any"] == 0).sum())
    # firm-cluster bootstrap for the recall_any odds ratio in model 2 (resample firms with replacement)
    rng = np.random.default_rng(20261014); cols = specs["m2_firm"]; firms_u = d["firm_unit"].unique(); boots = []
    for _ in range(500):
        pick = rng.choice(firms_u, len(firms_u), replace=True)
        db = pd.concat([d[d["firm_unit"] == f] for f in pick])
        try:
            rb = sm.Logit(db["unavailable"].astype(float), sm.add_constant(db[cols].astype(float))).fit(disp=0, maxiter=200)
            boots.append(float(rb.params["recall_any"]))
        except Exception:
            pass
    boots = np.array(boots); boots = boots[np.isfinite(boots)]
    out["recall_any_firm_bootstrap"] = dict(n_boot=int(len(boots)), or_lo=float(np.exp(np.percentile(boots, 2.5))), or_hi=float(np.exp(np.percentile(boots, 97.5))))
    out["median_recall_events_firm"] = float(firms["firm_recall_events_5y"].median())
    with open(os.path.join(RESULTS, "regressions.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k in specs: print(k, {c: (round(v["or_"], 2), round(v["or_lo"], 2), round(v["or_hi"], 2)) for c, v in out[k]["coefs"].items()})
    print("m4", {c: (round(v["or_"], 2), round(v["or_lo"], 2), round(v["or_hi"], 2)) for c, v in out["m4_age"]["coefs"].items()})


if __name__ == "__main__":
    main()
