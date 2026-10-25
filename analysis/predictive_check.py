"""
predictive_check.py -- two checks asked for by the domain referee in review round 1 of paper-v4-om (Sections 3.2 and 4.1).

(1) Leave-one-cell-out prediction. For every firm-drug cell, predict its unavailability from the unavailability share of the
    firm's other listed cells, and separately from the share of the drug's other listed cells, and from both (a logistic
    regression on the two shares fitted leaving the cell out); report the area under the ROC curve (AUC, the Mann-Whitney
    statistic) and the accuracy at a 0.5 threshold, overall and for cells whose firm has at least three listed drugs, with a
    firm-cluster bootstrap for the AUCs. This is the purchaser's question (does a firm's record on other drugs predict this
    one?) and a direct firm-versus-drug comparison that does not depend on covariances.
(2) Demand-increase cells dropped. Cells in which any presentation reports "Demand increase for the drug" are removed and
    the model-free firm tests rerun, as a bound on the residual-supplier (demand-spillover) reading of the firm clustering.

Writes results/predictive.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, MARKET_REASONS, RESULTS  # noqa: E402
from analysis.outcome_variants import model_free_row, SEED  # noqa: E402

N_BOOT = 1000


def loo_shares(cells):
    """Leave-one-out unavailability shares of the firm's other cells and of the drug's other cells (NaN if none)."""
    d = cells[["firm_unit", "drug", "unavailable"]].copy(); d["log_pres"] = np.log(cells["n_presentations"].values)
    for key, nm in (("firm_unit", "firm"), ("drug", "drug")):
        g = d.groupby(key)["unavailable"]
        tot = g.transform("sum"); n = g.transform("size")
        d[f"{nm}_other"] = np.where(n > 1, (tot - d["unavailable"]) / (n - 1).clip(lower=1), np.nan)
        d[f"{nm}_n_other"] = n - 1
    return d


def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    if y.min() == y.max() or len(y) < 3:
        return float("nan")
    u = stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").statistic
    return float(u / ((y == 1).sum() * (y == 0).sum()))


def loo_logistic(d, cols):
    """Leave-one-cell-out logistic prediction from the given columns (plain Newton logistic regression, no package needed)."""
    X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in cols]); y = d["unavailable"].values.astype(float)
    k = X.shape[1]; pred = np.empty(len(d))
    for i in range(len(d)):
        m = np.ones(len(d), bool); m[i] = False
        b = np.zeros(k)
        for _ in range(25):
            p = 1 / (1 + np.exp(-X[m] @ b)); W = p * (1 - p)
            H = X[m].T @ (X[m] * W[:, None]) + 1e-6 * np.eye(k); g = X[m].T @ (y[m] - p)
            step = np.linalg.solve(H, g); b += step
            if np.abs(step).max() < 1e-8:
                break
        pred[i] = 1 / (1 + np.exp(-X[i] @ b))
    return pred


def summarise(d, rng):
    out = {}
    both = d.dropna(subset=["firm_other", "drug_other"]).copy()
    both["both_pred"] = loo_logistic(both, ["firm_other", "drug_other"])
    # cell size as the benchmark (log presentations is the strongest single covariate), and the nested leave-one-out models
    # size; size + drug share; size + drug share + firm share, so that the firm share's increment over the rest is explicit
    both["size_pred"] = loo_logistic(both, ["log_pres"])
    both["size_drug_pred"] = loo_logistic(both, ["log_pres", "drug_other"])
    both["size_drug_firm_pred"] = loo_logistic(both, ["log_pres", "drug_other", "firm_other"])
    for label, sub in (("all", both), ("firm_ge3", both[both["firm_n_other"] >= 2])):
        r = dict(n_cells=int(len(sub)), n_unavailable=int(sub["unavailable"].sum()), rate=float(sub["unavailable"].mean()),
                 n_firms=int(sub["firm_unit"].nunique()), n_drugs=int(sub["drug"].nunique()))
        for pred in ("firm_other", "drug_other", "both_pred", "size_pred", "size_drug_pred", "size_drug_firm_pred"):
            r[f"auc_{pred}"] = auc(sub[pred], sub["unavailable"])
            r[f"acc_{pred}"] = float(((sub[pred] >= 0.5).astype(int) == sub["unavailable"]).mean())
        r["acc_majority"] = float(max(sub["unavailable"].mean(), 1 - sub["unavailable"].mean()))
        # firm-cluster bootstrap of the AUCs and of their difference
        firms = sub["firm_unit"].unique(); groups = {f: g for f, g in sub.groupby("firm_unit")}; bf, bd, bb, diff, binc = [], [], [], [], []
        for _ in range(N_BOOT):
            pick = rng.choice(firms, len(firms), replace=True)
            sb = pd.concat([groups[f] for f in pick])
            af, ad, ab = auc(sb["firm_other"], sb["unavailable"]), auc(sb["drug_other"], sb["unavailable"]), auc(sb["both_pred"], sb["unavailable"])
            bf.append(af); bd.append(ad); bb.append(ab); diff.append(af - ad)
            binc.append(auc(sb["size_drug_firm_pred"], sb["unavailable"]) - auc(sb["size_drug_pred"], sb["unavailable"]))
        for nm, arr in (("firm", bf), ("drug", bd), ("both", bb), ("diff", diff), ("inc", binc)):
            arr = np.array(arr, float); arr = arr[np.isfinite(arr)]
            r[f"boot_{nm}_lo"], r[f"boot_{nm}_hi"] = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        diff = np.array(diff, float); diff = diff[np.isfinite(diff)]
        r["p_firm_gt_drug"] = float((diff > 0).mean()); r["n_boot"] = int(len(diff))
        binc = np.array(binc, float); binc = binc[np.isfinite(binc)]; r["p_firm_increment_gt0"] = float((binc > 0).mean())
        r["auc_increment_firm"] = r["auc_size_drug_firm_pred"] - r["auc_size_drug_pred"]
        out[label] = r
    return out


def main():
    rng = np.random.default_rng(SEED + 1700)
    cells, pres, _, _ = load_cells("company")
    out = {"loo": summarise(loo_shares(cells), rng)}
    # marketer unit as well
    mk, _, _, _ = load_cells("marketer")
    out["loo_marketer"] = summarise(loo_shares(mk), np.random.default_rng(SEED + 1701))
    # (2) cells with a demand-increase reason dropped
    dem = pres[pres["shortage_reason"].str.contains("Demand increase", case=False)].groupby(["firm_unit", "drug"]).size().reset_index()[["firm_unit", "drug"]]
    key = set(zip(dem["firm_unit"], dem["drug"]))
    sub = cells[[(f, m) not in key for f, m in zip(cells["firm_unit"], cells["drug"])]].reset_index(drop=True)
    out["no_demand_cells"] = model_free_row(sub, sub["unavailable"].values, rng)
    out["no_demand_cells"].update(n_cells_dropped=int(len(cells) - len(sub)), n_firms=int(sub["firm_unit"].nunique()),
                                  n_unavailable_dropped=int(cells.loc[[(f, m) in key for f, m in zip(cells["firm_unit"], cells["drug"])], "unavailable"].sum()))
    # discontinuation counts for Section 3.2 (domain referee, round 3): cells unavailable only through discontinued presentations,
    # cells whose every presentation is discontinued, and how many of the former still have an available presentation of the drug
    only_disc = (cells["unavailable"] == 1) & (cells["unavailable_nodisc"] == 0)
    out["discontinuation"] = dict(cells_unavailable=int(cells["unavailable"].sum()), cells_only_through_discontinued=int(only_disc.sum()),
                                  cells_wholly_discontinued=int((cells["all_discontinued"] == 1).sum()),
                                  only_disc_with_available_presentation=int((only_disc & (cells["n_unavailable"] < cells["n_presentations"])).sum()))
    with open(os.path.join(RESULTS, "predictive.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["loo"].items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)})
    print("no demand cells:", {k: round(out["no_demand_cells"][k], 3) for k in ("n_cells", "disp_firm_ratio", "perm_firm_p_size", "perm_cov_firm_p_size", "cov_firm", "cov_drug")}, out["no_demand_cells"]["n_cells_dropped"])


if __name__ == "__main__":
    main()
