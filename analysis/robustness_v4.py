"""
robustness_v4.py -- model-free checks requested by the domain referee in review round 2 of paper-v4 (Section 3.2, Table 4).
(1) Schedule II drugs excluded: the oral stimulants on the list are under DEA production and procurement quotas, a regulatory
constraint that could cluster unavailability by firm without any firm shock. (2) Corporate groups as the firm unit in the
model-free tests (previously only in the model fits). (3) A conservative version of the unlisted-supplier check: the unlisted
holders' available cells are added only for firms already on the list, so that no firm enters as an all-available row.
(4) The share of multi-firm drugs with at least one available listed firm. Writes results/robustness_v4.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, RESULTS  # noqa: E402
from analysis.outcome_variants import model_free_row, SEED  # noqa: E402


def main():
    rng = np.random.default_rng(SEED + 91); out = {}
    cells, pres, _, _ = load_cells("company")
    # (1) Schedule II drugs excluded (DEA schedule of any presentation of the drug in the NDC Directory)
    cii = set(pres.loc[pres["dea_schedule"].fillna("").astype(str).str.upper().eq("CII"), "drug"])
    sub = cells[~cells["drug"].isin(cii)].reset_index(drop=True)
    out["no_schedule2"] = model_free_row(sub, sub["unavailable"].values, rng)
    out["no_schedule2"].update(n_drugs_dropped=len(cii), n_firms=int(sub["firm_unit"].nunique()))
    # (2) corporate groups as the firm unit
    cg, _, _, _ = load_cells("group")
    out["unit_group"] = model_free_row(cg, cg["unavailable"].values, rng); out["unit_group"]["n_firms"] = int(cg["firm_unit"].nunique())
    # (3) unlisted holders added only to firms already listed (marketer unit, as in unlisted_suppliers.py)
    mk, _, _, _ = load_cells("marketer")
    from analysis import unlisted_suppliers as us
    added = us.unlisted_cells(mk, load_cells("marketer")[1])
    keep = added[added["firm_unit"].isin(set(mk["firm_unit"]))]
    aug = pd.concat([mk[["firm_unit", "drug", "n_presentations", "unavailable"]], keep.assign(n_presentations=1, unavailable=0)], ignore_index=True)
    out["unlisted_existing_firms"] = model_free_row(aug, aug["unavailable"].values, rng)
    out["unlisted_existing_firms"]["n_added_cells"] = int(len(keep))
    # (3b) the same unlisted cells for every holder (new firms included), with outcomes imputed at the drug's listed
    #      unavailability rate instead of "available" (setting them all available would create all-available firm rows and
    #      inflate firm clustering); N_IMP imputations, summarised by the median and range of the size-stratified p-values
    N_IMP = 20; imp = []
    rate = mk.groupby("drug")["unavailable"].mean()
    for i in range(N_IMP):
        r_ = np.random.default_rng(SEED + 500 + i)
        yi = (r_.random(len(added)) < added["drug"].map(rate).values).astype(int)
        a2 = pd.concat([mk[["firm_unit", "drug", "n_presentations", "unavailable"]], added.assign(n_presentations=1, unavailable=yi)], ignore_index=True)
        row = model_free_row(a2, a2["unavailable"].values, r_)
        imp.append({k: row[k] for k in ("perm_firm_p_size", "perm_cov_firm_p_size", "perm_drug_p_size", "perm_cov_drug_p_size", "cov_firm", "cov_drug")})
    out["unlisted_imputed"] = {k: dict(median=float(np.median([x[k] for x in imp])), lo=float(min(x[k] for x in imp)), hi=float(max(x[k] for x in imp)))
                               for k in imp[0]}
    out["unlisted_imputed"]["n_imputations"] = N_IMP; out["unlisted_imputed"]["n_added_cells"] = int(len(added))
    # (3c) firm-weighted same-firm co-unavailability: for each firm with >= 2 cells and >= 1 unavailable, the share of its
    #      other cells unavailable given one unavailable cell, U(U-1)/(U(k-1)), averaged with equal weight per firm
    g = cells.groupby("firm_unit")["unavailable"].agg(["size", "sum"]); g = g[(g["size"] >= 2) & (g["sum"] >= 1)]
    def fw(df):
        gg = df.groupby("firm_unit")["unavailable"].agg(["size", "sum"]); gg = gg[(gg["size"] >= 2) & (gg["sum"] >= 1)]
        return float(((gg["sum"] - 1) / (gg["size"] - 1)).mean()), int(len(gg))
    out["co_same_firm_firm_weighted"], out["co_firm_weighted_n_firms"] = fw(cells)
    # its value under the within-drug permutation null (outcomes reshuffled among firms within each drug): conditioning on a
    # firm having an unavailable cell makes the statistic smaller than the cell rate even under independence
    base = cells[["firm_unit", "drug", "unavailable"]].reset_index(drop=True); nulls = []; r2 = np.random.default_rng(SEED + 700)
    for _ in range(2000):
        y = base["unavailable"].values.copy()
        for idx in base.groupby("drug").indices.values():
            y[idx] = y[idx][r2.permutation(len(idx))]
        nulls.append(fw(base.assign(unavailable=y))[0])
    nulls = np.array(nulls); obs = out["co_same_firm_firm_weighted"]
    out["co_firm_weighted_null_mean"] = float(nulls.mean()); out["co_firm_weighted_p"] = float((1 + (nulls >= obs).sum()) / (1 + len(nulls)))
    # (4) multi-firm drugs with at least one available listed firm
    g = cells.groupby("drug")["unavailable"].agg(["size", "sum"]); g = g[g["size"] >= 2]
    out["multi_firm_drugs"] = int(len(g)); out["multi_firm_drugs_some_available"] = int((g["sum"] < g["size"]).sum())
    for k in ("no_schedule2", "unit_group", "unlisted_existing_firms"):
        print(k, {kk: round(v, 3) for kk, v in out[k].items() if isinstance(v, (int, float))})
    print(out["multi_firm_drugs"], out["multi_firm_drugs_some_available"])
    with open(os.path.join(RESULTS, "robustness_v4.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
