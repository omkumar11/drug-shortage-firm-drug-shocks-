"""
outcome_variants.py -- the model-free statistics of Section 2.3 (dispersion permutation tests and sharing covariances
with their conditional permutation nulls) repeated for every outcome definition used in the paper, so that the
dependence of the firm-versus-drug pattern on the outcome is shown without the model. Writes results/outcome_variants.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, load_tbd, RESULTS  # noqa: E402
import analysis.descriptives as de  # noqa: E402

SEED = 20261016
N_PERM_DISP = 20000          # the same count as the Table 2 tests (descriptives.N_PERM), so the two tables agree up to Monte Carlo error
N_PERM_COV = 2000      # the same count as Table 2 (descriptives.py), so that the two tables differ only by random stream


def variants(cells):
    return {
        "main": cells["unavailable"],
        "unavailable_only": cells["unavailable_only"],
        "majority": (cells["n_unavailable"] / cells["n_presentations"] > 0.5).astype(int),
        "strict": cells["all_unavailable"],
        "partial_only": ((cells["unavailable"] == 1) & (cells["all_unavailable"] == 0)).astype(int),
        "allocation": cells["unavailable_alloc"],
        "note": cells["unavailable_note"],
        "wholesaler": cells["unavailable_wholesaler"],
        "no_discontinued": cells["unavailable_nodisc"],
    }


def size_class(n):
    """Cell-size strata used to hold the number of presentations fixed in the permutations."""
    return np.where(n <= 1, 1, np.where(n == 2, 2, 3))


def fully_out_check(cells, y, rng, n_perm=N_PERM_DISP, strata=None):
    """Model-free check related to Theorem 2: the number of multi-firm drugs whose listed firms are all unavailable,
    against the null that reshuffles outcomes within firms (firm-level independence across drugs), by number of
    listed firms (2, 3, 4+) and in total. With `strata` (e.g. cell-size classes) the reshuffling is restricted to
    cells of the same firm and stratum."""
    df = cells[["firm_unit", "drug"]].copy(); yv = np.asarray(y).astype(int)
    key = df["firm_unit"].astype(str) if strata is None else df["firm_unit"].astype(str) + "|" + pd.Series(strata, index=df.index).astype(str)
    codes = key.astype("category").cat.codes.values
    order = np.argsort(codes, kind="stable"); blocks = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    blocks = [b for b in blocks if len(b) > 1]
    dcode = df["drug"].astype("category").cat.codes.values; ndr = dcode.max() + 1
    kd = np.bincount(dcode, minlength=ndr)
    def counts(yy):
        u = np.bincount(dcode, weights=yy, minlength=ndr); full = (u == kd)
        return [int((full & (kd == 2)).sum()), int((full & (kd == 3)).sum()), int((full & (kd >= 4)).sum())]
    obs = counts(yv); null = np.empty((n_perm, 3))
    for i in range(n_perm):
        yp = yv.copy()
        for blk in blocks: yp[blk] = yv[rng.permutation(blk)]
        null[i] = counts(yp)
    tot_obs = sum(obs); tot_null = null.sum(axis=1)
    return dict(obs=obs, null_mean=[float(x) for x in null.mean(axis=0)], obs_total=tot_obs, null_mean_total=float(tot_null.mean()),
                p_total=float((np.sum(tot_null >= tot_obs) + 1) / (n_perm + 1)))


def extra_checks(cells, pres, rng):
    """(a) Main-outcome firm and drug dispersion with finer size strata (1, 2, 3-6, 7+ presentations).
    (b) Staleness: days since each presentation's last FDA update, its share of variance explained by firm, and its
        correlation with cell unavailability.
    (c) Demand spillover: across firms with unavailable presentations, the rank correlation between the share of them
        reported as Limited Availability and the share citing a demand increase."""
    out = {}
    n = cells["n_presentations"].values
    fine = np.where(n <= 1, 1, np.where(n == 2, 2, np.where(n <= 6, 3, 4)))
    df = cells[["firm_unit", "drug"]].copy(); df["unavailable"] = cells["unavailable"].values
    df["fx"] = df["firm_unit"].astype(str) + "|" + fine.astype(str); df["dx"] = df["drug"].astype(str) + "|" + fine.astype(str)
    out["main_firm_p_fine"] = de.permutation_test(df, "firm_unit", "dx", rng)["p_value"]
    out["main_drug_p_fine"] = de.permutation_test(df, "drug", "fx", rng)["p_value"]
    out["n_cells_fine"] = {str(c): int((fine == c).sum()) for c in (1, 2, 3, 4)}
    snap = pd.to_datetime(pres["update_date"], errors="coerce").max()
    days = (snap - pd.to_datetime(pres["update_date"], errors="coerce")).dt.days
    ok = days.notna()
    m = days[ok].groupby(pres.loc[ok, "firm_unit"]).transform("mean")
    out["update_days_median"] = float(days.median()); out["update_days_max"] = float(days.max())
    out["update_days_firm_r2"] = float(1 - ((days[ok] - m) ** 2).sum() / ((days[ok] - days[ok].mean()) ** 2).sum())
    cd = pres.assign(days=days).groupby(["firm_unit", "drug"])["days"].mean().reset_index().merge(cells[["firm_unit", "drug", "unavailable"]])
    out["update_days_corr_unavailable"] = float(np.corrcoef(cd["days"], cd["unavailable"])[0, 1])
    un = pres[pres["unavailable"] == 1]
    g = un.groupby("firm_unit").agg(n=("unavailable", "size"),
                                     lim=("availability", lambda s: float((s == "Limited Availability").mean())),
                                     dem=("shortage_reason", lambda s: float((s == "Demand increase for the drug").mean())))
    g = g[g["n"] >= 3]
    from scipy.stats import spearmanr
    rho, pv = spearmanr(g["lim"], g["dem"])
    out["spill_rho"] = float(rho); out["spill_p"] = float(pv); out["spill_n_firms"] = int(len(g))
    return out


def size_r2(cells):
    """Share of the variance of log cell size explained by drug means and by firm means (one-way decompositions)."""
    ls = np.log(cells["n_presentations"].values.astype(float)); tot = ((ls - ls.mean()) ** 2).sum()
    out = []
    for g in ("drug", "firm_unit"):
        m = cells.assign(ls=ls).groupby(g)["ls"].transform("mean").values
        out.append(float(1 - ((ls - m) ** 2).sum() / tot))
    return out


def model_free_row(cells, y, rng):
    """The Table 4 statistics for one outcome vector y on the cell table `cells` (dispersion ratios and permutation p-values
    for firms and drugs, sharing covariances with their conditional nulls, and the size-stratified versions)."""
    df = cells[["firm_unit", "drug"]].copy().reset_index(drop=True); df["unavailable"] = np.asarray(y).astype(int)
    r = dict(rate=float(df["unavailable"].mean()), n_cells=int(len(df)), n_unavailable=int(df["unavailable"].sum()))
    for grp, within, key in (("firm_unit", "drug", "firm"), ("drug", "firm_unit", "drug")):
        v, e, n = de.dispersion_stat(df, grp)
        pt = de.permutation_test(df, grp, within, rng)
        r[f"disp_{key}_ratio"] = v / e; r[f"perm_{key}_p"] = pt["p_value"]
    sc = de.sharing_covariances(df)
    r["cov_firm"] = sc["cov_shared_firm"]; r["cov_drug"] = sc["cov_shared_drug"]; r["cov_none"] = sc["cov_unrelated"]
    r["perm_cov_firm_p"] = de.permutation_cov_null(df, "drug", "cov_shared_firm", rng, N_PERM_COV)["p_value"]
    r["perm_cov_drug_p"] = de.permutation_cov_null(df, "firm_unit", "cov_shared_drug", rng, N_PERM_COV)["p_value"]
    # size-stratified versions: permute only among cells with the same number of presentations (1, 2, 3+)
    sz = size_class(np.asarray(cells["n_presentations"]))
    df["firm_x_size"] = df["firm_unit"].astype(str) + "|" + sz.astype(str)
    df["drug_x_size"] = df["drug"].astype(str) + "|" + sz.astype(str)
    for strat, key in (("drug_x_size", "firm"), ("firm_x_size", "drug")):       # cells alone in their stratum cannot move
        sizes = df.groupby(strat)["unavailable"].transform("size")
        r[f"permutable_cells_{key}_test"] = int((sizes > 1).sum()); r[f"singleton_strata_{key}_test"] = int((sizes == 1).sum())
    r["perm_firm_p_size"] = de.permutation_test(df, "firm_unit", "drug_x_size", rng)["p_value"]
    r["perm_drug_p_size"] = de.permutation_test(df, "drug", "firm_x_size", rng)["p_value"]
    r["perm_cov_firm_p_size"] = de.permutation_cov_null(df, "drug_x_size", "cov_shared_firm", rng, N_PERM_COV)["p_value"]
    r["perm_cov_drug_p_size"] = de.permutation_cov_null(df, "firm_x_size", "cov_shared_drug", rng, N_PERM_COV)["p_value"]
    co = de.conditional_co_unavailability(df)
    r["co_same_firm"] = co["same_firm"]["p_second_given_first"]; r["co_same_drug"] = co["same_drug"]["p_second_given_first"]; r["co_unrelated"] = co["unrelated"]["p_second_given_first"]
    return r


def cells_with_exits(cells, tbd):
    """The cell table with the 'To Be Discontinued' records of listed drugs added back as exits: a record makes its
    firm-drug cell unavailable (a firm leaving a presentation of a drug in shortage is treated like a discontinuation reported
    on a current record) and counts toward the cell's size; pairs that exist only through such records become new cells."""
    add = tbd.groupby(["firm_unit", "drug"]).size().rename("n_tbd").reset_index()
    out = cells[["firm_unit", "drug", "n_presentations", "unavailable", "injectable"]].merge(add, on=["firm_unit", "drug"], how="outer")
    out["n_tbd"] = out["n_tbd"].fillna(0).astype(int); out["n_presentations"] = out["n_presentations"].fillna(0).astype(int) + out["n_tbd"]
    out["unavailable"] = np.where(out["n_tbd"] > 0, 1, out["unavailable"].fillna(0)).astype(int)
    inj = tbd.groupby(["firm_unit", "drug"])["injectable"].max().rename("inj_tbd").reset_index()
    out = out.merge(inj, on=["firm_unit", "drug"], how="left"); out["injectable"] = out["injectable"].fillna(out["inj_tbd"]).fillna(0).astype(int)
    return out.drop(columns=["inj_tbd"]), int((add.shape[0])), int(len(out) - len(cells))


def main():
    rng = np.random.default_rng(SEED)
    cells, pres, _, _ = load_cells()
    de.N_PERM = N_PERM_DISP
    out = {}
    for name, y in variants(cells).items():
        out[name] = model_free_row(cells, y.values, rng)
        print(name, {k: round(v, 3) for k, v in out[name].items()})
    out["wholesaler"]["n_cells_changed"] = int((cells["unavailable_wholesaler"] != cells["unavailable"]).sum())
    # "discontinued presentations not listed" (editor, paper-v5 round 1): the loop above recodes discontinued presentations as
    # available, which keeps the cells made only of discontinued presentations as available cells. That version is kept under
    # its accurate name; the not-listed version drops those cells as well (separate random stream, so the stream of the
    # other rows is unchanged).
    out["no_discontinued_recoded"] = out.pop("no_discontinued")
    keep = cells["all_discontinued"].values == 0; sub = cells[keep].reset_index(drop=True)
    out["no_discontinued"] = model_free_row(sub, sub["unavailable_nodisc"].values, np.random.default_rng(SEED + 7))
    out["no_discontinued"]["n_cells_dropped"] = int((~keep).sum())
    print("no_discontinued (cells dropped)", {k: round(v, 3) for k, v in out["no_discontinued"].items()})
    # extended: presentations delisted from the NDC Directory also treated as discontinued (domain referee, paper-v5 round 3)
    keep = cells["all_discontinued_ext"].values == 0; sub = cells[keep].reset_index(drop=True)
    out["no_discontinued_ext"] = model_free_row(sub, sub["unavailable_nodisc_ext"].values, np.random.default_rng(SEED + 8))
    out["no_discontinued_ext"]["n_cells_dropped"] = int((~keep).sum())
    out["no_discontinued_ext"]["n_pres_delisted"] = int((pres["in_ndc_directory"] == 0).sum())
    out["no_discontinued_ext"]["n_pres_delisted_unavailable"] = int(((pres["in_ndc_directory"] == 0) & (pres["unavailable"] == 1)).sum())
    print("no_discontinued_ext", {k: round(v, 3) for k, v in out["no_discontinued_ext"].items()})
    # finer size strata (1, 2, 3-6, 7+ presentations) for the firm dispersion test of each of the five distinct definitions
    # (methods referee, paper-v4-om round 1); the main-outcome value is also in extra_checks()
    n = cells["n_presentations"].values; fine = np.where(n <= 1, 1, np.where(n == 2, 2, np.where(n <= 6, 3, 4)))
    for name in ("main", "unavailable_only", "majority", "strict", "partial_only"):
        df = cells[["firm_unit", "drug"]].copy(); df["unavailable"] = variants(cells)[name].values
        df["dx"] = df["drug"].astype(str) + "|" + fine.astype(str); df["fx"] = df["firm_unit"].astype(str) + "|" + fine.astype(str)
        out[name]["perm_firm_p_fine"] = de.permutation_test(df, "firm_unit", "dx", rng)["p_value"]
        out[name]["perm_drug_p_fine"] = de.permutation_test(df, "drug", "fx", rng)["p_value"]
    # the main outcome on the injectable cells only (the market the paper's motivation is about)
    inj = cells[cells["injectable"] == 1].reset_index(drop=True)
    out["injectable"] = model_free_row(inj, inj["unavailable"].values, rng)
    out["injectable"]["n_firms"] = int(inj["firm_unit"].nunique()); out["injectable"]["n_drugs"] = int(inj["drug"].nunique())
    print("injectable", {k: round(v, 3) for k, v in out["injectable"].items()})
    # the main and Unavailable-only outcomes with the firm defined as the marketer of record (normalised NDC labeler name,
    # all labeler codes of one company merged) and as the NDC labeler code: a contract manufacturer reporting for another
    # company's NDC is then counted under the company whose product it is
    for unit in ("marketer", "labeler"):
        cu, _, _, _ = load_cells(unit)
        for oname, col in (("main", "unavailable"), ("unavailable_only", "unavailable_only")):
            key = f"unit_{unit}" if oname == "main" else f"unit_{unit}_uo"
            out[key] = model_free_row(cu, cu[col].values, rng)
            out[key]["n_firms"] = int(cu["firm_unit"].nunique())
            print(key, {k: round(v, 3) for k, v in out[key].items()})
    # the main outcome with the To-Be-Discontinued records of listed drugs added back as exits
    tbd = load_tbd(); ext, n_pairs, n_new = cells_with_exits(cells, tbd)
    out["tbd_as_exit"] = model_free_row(ext, ext["unavailable"].values, rng)
    out["tbd_as_exit"].update(n_tbd_records=int(len(tbd)), n_tbd_firms=int(tbd["firm_unit"].nunique()), n_tbd_pairs=n_pairs, n_new_cells=n_new,
                              n_cells_turned_unavailable=int(((ext["n_tbd"] > 0)).sum()) - n_new - int(cells.merge(tbd[["firm_unit", "drug"]].drop_duplicates(), on=["firm_unit", "drug"])["unavailable"].sum()))
    print("tbd_as_exit", {k: round(v, 3) for k, v in out["tbd_as_exit"].items()})
    sz = size_class(cells["n_presentations"].values)
    # without the firms listed for the most drugs (domain referee, paper-v5 round 3): is the firm signal a property of the
    # three or five broad-line sterile-injectable manufacturers alone?
    nd = cells.groupby("firm_unit")["drug"].nunique().sort_values(ascending=False)
    for k in (3, 5):
        top = list(nd.index[:k]); sub = cells[~cells["firm_unit"].isin(top)].reset_index(drop=True)
        out[f"no_top{k}"] = model_free_row(sub, sub["unavailable"].values, np.random.default_rng(SEED + 10 + k))
        out[f"no_top{k}"].update(firms_dropped=top, n_cells_dropped=int(len(cells) - len(sub)))
        print(f"no_top{k}", {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in out[f"no_top{k}"].items()})
    out["fully_out_main"] = fully_out_check(cells, cells["unavailable"], rng)
    out["fully_out_strict"] = fully_out_check(cells, cells["all_unavailable"], rng)
    out["fully_out_main_size"] = fully_out_check(cells, cells["unavailable"], rng, strata=sz)
    out["fully_out_strict_size"] = fully_out_check(cells, cells["all_unavailable"], rng, strata=sz)
    print(out["fully_out_main"], out["fully_out_strict"], out["fully_out_main_size"], out["fully_out_strict_size"])
    out["extra"] = extra_checks(cells, pres, rng); print(out["extra"])
    out["size_share_drug_r2"], out["size_share_firm_r2"] = size_r2(cells)
    out["strict_rate_by_size"] = {str(c): float(cells["all_unavailable"][sz == c].mean()) for c in (1, 2, 3)}
    out["main_rate_by_size"] = {str(c): float(cells["unavailable"][sz == c].mean()) for c in (1, 2, 3)}
    out["n_cells_by_size"] = {str(c): int((sz == c).sum()) for c in (1, 2, 3)}
    out["n_perm_dispersion"] = N_PERM_DISP; out["n_perm_cov"] = N_PERM_COV
    with open(os.path.join(RESULTS, "outcome_variants.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
