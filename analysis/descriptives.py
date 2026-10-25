"""
descriptives.py -- model-free results (Section 3.1-3.2): sample description, within-drug and within-firm
dispersion, empirical sharing covariances (moment counterparts of eqs. (3)-(4)), permutation tests, timing.
Writes results/descriptives.json.
"""
import collections
import json
import os
import sys

import re
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, normalise_company, FIRM_UNIT, RESULTS  # noqa: E402

SEED = 20261014
N_PERM = 20000


def dispersion_stat(df, group):
    """Variance of the group-level unavailability shares (groups with >= 2 cells), and the binomial expectation
    under cell-level independence with the pooled rate."""
    g = df.groupby(group)["unavailable"].agg(["size", "sum"])
    g = g[g["size"] >= 2]
    share = g["sum"] / g["size"]
    p = df["unavailable"].mean()
    return float(share.var(ddof=1)), float(np.mean(p * (1 - p) / g["size"])), int(len(g))


def sharing_covariances(df):
    """Mean product-moment covariance over pairs of cells sharing a firm (different drugs) and pairs sharing a
    drug (different firms). Sample analogue of eqs. (3)-(4)."""
    y = df["unavailable"].values.astype(float); ybar = y.mean()
    f = df["firm_unit"].astype("category").cat.codes.to_numpy(); m = df["drug"].astype("category").cat.codes.to_numpy()
    n = len(df)
    iu = np.triu_indices(n, k=1)
    same_f = (f[:, None] == f[None, :])[iu]; same_m = (m[:, None] == m[None, :])[iu]
    prod = ((y - ybar)[:, None] * (y - ybar)[None, :])[iu]
    cov_f = prod[same_f & ~same_m].mean(); cov_m = prod[same_m & ~same_f].mean(); cov_0 = prod[~same_f & ~same_m].mean()
    return dict(cov_shared_firm=float(cov_f), n_pairs_firm=int((same_f & ~same_m).sum()),
                cov_shared_drug=float(cov_m), n_pairs_drug=int((same_m & ~same_f).sum()),
                cov_unrelated=float(cov_0), n_pairs_unrelated=int((~same_f & ~same_m).sum()), var_y=float(y.var()))


def ingredient_family(name):
    """Crude ingredient-family key: the FDA generic name without salts, strengths and dosage-form words, so that e.g. the
    dextrose strengths or the injection and capsule of one ingredient fall in one family."""
    d = str(name).lower()
    d = re.sub(r"\b(monohydrate|hydrochloride|sodium|phosphate|acetate|sulfate|citrate|bromide|bitartrate|dimesylate|tromethamine|hexacetonide|iodide|mesylate|succinate|tartrate|maleate|aspartate|saccharate|dextroamphetamine|amphetamine)\b", " ", d)
    d = re.sub(r"\d+(\.\d+)?%?", " ", d)
    d = re.sub(r"\b(injection|tablet|tablets|capsule|capsules|solution|suspension|oral|extended|release|chewable|film|for|irrigant|sterile|usp|powder|cream|ointment|emulsion|kit|concentrate|spray|ophthalmic|tc|m)\b", " ", d)
    d = re.sub(r"[^a-z ]", " ", d)
    d = " ".join(d.split())
    return d if d else str(name).lower()      # a name made only of salts and forms is its own family


def conditional_co_unavailability(df):
    """Buyer-facing re-expression of the sharing covariances: among ordered pairs of distinct cells in which the first cell
    is unavailable, the share whose second cell is also unavailable, for pairs that share a firm (different drugs), share a
    drug (different firms), or share neither. Under cell-level independence all three equal the overall cell rate."""
    y = df["unavailable"].values.astype(int)
    f = df["firm_unit"].astype("category").cat.codes.to_numpy(); m = df["drug"].astype("category").cat.codes.to_numpy()
    same_f = f[:, None] == f[None, :]; same_m = m[:, None] == m[None, :]
    np.fill_diagonal(same_f, False); np.fill_diagonal(same_m, False)
    off_diag = ~np.eye(len(y), dtype=bool)                      # a cell is never paired with itself, in any class
    first = y[:, None] == 1; both = (y[:, None] == 1) & (y[None, :] == 1)
    out = {}
    second = np.broadcast_to(y[None, :] == 1, same_f.shape)
    for key, mask in (("same_firm", same_f & ~same_m), ("same_drug", same_m & ~same_f), ("unrelated", ~same_f & ~same_m & off_diag)):
        n1 = int((first & mask).sum()); n2 = int((both & mask).sum()); n0 = int(mask.sum()); ns = int((second & mask).sum())
        # p_second: unconditional share of unavailable second cells in the same class of pairs (the class's own baseline)
        out[key] = dict(p_second_given_first=n2 / n1 if n1 else float("nan"), p_second=ns / n0 if n0 else float("nan"),
                        n_ordered_pairs_first_unavailable=n1, n_ordered_pairs=n0)
    out["cell_rate"] = float(y.mean())
    return out


def pigeonhole_bootstrap_cov(df, rng, n_boot=2000):
    """Two-way (firm x drug) 'pigeonhole' bootstrap: resample firms and drugs with replacement, keep the cells whose firm
    and drug are both selected (with multiplicity), recompute the sharing covariances and their ratio."""
    firms = df["firm_unit"].unique(); drugs = df["drug"].unique(); res = []
    for _ in range(n_boot):
        fw = pd.Series(rng.choice(firms, len(firms), replace=True)).value_counts()
        dw = pd.Series(rng.choice(drugs, len(drugs), replace=True)).value_counts()
        w = df["firm_unit"].map(fw).fillna(0).values * df["drug"].map(dw).fillna(0).values
        sub = df.loc[np.repeat(df.index.values, w.astype(int))]
        if sub["firm_unit"].nunique() < 2 or sub["drug"].nunique() < 2: continue
        sc = sharing_covariances(sub.reset_index(drop=True))
        res.append((sc["cov_shared_firm"], sc["cov_shared_drug"], sc["cov_unrelated"]))
    a = np.array(res)
    ratio = (a[:, 0] - a[:, 2]) / np.where(np.abs(a[:, 1] - a[:, 2]) < 1e-9, np.nan, (a[:, 1] - a[:, 2]))
    return dict(n_boot=int(len(a)), cov_firm_lo=float(np.percentile(a[:, 0], 2.5)), cov_firm_hi=float(np.percentile(a[:, 0], 97.5)),
                cov_drug_lo=float(np.percentile(a[:, 1], 2.5)), cov_drug_hi=float(np.percentile(a[:, 1], 97.5)),
                cov_none_lo=float(np.percentile(a[:, 2], 2.5)), cov_none_hi=float(np.percentile(a[:, 2], 97.5)),
                excess_firm_lo=float(np.percentile(a[:, 0] - a[:, 2], 2.5)), excess_firm_hi=float(np.percentile(a[:, 0] - a[:, 2], 97.5)),
                excess_drug_lo=float(np.percentile(a[:, 1] - a[:, 2], 2.5)), excess_drug_hi=float(np.percentile(a[:, 1] - a[:, 2], 97.5)),
                p_excess_drug_le_0=float(np.mean((a[:, 1] - a[:, 2]) <= 0)), p_firm_gt_drug=float(np.mean(a[:, 0] > a[:, 1])))


def permutation_cov_null(df, within, which, rng, n_perm=2000):
    """Conditional permutation null for a sharing covariance: permute outcomes within each `within` group and
    recompute `which` ('cov_shared_firm' or 'cov_shared_drug')."""
    obs = sharing_covariances(df)[which]
    y = df["unavailable"].values.copy(); codes = df[within].astype("category").cat.codes.values
    order = np.argsort(codes, kind="stable"); bounds = np.flatnonzero(np.diff(codes[order])) + 1; blocks = np.split(order, bounds)
    null = np.empty(n_perm); tmp = df.copy()
    for i in range(n_perm):
        yp = y.copy()
        for blk in blocks: yp[blk] = y[rng.permutation(blk)]
        tmp["unavailable"] = yp; null[i] = sharing_covariances(tmp)[which]
    return dict(observed=float(obs), null_mean=float(null.mean()), null_q95=float(np.quantile(null, 0.95)), p_value=float((np.sum(null >= obs) + 1) / (n_perm + 1)))


def permutation_test(df, group, within, rng):
    """Permute the outcome labels within each `within` group (preserving that group's number of unavailable
    cells) and recompute the dispersion statistic over `group`. One-sided p-value for excess dispersion."""
    obs, _, _ = dispersion_stat(df, group)
    y = df["unavailable"].values.copy()
    codes = df[within].astype("category").cat.codes.values
    order = np.argsort(codes, kind="stable"); bounds = np.flatnonzero(np.diff(codes[order])) + 1
    blocks = np.split(order, bounds)
    null = np.empty(N_PERM)
    tmp = df[[group]].copy()
    for i in range(N_PERM):
        yp = y.copy()
        for blk in blocks:
            yp[blk] = y[rng.permutation(blk)]
        tmp["unavailable"] = yp
        null[i], _, _ = dispersion_stat(tmp, group)
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return dict(observed=obs, null_mean=float(null.mean()), null_sd=float(null.sd() if hasattr(null, "sd") else null.std()),
                null_q95=float(np.quantile(null, 0.95)), p_value=p)


def firm_table(cells, min_drugs):
    """Rows for the firms listed for at least min_drugs drugs: listed drugs, unavailable cells, cells with an outright
    Unavailable presentation, and the reported reason type of the unavailable cells. Returns (rows, the five largest)."""
    g = cells.groupby("firm_unit")["unavailable"].agg(["size", "sum"]).sort_values(["size", "sum"], ascending=False)
    rows = [dict(firm=str(f), listed=int(r["size"]), unavailable=int(r["sum"])) for f, r in g[g["size"] >= min_drugs].iterrows()]
    uo = cells.groupby("firm_unit")["unavailable_only"].sum()
    rt = cells[cells["unavailable"] == 1].groupby("firm_unit")["reason_type"].value_counts().unstack(fill_value=0)
    for row in rows:
        f = row["firm"]; row["unavailable_only"] = int(uo.get(f, 0))
        for k in ("firm", "market", "other", "none"): row[f"reason_{k}"] = int(rt.loc[f, k]) if (f in rt.index and k in rt.columns) else 0
    return rows, rows[:5]


def main():
    rng = np.random.default_rng(SEED)
    cells, pres, drugs, firms = load_cells()
    out = dict(firm_unit=FIRM_UNIT)
    out["n_presentations"] = int(len(pres)); out["n_presentations_unavailable"] = int(pres["unavailable"].sum())
    out["n_cells"] = int(len(cells)); out["n_cells_unavailable"] = int(cells["unavailable"].sum())
    out["cell_rate"] = float(cells["unavailable"].mean())
    out["n_drugs"] = int(cells["drug"].nunique()); out["n_firms"] = int(cells["firm_unit"].nunique())
    out["n_labeler_codes"] = int(cells["firm"].nunique())
    out["n_drugs_injectable"] = int(drugs["injectable"].sum())
    out["share_presentations_injectable"] = float(pres["injectable"].mean())
    out["firms_with_2plus_drugs"] = int((cells.groupby("firm_unit").size() >= 2).sum())
    out["drugs_with_2plus_firms"] = int((cells.groupby("drug").size() >= 2).sum())
    out["drugs_single_firm"] = int((cells.groupby("drug").size() == 1).sum())
    out["median_firms_per_drug"] = float(cells.groupby("drug").size().median())
    out["max_firms_per_drug"] = int(cells.groupby("drug").size().max())
    out["median_drugs_per_firm"] = float(cells.groupby("firm_unit").size().median())
    top = cells.groupby("firm_unit").agg(n=("drug", "size"), k=("unavailable", "sum")).sort_values("n", ascending=False)
    out["top_firms"] = [dict(firm=i, n_drugs=int(r.n), n_unavailable=int(r.k)) for i, r in top.head(10).iterrows()]
    out["share_cells_top5_firms"] = float(top["n"].head(5).sum() / len(cells))
    # availability by reason (presentation level)
    out["reason_counts"] = {k: int(v) for k, v in pres["shortage_reason"].replace("", "(none)").value_counts().items()}
    out["reason_counts_unavailable"] = {k: int(v) for k, v in pres.loc[pres["unavailable"] == 1, "shortage_reason"].replace("", "(none)").value_counts().items()}
    out["reason_unavailable_rate"] = {k: float(v) for k, v in pres.groupby(pres["shortage_reason"].replace("", "(none)"))["unavailable"].mean().items()}
    # dispersion
    for grp, within, name in (("firm_unit", "drug", "firm"), ("drug", "firm_unit", "drug")):
        v, e, n = dispersion_stat(cells, grp)
        out[f"dispersion_{name}"] = dict(observed_var=v, binomial_var=e, ratio=v / e, n_groups=n)
        out[f"perm_{name}"] = permutation_test(cells, grp, within, rng)
    out["sharing_cov"] = sharing_covariances(cells)
    out["co_unavailability"] = conditional_co_unavailability(cells)
    # the same statistic without the three firms listed for the most drugs, which contribute most same-firm pairs
    sizes = cells.groupby("firm_unit")["drug"].size().sort_values(ascending=False)
    top3 = sizes.index[:3].tolist()
    out["co_unavailability_excl_top3"] = conditional_co_unavailability(cells[~cells["firm_unit"].isin(top3)])
    out["co_unavailability_excl_top3"]["firms"] = top3
    out["co_unavailability_excl_top3"]["share_same_firm_pairs_from_top3"] = float(sum(s * (s - 1) for s in sizes.iloc[:3]) / sum(s * (s - 1) for s in sizes))
    # firm-level tables: firms listed for at least five drugs, with their unavailable cells, by availability status and
    # reported reason type; once with the reporting company as the firm (supplementary) and once with the marketer of
    # record (normalised NDC labeler name; the purchaser-facing table of Section 3.2)
    out["firm_table_min_drugs"] = 5
    out["firm_table"], top = firm_table(cells, 5)
    cells_mk, _, _, _ = load_cells("marketer")
    out["firm_table_marketer"], top_mk = firm_table(cells_mk, 5)
    out["n_marketers"] = int(cells_mk["firm_unit"].nunique()); out["n_cells_marketer"] = int(len(cells_mk))
    # the five firms listed for the most drugs, each classified once by the most common reason type of its unavailable
    # cells (firm-type, drug-type, or Other/none; ties go to Other/none, then firm-type), so the three lists partition them
    for name, t in (("", top_mk), ("_company", top)):
        cls = {}
        for r in t:
            c = {"firm": r["reason_firm"], "market": r["reason_market"], "other": r["reason_other"] + r["reason_none"]}
            cls[r["firm"]] = max(("other", "firm", "market"), key=lambda k: (c[k], {"other": 2, "firm": 1, "market": 0}[k]))
        for k, nm in (("market", "market_majority"), ("firm", "firm_majority"), ("other", "other_majority")):
            out[f"firm_table_top5_{nm}{name}"] = [f for f, v in cls.items() if v == k]
    # co-unavailability under the Unavailable-only outcome and on the injectable cells (main outcome)
    duo = cells[["firm_unit", "drug"]].copy(); duo["unavailable"] = cells["unavailable_only"].values
    out["co_unavailability_unavailable_only"] = conditional_co_unavailability(duo)
    out["co_unavailability_injectable"] = conditional_co_unavailability(cells[cells["injectable"] == 1].reset_index(drop=True))
    # what the notes of Limited Availability presentations say
    lim = pres[pres["availability"].astype(str).str.strip().str.lower() == "limited availability"]; note = lim["related_info"].fillna("")
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    out["limited_notes"] = dict(n=int(len(lim)), allocation=int(note.str.contains("allocat|existing customer", case=False, regex=True).sum()),
                                date=int(note.str.contains(rf"(?:{months})\s+20\d\d|\b20\d\d\b|Q[1-4]", case=False, regex=True).sum()),
                                limited_supply=int(note.str.contains("limited supply", case=False).sum()), blank=int((note.str.strip() == "").sum()))
    # unavailable presentations whose note names a recovery or release month that had already passed at the export
    snap = pd.to_datetime(pres["update_date"]).max(); mlist = months.split("|")
    past = [f"{m} {y}" for y in range(2020, snap.year + 1) for mi, m in enumerate(mlist, start=1) if (y < snap.year) or (mi < snap.month)]
    un = pres[pres["unavailable"] == 1]["related_info"].fillna("")
    out["stale_notes"] = dict(n=int(un.str.contains("|".join(past), case=False, regex=True).sum()), snapshot=str(snap.date()))
    # package NDCs listed under more than one reporting company, and exact duplicates within a company
    g = pres.groupby("package_ndc").agg(n=("presentation", "size"), ncomp=("company_name", "nunique"))
    out["ndc_duplicates"] = dict(across_companies_ndcs=int((g["ncomp"] > 1).sum()), across_companies_presentations=int(g.loc[g["ncomp"] > 1, "n"].sum()),
                                 within_company_ndcs=int(((g["n"] > 1) & (g["ncomp"] == 1)).sum()), within_company_presentations=int(g.loc[(g["n"] > 1) & (g["ncomp"] == 1), "n"].sum()))
    # discontinued presentations (reported reason "Discontinuation of the manufacture of the drug")
    from analysis.common import load_tbd
    tbd = load_tbd()
    out["discontinued"] = dict(presentations=int(pres["discontinued"].sum()), by_reason=int(pres["discontinued_reason"].sum()),
                               by_note_only=int(((pres["discontinued_note"] == 1) & (pres["discontinued_reason"] == 0)).sum()),
                               tbd_records=int(len(tbd)), tbd_firms=int(tbd["firm_unit"].nunique()), tbd_drugs=int(tbd["drug"].nunique()),
                               presentations_unavailable=int(((pres["discontinued"] == 1) & (pres["unavailable"] == 1)).sum()),
                               cells_only_via_discontinued=int(((cells["unavailable"] == 1) & (cells["unavailable_nodisc"] == 0)).sum()),
                               cells_all_discontinued=int((cells["all_discontinued"] == 1).sum()),
                               cells_unavailable_nodisc=int(cells["unavailable_nodisc"].sum()))
    # unavailable presentations whose note names a different distributor, marketer or labeler than the reporting company
    pat = r"distribut|labeler|marketed by|marketer|on behalf"
    un_all = pres[pres["unavailable"] == 1].copy(); un_all["firm_name"] = un_all["firm_name"].fillna("")
    mk = un_all["related_info"].fillna("").str.contains(pat, case=False, regex=True)
    byl = un_all[mk & (un_all["firm_name"] != "")].groupby("firm_name").agg(n=("presentation", "size"), reporters=("company_name", lambda s: sorted(set(s))))
    # the labeler's own row as a reporting company: listed cells and unavailable cells (0 means the row hides the unavailability)
    byl["unit"] = [normalise_company(x) for x in byl.index]
    byl["unit_cells"] = [int((cells["firm_unit"] == u).sum()) for u in byl["unit"]]
    byl["unit_unavailable"] = [int(cells.loc[cells["firm_unit"] == u, "unavailable"].sum()) for u in byl["unit"]]
    byl["hidden"] = (byl["unit_cells"] > 0) & (byl["unit_unavailable"] == 0)       # listed as a reporting company with no unavailable cell
    byl = byl.sort_values(["hidden", "n"], ascending=[False, False])
    lab = str(byl.index[0]) if len(byl) else ""; unit = normalise_company(lab) if lab else ""
    own = pres[(pres["firm_name"].fillna("") == lab) & (pres["unavailable"] == 1) & ~pres["related_info"].fillna("").str.contains(pat, case=False, regex=True)]
    out["marketer_notes"] = dict(n=int(mk.sum()), n_reporting_companies=int(un_all.loc[mk, "company_norm"].nunique()), n_labelers=int(un_all.loc[mk & (un_all["firm_name"] != ""), "firm_name"].nunique()),
                                 top_labeler=lab, top_n=int(byl["n"].iloc[0]) if len(byl) else 0, top_reporters=list(byl["reporters"].iloc[0]) if len(byl) else [],
                                 top_labeler_own_unavailable=int(len(own)), top_labeler_unit=unit,
                                 n_labelers_with_no_unavailable_own_cell=int(byl["hidden"].sum()), n_labelers_listed_as_reporter=int((byl["unit_cells"] > 0).sum()),
                                 top_labeler_unit_cells=int((cells["firm_unit"] == unit).sum()), top_labeler_unit_unavailable_cells=int(cells.loc[cells["firm_unit"] == unit, "unavailable"].sum()))
    # same-firm cell pairs that lie within one ingredient family (several FDA generic names for one clinical product)
    fam = cells["drug"].map(ingredient_family); cf = cells.assign(fam=fam)
    multi = cf.groupby("fam")["drug"].nunique()
    within = {f: int(sum(c * (c - 1) // 2 for c in g["fam"].value_counts())) for f, g in cf.groupby("firm_unit")}
    total = int(sum(len(g) * (len(g) - 1) // 2 for _, g in cf.groupby("firm_unit")))
    out["ingredient_families"] = dict(families_with_multiple_names=int((multi > 1).sum()), names_in_such_families=int(multi[multi > 1].sum()),
                                      same_firm_pairs=total, same_firm_pairs_within_family=int(sum(within.values())), firms_with_within_family_pairs=int(sum(v > 0 for v in within.values())),
                                      families={k: sorted(v) for k, v in cf[cf["fam"].isin(multi[multi > 1].index)].groupby("fam")["drug"].unique().items()})
    # free-text reasons ("Shortage per Manufacturer: ...") carried by presentations whose reason field is Other or blank
    ri_all = pres["related_info"].fillna(""); pm = ri_all.str.contains("per manufacturer", case=False)
    other_blank = pres["shortage_reason"].fillna("").str.strip().isin(["", "Other"])
    out["per_manufacturer_notes"] = dict(n=int(pm.sum()), n_other_or_blank_reason=int((pm & other_blank).sum()),
                                         wholesaler_unavailable=int((ri_all.str.contains("wholesal", case=False) & (pres["unavailable"] == 1)).sum()))
    out["firms_no_recall_event_5y"] = int((cells.groupby("firm_unit")["firm_recall_events_5y"].max() == 0).sum())
    # the carboplatin example of the Introduction: listed firms and distinct initial posting dates
    carbo = cells[cells["drug"].str.contains("carboplatin", case=False)]
    pc = pres[pres["drug"].str.contains("carboplatin", case=False)]
    out["carboplatin"] = dict(n_firms=int(carbo["firm_unit"].nunique()), n_posting_dates=int(pc["initial_posting_date"].nunique()), posting_dates=sorted(pc["initial_posting_date"].astype(str).unique().tolist()))
    out["sharing_cov_boot"] = pigeonhole_bootstrap_cov(cells, rng)
    out["perm_cov_firm"] = permutation_cov_null(cells, "drug", "cov_shared_firm", rng)
    out["perm_cov_drug"] = permutation_cov_null(cells, "firm_unit", "cov_shared_drug", rng)
    top3 = cells.groupby("firm_unit").size().sort_values(ascending=False).index[:3]
    sc3 = sharing_covariances(cells[~cells["firm_unit"].isin(top3)].reset_index(drop=True))
    out["cov_excl_top3"] = dict(firms=list(top3), cov_shared_firm=sc3["cov_shared_firm"], n_pairs_firm=sc3["n_pairs_firm"])
    out["unavailable_without_reason"] = int(((pres["unavailable"] == 1) & (pres["shortage_reason"] == "")).sum())
    carb = pres[(pres["drug"] == "Carboplatin Injection") & (pres["unavailable"] == 1)]
    short = {"Discontinuation of the manufacture of the drug": "discontinuation", "Demand increase for the drug": "demand increase", "Delay in shipping of the drug": "shipping delay",
             "Requirements related to complying with good manufacturing practices": "good manufacturing practice", "Shortage of an active ingredient": "active ingredient shortage", "Other": "Other"}
    rc = carb.groupby("firm_unit")["shortage_reason"].agg(lambda s: sorted({short.get(x, x) for x in s if x}))
    cnt = collections.Counter(r for lst in rc for r in lst)
    out["carboplatin_reasons"] = "; ".join(f"{k} ({v} firm{'s' if v > 1 else ''})" for k, v in cnt.most_common())
    out["carboplatin_firms_unavailable"] = int(len(rc))
    out["carboplatin_gmp_firms"] = sorted(str(f) for f, lst in rc.items() if "good manufacturing practice" in lst)
    out["carboplatin_demand_firms"] = int(sum("demand increase" in lst for lst in rc))
    multi = cells[cells["unavailable"] == 1].groupby("drug")["reason_type"].agg(lambda s: len(set(s) - {"none"}))
    nd = (cells[cells["unavailable"] == 1].groupby("drug").size() >= 2)
    out["drugs_2plus_unavail_firms"] = int(nd.sum()); out["drugs_mixed_reason_types"] = int((multi[nd[nd].index] >= 2).sum())
    pdts = pd.to_datetime(pres["initial_posting_date"]); out["presentations_posted_2012_01_01"] = int((pdts == pd.Timestamp("2012-01-01")).sum())
    out["n_labeler_codes"] = int(pres["firm"].nunique())
    out["available_with_reason"] = int(((pres["availability"] == "Available") & (pres["shortage_reason"] != "")).sum())
    av = pres[pres["availability"] == "Available"]
    out["available_related_info"] = dict(blank=int((av["related_info"].fillna("") == "").sum()), wholesalers=int(av["related_info"].fillna("").str.lower().str.contains("wholesaler").sum()),
                                         allocation=int((av["allocation_note"] == 1).sum()), n=int(len(av)))
    ri = av["related_info"].fillna("")
    out["available_related_info"]["other"] = int(((ri != "") & ~ri.str.lower().str.contains("wholesaler") & (av["allocation_note"] != 1)).sum())
    out["cells_changed_by_allocation"] = int((cells["unavailable_alloc"] != cells["unavailable"]).sum())
    out["labeler_codes_multi_company"] = int((pres.groupby("firm")["company_norm"].nunique() > 1).sum())
    out["companies_multi_labeler"] = int((pres.groupby("company_norm")["firm"].nunique() > 1).sum())
    out["median_apps_in_market"] = float(drugs["n_apps_in_market"].median()); out["median_labelers_in_market"] = float(drugs["n_firms_in_market"].median())
    out["recall_pre_share_cells"] = float((cells["firm_recall_events_pre_posting"] > 0).mean())
    # share distribution within drugs
    g = cells.groupby("drug")["unavailable"].agg(["size", "sum"]); g = g[g["size"] >= 2]; sh = g["sum"] / g["size"]
    out["drug_share_hist"] = dict(all_available=int((sh == 0).sum()), partial=int(((sh > 0) & (sh < 1)).sum()), all_unavailable=int((sh == 1).sum()))
    # chi-square test of homogeneity of the unavailable share across drugs (>= 2 firms)
    tab = np.array([g["sum"], g["size"] - g["sum"]]).T
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    out["drug_homogeneity_chi2"] = dict(chi2=float(chi2), dof=int(dof), p=float(p))
    gf = cells.groupby("firm_unit")["unavailable"].agg(["size", "sum"]); gf = gf[gf["size"] >= 2]
    tab = np.array([gf["sum"], gf["size"] - gf["sum"]]).T
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    out["firm_homogeneity_chi2"] = dict(chi2=float(chi2), dof=int(dof), p=float(p))
    # covariates
    out["unavailable_by_injectable"] = {str(k): dict(rate=float(v["mean"]), n=int(v["size"])) for k, v in cells.groupby("injectable")["unavailable"].agg(["mean", "size"]).iterrows()}
    out["unavailable_by_controlled"] = {str(k): dict(rate=float(v["mean"]), n=int(v["size"])) for k, v in cells.groupby("controlled")["unavailable"].agg(["mean", "size"]).iterrows()}
    rec = cells["firm_recall_events_5y"] > 0
    out["unavailable_by_recall"] = {str(k): dict(rate=float(v["mean"]), n=int(v["size"])) for k, v in cells.groupby(rec)["unavailable"].agg(["mean", "size"]).iterrows()}
    # timing: within-drug spread of first posting dates across firms
    fp = pd.to_datetime(cells["first_posting"])
    spread = cells.assign(fp=fp).groupby("drug")["fp"].agg(lambda s: (s.max() - s.min()).days)
    spread = spread[cells.groupby("drug").size() >= 2]
    out["posting_spread"] = dict(n_drugs=int(len(spread)), share_zero=float((spread == 0).mean()),
                                 median_days_nonzero=float(spread[spread > 0].median()) if (spread > 0).any() else None,
                                 n_nonzero=int((spread > 0).sum()))
    pres_dates = pd.to_datetime(pres["initial_posting_date"])
    out["posting_year_counts"] = {str(k): int(v) for k, v in pres_dates.dt.year.value_counts().sort_index().items()}
    out["first_posting_min"] = str(pres_dates.min().date()); out["first_posting_max"] = str(pres_dates.max().date())
    out["share_presentations_posted_before_2021"] = float((pres_dates.dt.year < 2021).mean())
    # NDC-universe context for listed drugs
    out["median_applications_in_market"] = float(drugs["n_apps_in_market"].median())
    out["median_firms_listed"] = float(drugs["n_firms_listed"].median())
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "descriptives.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("n_cells", "n_cells_unavailable", "dispersion_firm", "dispersion_drug", "perm_firm", "perm_drug", "sharing_cov")}, indent=1))


if __name__ == "__main__":
    main()
