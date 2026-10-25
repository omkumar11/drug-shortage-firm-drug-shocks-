"""
panel.py -- three snapshots of the FDA shortage list (Section 3.6). The openFDA shortage export has no history, but the
Internet Archive holds two earlier copies of the same file (download.open.fda.gov/drug/shortages/...json.zip, captured
2025-06-29 and 2026-01-28; meta.last_updated 2025-06-28 and 2026-01-27; data/raw/archive/, logged in download_log.txt).
Together with the analysis snapshot (2026-10-13) they give three waves about seven and nine months apart.

Cells are built exactly as for the primary analysis (firm = normalised reporting company, drug = FDA generic name,
Current records only, a cell unavailable if any presentation is Unavailable or Limited Availability), from the shortage
records alone, so no NDC or Orange Book file of the earlier dates is needed; tests/test_panel.py checks that the
2026-10-13 wave reproduces the primary cell table.

(1) Replication: the model-free statistics of Table 4 (main outcome) on each earlier wave. The waves overlap heavily
    (most cells persist), so these are repeated measurements of a slowly changing list, not independent samples.
(2) Prospective association: for each transition (t -> t+1) and each cell listed at t+1, the firm's unavailable share
    among its OTHER drugs at t (fs) and the drug's unavailable share among its OTHER firms at t (ds). The primary
    estimand is onset: among cells listed and available at t, does fs predict unavailability at t+1, given ds and log
    cell size (logistic regression with a transition indicator, both transitions pooled)? Inference: firm-cluster
    bootstrap of the odds ratio and a one-sided firm-level permutation test (fs values permuted among the firms of each
    transition; H1: positive association). Persistence (cells unavailable at t) is reported the same way.
(3) Out-of-sample prediction: logistic models fitted on the first transition and applied unchanged to the second; AUC of
    the base model (own state at t, ds, log size) and of base + fs, with a firm-cluster bootstrap of the increment on the
    test transition.
(4) Firm persistence: Spearman correlation of firms' unavailable shares between consecutive waves (firms with >= 2 cells
    in both).
(5) Synchronised changes: for cells listed in both waves of a transition, the change in status (delta = y_t+1 - y_t in
    {-1, 0, 1}), and separately onset (cells available at t: unavailable at t+1) and recovery (cells unavailable at t:
    available at t+1). A firm-level shock moves a firm's cells together; a stable firm difference does not. Statistic: the
    firm-sharing covariance of the change over pairs of cells of the same firm and transition (different drugs), with the
    conditional null that permutes changes among the firms of each drug and transition (optionally also within cell-size
    strata), so drug-level changes are held fixed; and the drug-sharing covariance with the mirror null. Both transitions
    are pooled with firm and drug labels made transition-specific, so no pair spans two transitions.
(6) Persistence of the latent states (Proposition 3): on the cells listed in both waves of a pair, the same-wave
    firm-sharing covariance C_F(0) (mean of the two waves) and the cross-wave one C_F(lag) (pairs of cells of one firm in
    different drugs, one cell from each wave, symmetrised); their ratio estimates r_F, the correlation of the firm state
    across the interval. Likewise for drugs. Intervals: cluster bootstrap over firms (for r_F) and over drugs (for the
    drug covariances), resampled units relabelled so copies do not pair with each other.
Writes results/panel.json.
"""
import difflib
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import normalise_company, RESULTS, DISCONTINUED_NOTE_PATTERN  # noqa: E402
from analysis.outcome_variants import model_free_row, N_PERM_DISP, size_class  # noqa: E402
from analysis.predictive_check import auc  # noqa: E402
import analysis.descriptives as de  # noqa: E402
from pipeline.build_data import norm_avail, load_zip_json  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
WAVES = [("w1", os.path.join(RAW, "archive", "drug-shortages-wayback-20250629203306.json.zip")),
         ("w2", os.path.join(RAW, "archive", "drug-shortages-wayback-20260128115236.json.zip")),
         ("w3", os.path.join(RAW, "drug-shortages-0001-of-0001.json.zip"))]
SEED = 20261019
N_BOOT = 2000
N_PERM = 5000


# reporting companies renamed between snapshots (same company, new name): mapped to the name of the analysis export
RENAMES = {"ICU": "OTSUKA ICU", "PAR": "PAR HEALTH"}
# availability strings that occur only in the archived exports: a discontinuation is unavailable (as in the analysis
# export, where discontinued presentations are marked Unavailable); "pending information" and pointers to attached
# documents carry no status and the presentation is dropped
DISC_AVAIL = ("discontinued", "discontinuation")
# large-volume parenterals (intravenous fluids), whose shortages after Hurricane Helene's damage to Baxter's North Cove
# plant resolved during the first interval
IV_FLUID_PATTERN = r"sodium chloride|dextrose|lactated ringer|ringer's|sterile water|electrolyte|plasma-lyte"


def wave_presentations(path):
    """Current presentations of one shortage export with the primary firm unit, drug, outcome and discontinuation flag;
    the export's last_updated date; and the number of presentations dropped for an uninformative availability field."""
    z = zipfile.ZipFile(path); d = json.loads(z.read(z.namelist()[0]))
    rows = []; dropped = 0
    for r in d["results"]:
        if r.get("status") != "Current":
            continue
        raw = (r.get("availability") or "").strip().lower(); a = norm_avail(raw)
        if raw.startswith(DISC_AVAIL):
            a = "Unavailable"
        elif a == "Unknown":
            dropped += 1; continue
        firm = normalise_company((r.get("company_name") or "").strip()); firm = RENAMES.get(firm, firm)
        reason = r.get("shortage_reason") or ""; note = r.get("related_info") or ""
        disc = int(raw.startswith(DISC_AVAIL) or "discontinuation" in reason.lower()
                   or bool(re.search(DISCONTINUED_NOTE_PATTERN, note, flags=re.I)))
        prod = ["-".join(x.split("-")[:2]) for x in ((r.get("openfda") or {}).get("product_ndc") or [])]
        pk = (r.get("package_ndc") or "").split("-")
        if len(pk) >= 2: prod.append("-".join(pk[:2]))
        rows.append(dict(firm_unit=firm, labeler=(pk[0] if pk and pk[0] else firm), drug=r["generic_name"].strip(), unavailable=int(a in ("Unavailable", "Limited Availability")),
                         unavailable_only=int(a == "Unavailable"), discontinued=disc, product_ndcs=";".join(sorted(set(prod)))))
    return pd.DataFrame(rows), d["meta"].get("last_updated"), dropped


def wave_cells(path, nodisc=False, unit="company"):
    """Firm-drug cells of one shortage export (Current records), the export's last_updated date, and presentation counts.
    nodisc: discontinued presentations treated as not listed (cells made only of them dropped). unit="labeler": the NDC
    labeler code of the package NDC as the firm (robust to changes in the reporting company named on a record)."""
    p, date, _ = wave_presentations(path)
    if nodisc:
        p = p[p["discontinued"] == 0]
    if unit == "labeler":
        p = p.assign(firm_unit=p["labeler"])
    cells = (p.groupby(["firm_unit", "drug"], as_index=False)
             .agg(n_presentations=("unavailable", "size"), unavailable=("unavailable", "max"),
                  all_unavailable=("unavailable", "min"), unavailable_only=("unavailable_only", "max")))
    return cells, date, int(len(p)), int(p["unavailable"].sum())


def schedule2_drugs(paths):
    """Drugs (FDA generic names) with any presentation whose product NDC is Schedule II in the NDC Directory export."""
    ndc = load_zip_json("drug-ndc-0001-of-0001.json.zip")["results"]
    cii = {r["product_ndc"] for r in ndc if (r.get("dea_schedule") or "").upper() == "CII"}
    out = set()
    for path in paths:
        p, _, _ = wave_presentations(path)
        for drug, prods in zip(p["drug"], p["product_ndcs"]):
            if any(x in cii for x in prods.split(";") if x):
                out.add(drug)
    return out


def transition(A, B):
    """Cells of wave B with lagged predictors from wave A: own state at A (avail / unav / new), the firm's unavailable share
    among its other drugs at A (fs; NaN if none) and the drug's unavailable share among its other firms at A (ds)."""
    fa = A.groupby("firm_unit")["unavailable"].agg(["sum", "size"]); da = A.groupby("drug")["unavailable"].agg(["sum", "size"])
    M = B.merge(A[["firm_unit", "drug", "unavailable"]].rename(columns={"unavailable": "own_lag"}), on=["firm_unit", "drug"], how="left")
    listed = M["own_lag"].notna().astype(int); own = M["own_lag"].fillna(0)
    fn = M["firm_unit"].map(fa["size"]).fillna(0) - listed; fu = M["firm_unit"].map(fa["sum"]).fillna(0) - own
    dn = M["drug"].map(da["size"]).fillna(0) - listed; du = M["drug"].map(da["sum"]).fillna(0) - own
    M["f_other_n"] = fn.astype(int); M["d_other_n"] = dn.astype(int)
    M["f_sum_lag"] = M["firm_unit"].map(fa["sum"]).fillna(0).astype(int); M["f_size_lag"] = M["firm_unit"].map(fa["size"]).fillna(0).astype(int)
    M["fs"] = np.where(fn > 0, fu / fn.clip(lower=1), np.nan); M["ds"] = np.where(dn > 0, du / dn.clip(lower=1), np.nan)
    M["state"] = np.where(M["own_lag"].isna(), "new", np.where(M["own_lag"] == 1, "unav", "avail"))
    M["log_pres"] = np.log(M["n_presentations"].values)
    return M


def fit_logit(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]); step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def design(d, cols, trans_dummy=True):
    X = [np.ones(len(d))] + [d[c].values.astype(float) for c in cols]
    if trans_dummy:
        X.append((d["transition"] == "t2").values.astype(float))
    return np.column_stack(X)


def prospective(d, rng, label):
    """Logistic odds ratio of fs (0 -> 1) given ds and log size (pooled transitions), with firm-cluster bootstrap and a
    one-sided firm-level permutation test of fs."""
    d = d.dropna(subset=["fs", "ds"]).reset_index(drop=True)
    cols = ["fs", "ds", "log_pres"]; y = d["unavailable"].values.astype(float)
    b = fit_logit(design(d, cols), y)
    out = dict(n_cells=int(len(d)), n_unavailable=int(y.sum()), rate=float(y.mean()), n_firms=int(d["firm_unit"].nunique()),
               or_fs=float(np.exp(b[1])), or_ds=float(np.exp(b[2])))
    hi = d["fs"] > 0.5
    out.update(rate_fs_high=float(y[hi].mean()) if hi.any() else float("nan"), n_fs_high=int(hi.sum()),
               rate_fs_low=float(y[~hi].mean()) if (~hi).any() else float("nan"), n_fs_low=int((~hi).sum()))
    # firm-cluster bootstrap (a firm's cells in both transitions move together)
    groups = {f: g for f, g in d.groupby("firm_unit")}; firms = list(groups); bs = []
    for _ in range(N_BOOT):
        sb = pd.concat([groups[f] for f in rng.choice(firms, len(firms), replace=True)])
        if sb["unavailable"].nunique() < 2:
            continue
        bs.append(fit_logit(design(sb, cols), sb["unavailable"].values.astype(float))[1])
    bs = np.array(bs); bs = bs[np.isfinite(bs) & (np.abs(bs) < 50)]
    out["or_fs_lo"], out["or_fs_hi"] = float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))
    out["p_boot_or_gt1"] = float((bs > 0).mean())
    # firm-level permutation: within each transition, the firms' lagged totals (unavailable cells, listed cells at t) are
    # permuted among the firms of the set and fs is recomputed for every cell with its own drug left out. Used only on
    # cells listed at t (onset: own state 0; persistence: own state 1), for which every firm of the set has >= 2 cells at t
    # (and, for persistence, >= 1 unavailable), so the recomputed fs is always defined and in [0, 1].
    own = d["own_lag"].values.astype(float); null = np.empty(N_PERM)
    keys = [(idx, d.loc[idx].groupby("firm_unit")[["f_sum_lag", "f_size_lag"]].first()) for _, idx in d.groupby("transition").groups.items()]
    for i in range(N_PERM):
        fs = np.empty(len(d))
        for idx, tot in keys:
            perm = tot.iloc[rng.permutation(len(tot))].set_index(tot.index)
            s_ = d.loc[idx, "firm_unit"].map(perm["f_sum_lag"]).values; n_ = d.loc[idx, "firm_unit"].map(perm["f_size_lag"]).values
            fs[idx] = (s_ - own[idx]) / (n_ - 1)
        null[i] = fit_logit(design(d.assign(fs=fs), cols), y)[1]
    out["perm_p_one_sided"] = float((1 + (null >= b[1]).sum()) / (1 + N_PERM))
    out["n_perm"] = N_PERM; out["n_boot"] = int(len(bs))
    print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def out_of_sample(T1, T2, rng):
    """Fit on transition 1, predict transition 2 unchanged: base (own state, ds, log size) vs base + fs."""
    def prep(T):
        T = T.dropna(subset=["fs", "ds"]).reset_index(drop=True).copy()
        T["unav_lag"] = (T["state"] == "unav").astype(int); T["new_lag"] = (T["state"] == "new").astype(int)
        return T
    a, b = prep(T1), prep(T2); res = {}
    for name, cols in (("base", ["unav_lag", "new_lag", "ds", "log_pres"]), ("base_fs", ["unav_lag", "new_lag", "ds", "log_pres", "fs"])):
        coef = fit_logit(design(a, cols, False), a["unavailable"].values.astype(float))
        b[f"pred_{name}"] = 1 / (1 + np.exp(-design(b, cols, False) @ coef))
        res[f"auc_{name}"] = auc(b[f"pred_{name}"], b["unavailable"])
        on = b["state"] == "avail"
        res[f"auc_{name}_onset"] = auc(b.loc[on, f"pred_{name}"], b.loc[on, "unavailable"])
        res[f"coef_{name}"] = [float(x) for x in coef]
    res["auc_fs_alone"] = auc(b["fs"], b["unavailable"]); res["auc_fs_alone_onset"] = auc(b.loc[b["state"] == "avail", "fs"], b.loc[b["state"] == "avail", "unavailable"])
    res["auc_increment"] = res["auc_base_fs"] - res["auc_base"]; res["auc_increment_onset"] = res["auc_base_fs_onset"] - res["auc_base_onset"]
    groups = {f: g for f, g in b.groupby("firm_unit")}; firms = list(groups); inc, inc_on = [], []
    for _ in range(N_BOOT):
        sb = pd.concat([groups[f] for f in rng.choice(firms, len(firms), replace=True)])
        inc.append(auc(sb["pred_base_fs"], sb["unavailable"]) - auc(sb["pred_base"], sb["unavailable"]))
        so = sb[sb["state"] == "avail"]
        inc_on.append(auc(so["pred_base_fs"], so["unavailable"]) - auc(so["pred_base"], so["unavailable"]))
    for nm, arr in (("inc", inc), ("inc_onset", inc_on)):
        arr = np.array(arr, float); arr = arr[np.isfinite(arr)]
        res[f"boot_{nm}_lo"], res[f"boot_{nm}_hi"] = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        res[f"p_{nm}_gt0"] = float((arr > 0).mean())
    res.update(n_train=int(len(a)), n_test=int(len(b)), n_test_onset=int((b["state"] == "avail").sum()),
               n_test_onset_unavailable=int(b.loc[b["state"] == "avail", "unavailable"].sum()))
    print("out of sample", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not k.startswith("coef")})
    return res


def synchronised_changes(W, rng, n_perm=N_PERM):
    """Section (5) of the module docstring."""
    rows = []
    for t, (a, b) in (("t1", ("w1", "w2")), ("t2", ("w2", "w3"))):
        m = W[a].merge(W[b], on=["firm_unit", "drug"], suffixes=("_a", "_b")); m["transition"] = t; rows.append(m)
    D = pd.concat(rows, ignore_index=True)
    D["delta"] = D["unavailable_b"] - D["unavailable_a"]
    D["delta_c"] = D["delta"] - D.groupby("transition")["delta"].transform("mean")   # centred within each interval
    out = dict(n_cells=int(len(D)), n_onsets=int(((D["unavailable_a"] == 0) & (D["unavailable_b"] == 1)).sum()),
               n_recoveries=int(((D["unavailable_a"] == 1) & (D["unavailable_b"] == 0)).sum()),
               n_unchanged=int((D["delta"] == 0).sum()))
    for label, sub, val in (("change", D, D["delta_c"]), ("onset", D[D["unavailable_a"] == 0], D.loc[D["unavailable_a"] == 0, "unavailable_b"]),
                            ("recovery", D[D["unavailable_a"] == 1], 1 - D.loc[D["unavailable_a"] == 1, "unavailable_b"])):
        df = pd.DataFrame(dict(firm_unit=(sub["firm_unit"] + "|" + sub["transition"]).values, drug=(sub["drug"] + "|" + sub["transition"]).values,
                               unavailable=val.values.astype(float)))
        sz = size_class(sub["n_presentations_b"].values).astype(str)
        df["drug_x_size"] = df["drug"] + "|" + sz; df["firm_x_size"] = df["firm_unit"] + "|" + sz
        sc = de.sharing_covariances(df)
        pf = de.permutation_cov_null(df, "drug", "cov_shared_firm", rng, n_perm)
        pfs = de.permutation_cov_null(df, "drug_x_size", "cov_shared_firm", rng, n_perm)
        pdg = de.permutation_cov_null(df, "firm_unit", "cov_shared_drug", rng, n_perm)
        pds = de.permutation_cov_null(df, "firm_x_size", "cov_shared_drug", rng, n_perm)
        out[label] = dict(n_cells=int(len(df)), n_events=int(np.abs(df["unavailable"]).sum()),
                          cov_firm=sc["cov_shared_firm"], n_pairs_firm=sc["n_pairs_firm"], cov_drug=sc["cov_shared_drug"], n_pairs_drug=sc["n_pairs_drug"],
                          cov_firm_null_mean=pf["null_mean"], p_firm=pf["p_value"], p_firm_size=pfs["p_value"],
                          cov_drug_null_mean=pdg["null_mean"], p_drug=pdg["p_value"], p_drug_size=pds["p_value"])
        print("sync", label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[label].items()})
    return out


def interval_frame(W, how="inner", drop_drugs=()):
    """Cells of both intervals with their status at the two dates (columns unavailable_a, unavailable_b). how="inner": cells
    listed at both dates; how="outer": also cells that enter or leave the list, a cell that is not listed counting as
    available (entries are onsets, exits recoveries)."""
    rows = []
    for t, (a, b) in (("t1", ("w1", "w2")), ("t2", ("w2", "w3"))):
        m = W[a].merge(W[b], on=["firm_unit", "drug"], suffixes=("_a", "_b"), how=how)
        m["listed_a"] = m["unavailable_a"].notna().astype(int); m["listed_b"] = m["unavailable_b"].notna().astype(int)
        m["n_presentations_b"] = m["n_presentations_b"].fillna(m["n_presentations_a"])
        m[["unavailable_a", "unavailable_b"]] = m[["unavailable_a", "unavailable_b"]].fillna(0)
        m["transition"] = t; rows.append(m)
    D = pd.concat(rows, ignore_index=True)
    return D[~D["drug"].isin(set(drop_drugs))].reset_index(drop=True)


def change_test(D, rng, n_perm=N_PERM):
    """Firm- and drug-sharing covariances of the change y_b - y_a with their conditional permutation p-values (the
    statistic and nulls of synchronised_changes, computed with the vectorised helpers)."""
    fc = (D["firm_unit"] + "|" + D["transition"]).astype("category").cat.codes.values
    dc = (D["drug"] + "|" + D["transition"]).astype("category").cat.codes.values
    delta = (D["unavailable_b"] - D["unavailable_a"]).astype(float)
    delta = (delta - delta.groupby(D["transition"]).transform("mean")).values; PF, PD = _pairs(fc, dc)
    return dict(n_cells=int(len(D)), n_changes=int((D["unavailable_b"] != D["unavailable_a"]).sum()), cov_firm=_cov(delta, PF), p_firm=_perm_p(delta, PF, dc, rng, n_perm),
                cov_drug=_cov(delta, PD), p_drug=_perm_p(delta, PD, fc, rng, n_perm), n_pairs_firm=int(len(PF[0])), n_pairs_drug=int(len(PD[0])))


def transition_test(W, rng, n_perm=N_PERM, by_size=False, start=None):
    """Do transitions depend on the firm? On cells listed at both dates of each interval, the later status is permuted among
    cells of the same drug, interval and earlier status (so drug-level transitions and each cell's starting point are held
    fixed) and the cross-date and later-date same-firm covariances are recomputed. Under the null the firm plays no role in
    which cells change; firm clustering that survives the observed turnover makes both covariances exceed their null."""
    D = interval_frame(W)
    fc = (D["firm_unit"] + "|" + D["transition"]).astype("category").cat.codes.values
    dc = (D["drug"] + "|" + D["transition"]).astype("category").cat.codes.values
    PF, _ = _pairs(fc, dc); I, J = PF
    ya = D["unavailable_a"].values.astype(float); yb = D["unavailable_b"].values.astype(float)
    tr = D["transition"].values
    def centre(y):
        out = y.copy()
        for t in np.unique(tr): out[tr == t] -= y[tr == t].mean()
        return out
    ca = centre(ya)
    def stats_(y):
        cb = centre(y); return (0.5 * ((ca[I] * cb[J]).mean() + (cb[I] * ca[J]).mean()), (cb[I] * cb[J]).mean())
    key = D["drug"] + "|" + D["transition"] + "|" + D["unavailable_a"].astype(int).astype(str)
    if by_size:   # also within cell-size classes at the later date (1, 2, 3+ presentations), as in the cross-sectional tests
        key = key + "|" + pd.Series(size_class(D["n_presentations_b"].values), index=D.index).astype(str)
    if start is not None:   # permute only the cells with this earlier status (1: which cells stay unavailable; 0: which become
        # unavailable); every other cell keeps its observed later status (a block of its own), and the statistic uses all cells
        fixed = D["unavailable_a"].astype(int).values != start
        key = key.where(~fixed, "fixed|" + pd.Series(np.arange(len(D)), index=D.index).astype(str))
    obs = stats_(yb); block = key.astype("category").cat.codes.values
    order = np.lexsort((np.arange(len(yb)), block)); null = np.empty((n_perm, 2))
    for k in range(n_perm):
        new = np.lexsort((rng.random(len(yb)), block)); yp = np.empty_like(yb); yp[order] = yb[new]; null[k] = stats_(yp)
    return dict(n_cells=int(len(D)), cross_cov=float(obs[0]), cross_null_mean=float(null[:, 0].mean()), p_cross=float((1 + (null[:, 0] >= obs[0]).sum()) / (1 + n_perm)),
                later_cov=float(obs[1]), later_null_mean=float(null[:, 1].mean()), p_later=float((1 + (null[:, 1] >= obs[1]).sum()) / (1 + n_perm)),
                share_unavailable_stay=float(yb[ya == 1].mean()), share_available_stay=float(1 - yb[ya == 0].mean()))


def change_sensitivities(W, Wnd, cii, rng):
    """The change test of (5) by interval, with entries and exits, with discontinued presentations not listed, and without
    Schedule II drugs; and the persistence ratio without the three firms that supply most same-firm pairs."""
    out = {}
    D = interval_frame(W)
    for t in ("t1", "t2"):
        out[f"interval_{t}"] = change_test(D[D["transition"] == t].reset_index(drop=True), rng)
    E = interval_frame(W, how="outer")
    out["entries_exits"] = change_test(E, rng)
    out["entries_exits"].update(n_entries=int(((E["listed_a"] == 0)).sum()), n_exits=int(((E["listed_b"] == 0)).sum()),
                                n_entry_onsets=int(((E["listed_a"] == 0) & (E["unavailable_b"] == 1)).sum()),
                                n_exit_recoveries=int(((E["listed_b"] == 0) & (E["unavailable_a"] == 1)).sum()))
    out["no_discontinued"] = change_test(interval_frame(Wnd), rng)
    out["no_schedule2"] = change_test(interval_frame(W, drop_drugs=cii), rng); out["no_schedule2"]["n_drugs_dropped"] = int(len(cii))
    # the three firms with the most same-firm pairs among cells listed at both dates
    pairs = D.groupby(["transition", "firm_unit"]).size(); pairs = (pairs * (pairs - 1) / 2).groupby("firm_unit").sum().sort_values(ascending=False)
    top3 = list(pairs.index[:3]); out["top3_firms"] = top3; out["top3_pair_share"] = float(pairs.iloc[:3].sum() / pairs.sum())
    out["no_top3"] = change_test(D[~D["firm_unit"].isin(top3)].reset_index(drop=True), rng)
    for ab, (a, b) in (("w1_w2", ("w1", "w2")), ("w2_w3", ("w2", "w3"))):
        m = W[a].merge(W[b], on=["firm_unit", "drug"], suffixes=("_a", "_b")); m = m[~m["firm_unit"].isin(top3)].reset_index(drop=True)
        out[f"ratio_no_top3_{ab}"] = lagged_cov(m)["firm_ratio"]
    for k, v in out.items():
        if isinstance(v, dict):
            print("change sensitivity", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return out


def _pairs(fc, dc):
    """Index arrays of the unordered cell pairs that share the firm label but not the drug label, and conversely."""
    I, J = np.triu_indices(len(fc), 1)
    sf = fc[I] == fc[J]; sd = dc[I] == dc[J]
    return (I[sf & ~sd], J[sf & ~sd]), (I[sd & ~sf], J[sd & ~sf])


def _cov(y, IJ):
    m = y.mean(); return float(((y[IJ[0]] - m) * (y[IJ[1]] - m)).mean())


def _perm_p(y, IJ, block, rng, n_perm):
    """One-sided p-value of the pair covariance against permutation of y within blocks (vectorised; same statistic and
    null as descriptives.permutation_cov_null)."""
    obs = _cov(y, IJ); order = np.lexsort((np.arange(len(y)), block)); null = np.empty(n_perm)
    for k in range(n_perm):
        new = np.lexsort((rng.random(len(y)), block)); yp = np.empty_like(y); yp[order] = y[new]
        null[k] = _cov(yp, IJ)
    return float((1 + (null >= obs).sum()) / (1 + n_perm))


def dynamic_sim(W, fit, rng, n_sim=500, n_perm=500):
    """Size and power of the change tests and the sampling distribution of the persistence ratio (Proposition 3) under a
    dynamic version of the fitted model, on the observed cells of each interval (cells listed at both dates).
    Parameters: posterior medians of the main fit, except the firm effect a, which is calibrated so that the simulated
    same-date firm-sharing covariance of these cells matches the observed one (at the medians it is about twice as large). At the earlier date Z_f ~ Bern(phi), W_m ~ Bern(mu) and cells follow
    Equation (1). At the later date each firm keeps its state with probability r_F and otherwise redraws it from
    Bern(phi) (so Corr = r_F, stationary), drugs likewise with r_M; each cell keeps its earlier status with probability
    kappa and is otherwise redrawn from Equation (1) at the later date's states. kappa is set in each scenario so that the
    expected share of cells keeping their status matches the observed share. Scenarios: firm states persistent or
    renewed (r_F = 1, 0.9, 0.7, 0.5, 0), drug states renewed (r_M = 0), and both persistent (r_F = r_M = 1)."""
    eps, a, b, phi, mu = (fit[k]["median"] for k in ("eps", "a", "b", "phi", "mu"))
    a_median = a
    blocks = []
    for t, (wa, wb) in (("t1", ("w1", "w2")), ("t2", ("w2", "w3"))):
        m = W[wa].merge(W[wb], on=["firm_unit", "drug"], suffixes=("_a", "_b")); m["transition"] = t; blocks.append(m)
    D = pd.concat(blocks, ignore_index=True)
    fc = (D["firm_unit"] + "|" + D["transition"]).astype("category").cat.codes.values
    dc = (D["drug"] + "|" + D["transition"]).astype("category").cat.codes.values
    PF, PD = _pairs(fc, dc); nf, nd = fc.max() + 1, dc.max() + 1
    obs_keep = float((D["unavailable_a"] == D["unavailable_b"]).mean())
    # cross-date pairs for the persistence ratio, pooled over the two intervals
    I, J = PF

    def ratio(ya, yb):
        ca = ya - ya.mean(); cb = yb - yb.mean()
        c0 = 0.5 * ((ca[I] * ca[J]).mean() + (cb[I] * cb[J]).mean()); c1 = 0.5 * ((ca[I] * cb[J]).mean() + (cb[I] * ca[J]).mean())
        return c1 / c0 if c0 > 0 else np.nan

    def draw(r_F, r_M, kappa):
        Z0 = rng.random(nf) < phi; W0 = rng.random(nd) < mu
        Z1 = np.where(rng.random(nf) < r_F, Z0, rng.random(nf) < phi); W1 = np.where(rng.random(nd) < r_M, W0, rng.random(nd) < mu)
        p0 = 1 - (1 - eps) * (1 - a) ** Z0[fc] * (1 - b) ** W0[dc]; p1 = 1 - (1 - eps) * (1 - a) ** Z1[fc] * (1 - b) ** W1[dc]
        ya = (rng.random(len(fc)) < p0).astype(float); fresh = (rng.random(len(fc)) < p1).astype(float)
        yb = np.where(rng.random(len(fc)) < kappa, ya, fresh)
        return ya, yb

    def calibrate(r_F, r_M):
        lo, hi = 0.0, 0.99
        for _ in range(14):
            k = 0.5 * (lo + hi)
            keep = np.mean([np.mean(np.equal(*draw(r_F, r_M, k))) for _ in range(40)])
            lo, hi = (k, hi) if keep < obs_keep else (lo, k)
        return 0.5 * (lo + hi)

    obs_ratio = ratio(D["unavailable_a"].values.astype(float), D["unavailable_b"].values.astype(float))
    # calibration of the firm effect (methods referee, paper-v5 round 2): at the posterior medians the model implies about
    # twice the same-date firm-sharing covariance of these cells, so a is set (other parameters at their medians) to match
    # the observed same-date covariance, averaged over the two dates of each interval
    ya_o = D["unavailable_a"].values.astype(float); yb_o = D["unavailable_b"].values.astype(float)
    def same_date_cov(ya, yb):
        ca = ya - ya.mean(); cb = yb - yb.mean(); return 0.5 * ((ca[I] * ca[J]).mean() + (cb[I] * cb[J]).mean())
    obs_c0 = same_date_cov(ya_o, yb_o)
    def sim_c0(aa):
        nonlocal a
        a_old = a; a = aa
        v = np.mean([same_date_cov(*draw(1.0, 0.0, 0.0)) for _ in range(150)]); a = a_old; return v
    grid = np.linspace(0.05, min(0.95, a_median + 0.2), 19); vals = np.array([sim_c0(g) for g in grid])
    a = float(grid[np.argmin(np.abs(vals - obs_c0))]); c0_at_median = float(sim_c0(a_median))
    # onset statistics on the same cells: among cells available at the earlier date, the logistic odds ratio of the firm's
    # share of unavailable other cells at that date (drug share and log size held fixed), and the onset firm-sharing
    # covariance against the within-drug permutation null
    tr2 = (D["transition"] == "t2").values.astype(float); logn = np.log(D["n_presentations_b"].values.astype(float))
    fn = np.bincount(fc, minlength=nf); dn = np.bincount(dc, minlength=nd)

    def onset_stats(ya, yb, perm=True):
        fu = np.bincount(fc, weights=ya, minlength=nf); du = np.bincount(dc, weights=ya, minlength=nd)
        fo = fn[fc] - 1; do = dn[dc] - 1
        fs = np.where(fo > 0, (fu[fc] - ya) / np.maximum(fo, 1), np.nan); ds = np.where(do > 0, (du[dc] - ya) / np.maximum(do, 1), np.nan)
        m = (ya == 0) & np.isfinite(fs) & np.isfinite(ds)
        X = np.column_stack([np.ones(m.sum()), fs[m], ds[m], logn[m], tr2[m]]); y = yb[m]
        orv = float(np.exp(fit_logit(X, y)[1])) if 0 < y.sum() < len(y) else float("nan")
        p = float("nan")
        if perm:
            on = ya == 0; sub_f = fc[on]; sub_d = dc[on]; yo = yb[on].astype(float); PFo, _ = _pairs(sub_f, sub_d)
            p = _perm_p(yo, PFo, sub_d, rng, n_perm)
        return orv, p

    obs_or, obs_onset_p = onset_stats(D["unavailable_a"].values.astype(float), D["unavailable_b"].values.astype(float))
    out = dict(observed_keep_share=obs_keep, n_sim=n_sim, n_perm=n_perm, params=dict(eps=eps, a=a, b=b, phi=phi, mu=mu),
               a_median=float(a_median), a_calibrated=float(a), observed_same_date_cov=float(obs_c0), same_date_cov_at_median=c0_at_median,
               observed_ratio_pooled=float(obs_ratio), observed_onset_or=obs_or, observed_onset_firm_p=obs_onset_p)
    for name, r_F, r_M in (("firm_persistent", 1.0, 0.0), ("firm_09", 0.9, 0.0), ("firm_07", 0.7, 0.0), ("firm_half", 0.5, 0.0),
                           ("firm_renewed", 0.0, 0.0), ("both_persistent", 1.0, 1.0)):
        kappa = calibrate(r_F, r_M); rf = rd = ro = 0; ratios = []; ors = []
        for _ in range(n_sim):
            ya, yb = draw(r_F, r_M, kappa); delta = yb - ya; delta = delta - np.where(tr2 == 1, delta[tr2 == 1].mean(), delta[tr2 == 0].mean())
            o_or, o_p = onset_stats(ya, yb); ors.append(o_or); ro += o_p < 0.05
            rf += _perm_p(delta, PF, dc, rng, n_perm) < 0.05
            rd += _perm_p(delta, PD, fc, rng, n_perm) < 0.05
            ratios.append(ratio(ya, yb))
        ratios = np.array(ratios, float); ratios = ratios[np.isfinite(ratios)]; ors = np.array(ors, float); ors = ors[np.isfinite(ors)]
        out[name] = dict(r_F=r_F, r_M=r_M, kappa=float(kappa), reject_firm=rf / n_sim, reject_drug=rd / n_sim,
                         reject_onset_firm=ro / n_sim, onset_or_median=float(np.median(ors)), onset_or_lo=float(np.percentile(ors, 2.5)),
                         onset_or_hi=float(np.percentile(ors, 97.5)), share_onset_or_le_obs=float((ors <= obs_or).mean()),
                         ratio_mean=float(ratios.mean()), ratio_lo=float(np.percentile(ratios, 2.5)), ratio_hi=float(np.percentile(ratios, 97.5)),
                         share_ratio_ge_obs=float((ratios >= obs_ratio).mean()))
        print("dynamic sim", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()})
    return out


def lagged_cov(m):
    """Same-wave and cross-wave sharing covariances on cells observed in two waves (columns unavailable_a, unavailable_b),
    centred at each wave's mean (Proposition 3)."""
    ya = m["unavailable_a"].values - m["unavailable_a"].mean(); yb = m["unavailable_b"].values - m["unavailable_b"].mean()
    f = m["firm_unit"].astype("category").cat.codes.values; d = m["drug"].astype("category").cat.codes.values
    sf = f[:, None] == f[None, :]; sd = d[:, None] == d[None, :]; off = ~np.eye(len(m), dtype=bool)
    out = {}
    for nm, mask in (("firm", sf & ~sd), ("drug", sd & ~sf), ("none", ~sf & ~sd & off)):
        if not mask.any():
            out.update({f"{nm}_c0": float("nan"), f"{nm}_c1": float("nan")}); continue
        c0 = 0.5 * ((ya[:, None] * ya[None, :])[mask].mean() + (yb[:, None] * yb[None, :])[mask].mean())
        c1 = 0.5 * ((ya[:, None] * yb[None, :])[mask].mean() + (yb[:, None] * ya[None, :])[mask].mean())
        out[f"{nm}_c0"], out[f"{nm}_c1"] = float(c0), float(c1)
    ratio = lambda a, b: a / b if b != 0 else float("nan")
    out["firm_ratio"] = ratio(out["firm_c1"], out["firm_c0"]); out["drug_ratio"] = ratio(out["drug_c1"], out["drug_c0"])
    return out


def persistence(W, rng, n_boot=N_BOOT):
    """Section (6) of the module docstring."""
    res = {}
    for a, b in (("w1", "w2"), ("w2", "w3"), ("w1", "w3")):
        m = W[a].merge(W[b], on=["firm_unit", "drug"], suffixes=("_a", "_b")).reset_index(drop=True)
        r = lagged_cov(m); r["n_cells"] = int(len(m))
        for unit, keys in (("firm", ("firm_ratio", "firm_c0", "firm_c1")), ("drug", ("drug_c0", "drug_c1"))):
            col = "firm_unit" if unit == "firm" else "drug"
            groups = {u: g for u, g in m.groupby(col)}; units = list(groups); boot = {k: [] for k in keys}
            for _ in range(n_boot):
                pick = rng.choice(units, len(units), replace=True)
                sb = pd.concat([groups[u].assign(**{col: f"{u}#{i}"}) for i, u in enumerate(pick)], ignore_index=True)
                rb = lagged_cov(sb)
                for k in keys:
                    boot[k].append(rb[k])
            for k in keys:
                arr = np.array(boot[k], float); arr = arr[np.isfinite(arr)]
                r[f"{k}_lo"], r[f"{k}_hi"] = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
            if unit == "drug":
                arr = np.array(boot["drug_c1"], float); r["p_drug_c1_gt0"] = float((arr > 0).mean())
        res[f"{a}_{b}"] = r
        print("persistence", a, b, {k: round(v, 4) for k, v in r.items()})
    return res


def main():
    rng = np.random.default_rng(SEED); de.N_PERM = N_PERM_DISP
    W, out = {}, {"waves": {}}
    for k, path in WAVES:
        c, date, n_pres, n_unav = wave_cells(path); W[k] = c
        out["waves"][k] = dict(date=date, n_presentations=n_pres, n_unavailable_presentations=n_unav, n_cells=int(len(c)),
                               n_unavailable_cells=int(c["unavailable"].sum()), n_firms=int(c["firm_unit"].nunique()),
                               n_drugs=int(c["drug"].nunique()))
    # (1) replication of the model-free statistics on the earlier waves (w3 is Table 4 itself)
    for k in ("w1", "w2"):
        c = W[k]; r = model_free_row(c.assign(n_presentations=c["n_presentations"]), c["unavailable"].values, rng)
        out["waves"][k].update({f"mf_{kk}": v for kk, v in r.items()})
        print(k, {kk: round(v, 3) for kk, v in r.items() if isinstance(v, float)})
    # overlap between waves
    for a, b in (("w1", "w2"), ("w2", "w3"), ("w1", "w3")):
        A, B = W[a], W[b]; m = A.merge(B, on=["firm_unit", "drug"], suffixes=("_a", "_b"))
        out[f"overlap_{a}_{b}"] = dict(cells_common=int(len(m)), share_of_later=float(len(m) / len(B)),
                                       drugs_common=int(len(set(A["drug"]) & set(B["drug"]))), firms_common=int(len(set(A["firm_unit"]) & set(B["firm_unit"]))),
                                       status_agreement=float((m["unavailable_a"] == m["unavailable_b"]).mean()))
    # names that enter or leave between consecutive waves, and how many of the entering ones closely resemble a leaving one
    # (difflib ratio >= 0.85), which would indicate a spelling change rather than a real entry
    for key in ("drug", "firm_unit"):
        for a, b in (("w1", "w2"), ("w2", "w3")):
            A, B = set(W[a][key]), set(W[b][key]); new, gone = sorted(B - A), sorted(A - B)
            near = sum(1 for x in new if difflib.get_close_matches(x, gone, 1, 0.85))
            out[f"names_{key}_{a}_{b}"] = dict(entering=len(new), leaving=len(gone), near_matches=near)
    # cells of w1 or w3 not in w2's cells: new to each later wave (a different set of cells from the analysis snapshot)
    out["w3_cells_not_in_w1"] = int(len(W["w3"].merge(W["w1"][["firm_unit", "drug"]], how="left", indicator=True).query("_merge=='left_only'")))
    # (2) prospective association, both transitions pooled
    T1 = transition(W["w1"], W["w2"]).assign(transition="t1"); T2 = transition(W["w2"], W["w3"]).assign(transition="t2")
    T = pd.concat([T1, T2], ignore_index=True)
    out["state_counts"] = {t: {s: dict(n=int((g["state"] == s).sum()), unavailable=int(g.loc[g["state"] == s, "unavailable"].sum()))
                               for s in ("avail", "unav", "new")} for t, g in (("t1", T1), ("t2", T2))}
    out["onset"] = prospective(T[T["state"] == "avail"], rng, "onset")
    out["persistence"] = prospective(T[T["state"] == "unav"], rng, "persistence")
    out["onset_by_transition"] = {t: prospective(T[(T["state"] == "avail") & (T["transition"] == t)].assign(transition="t1"), rng, f"onset {t}")
                                  for t in ("t1", "t2")}
    # (3) out-of-sample
    out["out_of_sample"] = out_of_sample(T1, T2, rng)
    # (4) firm persistence of unavailable shares
    for a, b in (("w1", "w2"), ("w2", "w3")):
        fa = W[a].groupby("firm_unit")["unavailable"].agg(["mean", "size"]); fb = W[b].groupby("firm_unit")["unavailable"].agg(["mean", "size"])
        j = fa.join(fb, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j["size_a"] >= 2) & (j["size_b"] >= 2)]
        rho, p = stats.spearmanr(j["mean_a"], j["mean_b"])
        out[f"firm_persistence_{a}_{b}"] = dict(n_firms=int(len(j)), spearman=float(rho), p=float(p))
    # (5) synchronised changes
    out["sync"] = synchronised_changes(W, rng)
    fit = json.load(open(os.path.join(RESULTS, "model_fit.json")))["main"]
    out["dynamic_sim"] = dynamic_sim(W, fit, rng)
    # (6) persistence of the latent states (Proposition 3)
    out["persistence_states"] = persistence(W, rng)
    # sensitivities of the change test and of the ratio (domain referee, paper-v5 round 1), and the recurrence statistics with
    # discontinued presentations not listed
    Wnd = {k: wave_cells(p_, nodisc=True)[0] for k, p_ in WAVES}
    cii = schedule2_drugs([p_ for _, p_ in WAVES])
    out["change_sensitivities"] = change_sensitivities(W, Wnd, cii, rng)
    # do transitions depend on the firm? (domain referee, paper-v5 round 2), for the reporting company and the labeler code
    out["transition_test"] = transition_test(W, rng); print("transition test", out["transition_test"])
    out["transition_test_size"] = transition_test(W, rng, by_size=True); print("transition test, size strata", out["transition_test_size"])
    # fragility of the transition test (domain referee, paper-v5 round 3): each interval alone, without the three firms with most
    # same-firm pairs, with discontinued presentations not listed
    top3 = out["change_sensitivities"]["top3_firms"]
    out["transition_test_variants"] = dict(
        t1_only=transition_test({"w1": W["w1"], "w2": W["w2"], "w3": W["w2"].iloc[0:0]}, rng),
        t2_only=transition_test({"w1": W["w2"].iloc[0:0], "w2": W["w2"], "w3": W["w3"]}, rng),
        no_top3=transition_test({k: v[~v["firm_unit"].isin(top3)] for k, v in W.items()}, rng),
        no_discontinued=transition_test(Wnd, rng),
        start_unavailable=transition_test(W, rng, start=1), start_available=transition_test(W, rng, start=0))
    print("transition variants", {k: round(v["p_cross"], 3) for k, v in out["transition_test_variants"].items()})
    WL = {k: wave_cells(p_, unit="labeler")[0] for k, p_ in WAVES}
    out["labeler_unit"] = dict(transition_test=transition_test(WL, rng), transition_test_size=transition_test(WL, rng, by_size=True), change=change_test(interval_frame(WL), rng),
                               ratios={ab: lagged_cov(WL[a].merge(WL[b], on=["firm_unit", "drug"], suffixes=("_a", "_b")))["firm_ratio"]
                                       for ab, (a, b) in (("w1_w2", ("w1", "w2")), ("w2_w3", ("w2", "w3")), ("w1_w3", ("w1", "w3")))})
    print("labeler unit", out["labeler_unit"])
    # cells unavailable at all three dates, by firm; and the latest posting date of a Current record in each export (the first
    # interval contains no new listings: no Current record of the second or third export was posted between the two dates)
    allu = W["w1"].merge(W["w2"], on=["firm_unit", "drug"], suffixes=("_1", "_2")).merge(W["w3"], on=["firm_unit", "drug"])
    allu = allu[(allu["unavailable_1"] == 1) & (allu["unavailable_2"] == 1) & (allu["unavailable"] == 1)]
    out["always_unavailable"] = dict(n_cells=int(len(allu)), top_firms=allu["firm_unit"].value_counts().head(3).to_dict())
    for k, p_ in WAVES:
        z = zipfile.ZipFile(p_); recs = json.loads(z.read(z.namelist()[0]))["results"]
        dates = []
        for r in recs:
            if r.get("status") != "Current": continue
            for f in ("%m/%d/%Y", "%Y-%m-%d"):
                try: dates.append(pd.Timestamp(pd.to_datetime(r.get("initial_posting_date", ""), format=f))); break
                except Exception: pass
        out["waves"][k]["max_posting_date"] = str(max(dates).date())
    for k in ("w1", "w2"):
        c = Wnd[k]; r = model_free_row(c, c["unavailable"].values, rng)
        out["waves"][k].update({f"nd_{kk}": v for kk, v in r.items()})
    # presentations dropped for an uninformative availability field, and discontinuation-coded presentations, per wave
    for k, p_ in WAVES:
        pr, _, dropped = wave_presentations(p_)
        out["waves"][k].update(n_dropped_unknown=dropped, n_discontinued_pres=int(pr["discontinued"].sum()))
    # paper-v5 round-3 domain points, on a separate random stream so that every earlier number is unchanged: entries and
    # exits without the large-volume intravenous fluids (whose post-Hurricane Helene resolutions are among the exits), and the
    # panel tests with the Unavailable-only outcome
    rng2 = np.random.default_rng(SEED + 100)
    W_all_drugs = sorted(set().union(*[set(w["drug"]) for w in W.values()]))
    ivf = {d for d in W_all_drugs if re.search(IV_FLUID_PATTERN, d, flags=re.I)}
    E = interval_frame(W, how="outer", drop_drugs=ivf)
    out["entries_exits_no_iv_fluids"] = change_test(E, rng2); out["entries_exits_no_iv_fluids"]["n_drugs_dropped"] = int(len(ivf))
    out["entries_exits_no_iv_fluids"]["drugs_dropped"] = sorted(ivf)
    WU = {k: v.assign(unavailable=v["unavailable_only"]) for k, v in W.items()}
    out["unavailable_only"] = dict(transition_test=transition_test(WU, rng2), transition_test_size=transition_test(WU, rng2, by_size=True),
                                   change=change_test(interval_frame(WU), rng2),
                                   ratios={ab: lagged_cov(WU[a].merge(WU[b], on=["firm_unit", "drug"], suffixes=("_a", "_b")))["firm_ratio"]
                                           for ab, (a, b) in (("w1_w2", ("w1", "w2")), ("w2_w3", ("w2", "w3")), ("w1_w3", ("w1", "w3")))})
    out["no_top3_size"] = transition_test({k: v[~v["firm_unit"].isin(top3)] for k, v in W.items()}, rng2, by_size=True)
    print("round-3 additions", {k: out[k] for k in ("entries_exits_no_iv_fluids", "unavailable_only", "no_top3_size")})
    with open(os.path.join(RESULTS, "panel.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
