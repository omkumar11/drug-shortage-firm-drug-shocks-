"""
long_panel.py -- the FDA shortage list from October 2019 to October 2026, from the Internet Archive's captures of the FDA
Drug Shortage Database CSV download (analysis/fda_csv.py). Extends the three-snapshot analysis of panel.py to a panel at
roughly quarterly dates. Every analysis is exploratory.

Grid. The latest capture in each calendar quarter, dropping a date that lies less than 60 days before the next one;
quarters without a capture are skipped, so intervals are usually about three months and sometimes much longer (their
lengths are recorded). Drugs are linked across the FDA's renamings (fda_csv.canonical_drugs); firms are the normalised
reporting company with the renamings of fda_csv.RENAMES. In early 2024 the agency changed the availability field from
free text to categorical values; intervals in which more than 15% of the common cells change from free-text to
categorical coding are "migration intervals" and are excluded from the interval tests (and included in a sensitivity).

(0) Validation against the three openFDA exports; agreement of the keyword coding with the firm's own categorical value
    for presentations coded both ways in consecutive captures across the migration; composition and listing activity.
(1) Recurrence: the firm and drug dispersion ratios with size-stratified conditional permutation tests at each grid date.
(2) Transitions (panel.transition_test on any number of intervals): later status permuted within drug x interval x
    earlier status (optionally x size class); split by starting status. Variants: periods, without the three firms with
    the most same-firm pairs, labeler, marketer, corporate group, allocation-harmonised coding, stale entries dropped,
    without discontinued presentations, migration intervals included.
(3) Change tests and their behaviour under a dynamic null: the firm-sharing covariance of the status changes against
    (a) the within drug x interval permutation (as for the three snapshots) and (b) the within drug x interval x earlier
    status permutation; size of both and of the transition test from data simulated with firm-independent transitions
    (each interval's later status drawn with the observed drug x interval x earlier-status rates, sequentially).
(4) First listings: each cell enters once, at its first appearance on the grid, and only if none of its package NDCs
    appeared, under any name, in a capture at or before the date of the firm's record; predictor: the firm's unavailable share over its cells at the grid
    date nearest one year earlier (at least nine months); logistic regression with the drug's share among its other firms,
    log cell size and year indicators; firm-cluster bootstrap and a global firm-relabelling permutation test (one
    permutation of firm labels for all dates). Sensitivities: firm size and injectable controls, related products
    excluded, ingredient families as drugs, units, codings, periods, two-year record.
(5) Lag covariance on first listings only: the covariance of a firm's cells at s with its cells first listed at t, over
    the same firms' same-date firm-sharing covariance, by lag; firm-cluster bootstrap that resamples numerator and
    denominator together.
(6) Records and events: the share of a firm's cells that carry the firm's most common update date; exits of single firms
    from drugs that stay listed; Hurricane Helene (intravenous fluids) and Akorn's shutdown as seen in the captures.
Writes results/long_panel.json and results/fda_csv_availability_coding.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import RESULTS, CORPORATE_GROUP  # noqa: E402
from analysis import fda_csv as fc  # noqa: E402
from analysis import panel  # noqa: E402
from analysis.outcome_variants import size_class  # noqa: E402
from analysis.descriptives import ingredient_family  # noqa: E402

SEED = 20261021
N_PERM = 5000
N_PERM_DISP = 20000
N_BOOT = 1000
MIGRATION_SHARE = 0.15
INJECTABLE = r"inject|infusion|for injection|emulsion|irrigant|irrigation|vial|syringe"


# ------------------------------------------------------------------ data
NOT_A_COMPANY = r"delivery|recovery|estimated|injection|tablet|capsule"


def firm_continuity(P):
    """Company renamings found from package-NDC continuity: firm A is renamed B if A stops appearing, at least half of A's
    package NDCs at its last capture appear later under B, and the names are similar (one contains the other, or a
    difflib ratio of at least 0.6); outright transfers between different companies (for example Athenex to Sagent) are
    not renamings, and their NDCs are never counted as first listings (first_listings)."""
    import difflib
    caps = sorted(P); last = {}; ndcs_last = {}
    for ts in caps:
        for f, g in P[ts].groupby("firm_unit"):
            last[f] = ts; ndcs_last[f] = set(g["package_ndc"].dropna())
    later = {}
    for f, ts in last.items():
        if ts == caps[-1] or not ndcs_last[f]:
            continue
        cnt = {}
        for t2 in caps[caps.index(ts) + 1:]:
            q = P[t2][P[t2]["package_ndc"].isin(ndcs_last[f])]
            for k, g2 in q.groupby("firm_unit"):
                cnt[k] = cnt.get(k, set()) | set(g2["package_ndc"])
        if cnt:
            b, s_ = max(cnt.items(), key=lambda kv: len(kv[1]))
            similar = f in b or b in f or difflib.SequenceMatcher(None, f, b).ratio() >= 0.6
            if b != f and len(s_) >= 0.5 * len(ndcs_last[f]) and similar:
                later[f] = b
    return later


def load():
    """Presentations of every capture with canonical drug names, repaired and continuous firm names, one row per package
    NDC, marketer and corporate group; and the grid. Writes the drug and firm crosswalks to results/."""
    caps = fc.captures()
    P = {ts: fc.read_capture(p, d).assign(date=d) for ts, d, p in caps}
    # firm names that are not company names (note text entered in the company field in a few captures) are repaired from the
    # NDC labeler code: the company most often reporting that labeler code elsewhere
    allp = pd.concat(P.values()); ok = ~allp["firm_unit"].str.contains(NOT_A_COMPANY, case=False, regex=True)
    by_lab = allp[ok & allp["labeler"].notna()].groupby("labeler")["firm_unit"].agg(lambda s: s.value_counts().index[0]).to_dict()
    n_repaired = 0
    for ts in P:
        bad = P[ts]["firm_unit"].str.contains(NOT_A_COMPANY, case=False, regex=True)
        n_repaired += int(bad.sum())
        P[ts].loc[bad, "firm_unit"] = P[ts].loc[bad, "labeler"].map(by_lab)
        P[ts] = P[ts][P[ts]["firm_unit"].notna()]
    renames = firm_continuity(P)
    def follow(f):
        seen = set()
        while f in renames and f not in seen:
            seen.add(f); f = renames[f]
        return f
    mapping, merges = fc.canonical_drugs(P)
    names = fc.marketer_names()
    n_dup = 0
    for ts in P:
        P[ts]["firm_unit"] = P[ts]["firm_unit"].map(follow)
        P[ts]["drug_raw"] = P[ts]["drug"]; P[ts]["drug"] = P[ts]["drug"].map(mapping)
        P[ts] = fc.add_marketer(P[ts], names)
        # one row per package NDC: a presentation reported by both a contract manufacturer and its marketer is kept once,
        # under the company that is its marketer of record if that is one of the two
        q = P[ts].assign(_own=lambda x: (x["firm_unit"] == x["marketer"]).astype(int)).sort_values("_own", ascending=False, kind="stable")
        dup = q["package_ndc"].notna() & q.duplicated("package_ndc", keep="first")
        n_dup += int(dup.sum()); P[ts] = q[~dup].drop(columns="_own").sort_index()
        P[ts]["group"] = P[ts]["firm_unit"].map(lambda n: CORPORATE_GROUP.get(n, n))
    pd.DataFrame(sorted((a, b) for a, b in mapping.items() if a != b), columns=["name", "linked_to"]).to_csv(os.path.join(RESULTS, "fda_csv_drug_crosswalk.csv"), index=False)
    pd.DataFrame(sorted(list(fc.RENAMES.items()) + list(renames.items())), columns=["company", "mapped_to"]).to_csv(os.path.join(RESULTS, "fda_csv_firm_crosswalk.csv"), index=False)
    q = pd.Series({ts: f"{d.year}Q{(d.month - 1) // 3 + 1}" for ts, d, _ in caps})
    grid = [q[q == k].index.max() for k in sorted(q.unique())]
    keep = [g for g, nxt in zip(grid, grid[1:] + [None]) if nxt is None or (pd.Timestamp(nxt[:8]) - pd.Timestamp(g[:8])).days >= 60]
    return P, keep, dict(n_captures=len(caps), drug_name_merges=int(merges), n_names_mapped=int(sum(a != b for a, b in mapping.items())),
                         grid_dropped=[g for g in grid if g not in keep], firm_renames_ndc=renames, n_firm_renames=len(fc.RENAMES) + len(renames),
                         n_repaired_firm=n_repaired, n_duplicate_ndc_rows=n_dup)


def cells(p, outcome="unavailable", unit=None, nodisc=False, drug_key=None):
    """Firm-drug cells of one capture: status, size, package NDCs, whether every presentation was coded categorically,
    discontinued-only flag and injectable form."""
    q = p[p[outcome].notna()]
    if nodisc:
        q = q[q["discontinued"] == 0]
    if unit in ("labeler", "marketer", "group"):
        q = q[q[unit].notna()].assign(firm_unit=lambda x: x[unit])
    if drug_key == "family":
        q = q.assign(drug=q["drug"].map(ingredient_family))
    c = (q.groupby(["firm_unit", "drug"], as_index=False)
         .agg(n_presentations=(outcome, "size"), unavailable=(outcome, "max"), share=(outcome, "mean"), categorical=("categorical", "min"),
              discontinued_all=("discontinued", "min"), ndcs=("package_ndc", lambda s: frozenset(x for x in s if isinstance(x, str)))))
    c["unavailable"] = c["unavailable"].astype(int)
    c["injectable"] = c["drug"].str.contains(INJECTABLE, case=False, regex=True).astype(int)
    return c


def grid_cells(P, grid, **kw):
    return {ts: cells(P[ts], **kw) for ts in grid}


# ------------------------------------------------------------------ (0) validation, coding and composition
def validation(P):
    out = {}
    for k, path in panel.WAVES:
        W, date, _, _ = panel.wave_cells(path)
        d = pd.Timestamp(date)
        ts = min(P, key=lambda t: abs((P[t]["date"].iloc[0] - d).days))
        C = fc.cells_of(P[ts].assign(drug=P[ts]["drug_raw"]))       # the export's own drug names
        m = W.merge(C, on=["firm_unit", "drug"], suffixes=("_ofda", "_csv"))
        out[k] = dict(openfda_date=str(d.date()), csv_capture=ts, days_apart=int((P[ts]["date"].iloc[0] - d).days),
                      cells_openfda=int(len(W)), cells_csv=int(len(C)), cells_common=int(len(m)),
                      share_common_of_openfda=float(len(m) / len(W)), status_agreement=float((m["unavailable_ofda"] == m["unavailable_csv"]).mean()))
    return out


def migration_agreement(P, start="20231201", end="20240801"):
    """Presentations (firm, package NDC) coded from free text in one capture and categorically in the next capture, in
    captures around the 2024 change of the availability field: agreement of the keyword code (and of the allocation-
    harmonised code) with the firm's own categorical value. A real change of status between the captures counts as
    disagreement, so this is a lower bound on the coding's accuracy."""
    ts = sorted(t for t in P if start <= t[:8] <= end); rows = []
    for a, b in zip(ts[:-1], ts[1:]):
        A = P[a][(P[a]["categorical"] == 0) & P[a]["unavailable"].notna() & P[a]["package_ndc"].notna()]
        B = P[b][(P[b]["categorical"] == 1) & P[b]["package_ndc"].notna()]
        m = A.merge(B, on=["firm_unit", "package_ndc"], suffixes=("_a", "_b")).drop_duplicates(["firm_unit", "package_ndc"])
        if len(m):
            rows.append(dict(a=a, b=b, n=int(len(m)), agree=float((m["unavailable_a"] == m["unavailable_b"]).mean()),
                             agree_harmonised=float((m["unavailable_harmonised_a"] == m["unavailable_b"]).mean()),
                             days=int((pd.Timestamp(b[:8]) - pd.Timestamp(a[:8])).days)))
    R = pd.DataFrame(rows)
    big = R[R["n"] >= 50]
    allm = []
    for a, b in zip(ts[:-1], ts[1:]):
        A = P[a][(P[a]["categorical"] == 0) & P[a]["unavailable"].notna() & P[a]["package_ndc"].notna()]
        B = P[b][(P[b]["categorical"] == 1) & P[b]["package_ndc"].notna()]
        allm.append(A.merge(B, on=["firm_unit", "package_ndc"], suffixes=("_a", "_b")).drop_duplicates(["firm_unit", "package_ndc"]))
    M = pd.concat(allm)
    by_firm = M.groupby("firm_unit").apply(lambda g: pd.Series(dict(n=len(g), agree=float((g["unavailable_a"] == g["unavailable_b"]).mean()),
                                                                   free_unav_to_avail=int(((g["unavailable_a"] == 1) & (g["unavailable_b"] == 0)).sum()))))
    by_firm = by_firm[by_firm["n"] >= 10].sort_values("n", ascending=False)
    one_way = int(((M["unavailable_a"] == 1) & (M["unavailable_b"] == 0)).sum()); n_unav = int((M["unavailable_a"] == 1).sum())
    return dict(pairs=rows, by_firm=by_firm.reset_index().to_dict(orient="records"), free_unavailable=n_unav, free_unavailable_to_available=one_way,
                n_total=int(R["n"].sum()), agree_pooled=float((R["agree"] * R["n"]).sum() / R["n"].sum()),
                agree_harmonised_pooled=float((R["agree_harmonised"] * R["n"]).sum() / R["n"].sum()),
                agree_min=float(big["agree"].min()), agree_max=float(big["agree"].max()), n_pairs_big=int(len(big)))


def coding_summary(P):
    txt = pd.concat([P[ts][["availability_text", "unavailable", "unavailable_harmonised", "categorical", "stale", "date"]] for ts in P])
    pre = txt[txt["date"] < pd.Timestamp("2024-03-01")]
    cod = txt.groupby(["availability_text"], dropna=False).agg(n=("categorical", "size"), coded=("unavailable", "first"),
                                                               coded_harmonised=("unavailable_harmonised", "first"), categorical=("categorical", "max")).reset_index()
    cod.sort_values("n", ascending=False).to_csv(os.path.join(RESULTS, "fda_csv_availability_coding.csv"), index=False)
    ft = txt[txt["categorical"] == 0]
    return dict(presentations=int(len(txt)), categorical=int(txt["categorical"].sum()), free_text_coded=int((ft["unavailable"].notna()).sum()),
                uninformative=int(txt["unavailable"].isna().sum()), distinct_strings=int(len(cod)),
                harmonised_recoded=int(((txt["unavailable"] == 1) & (txt["unavailable_harmonised"] == 0)).sum()),
                stale=int(txt["stale"].sum()), stale_share_of_free_text_unavailable=float(txt["stale"].sum() / max(1, (ft["unavailable"] == 1).sum())),
                pre2024_categorical_unavailable=float(pre.loc[pre["categorical"] == 1, "unavailable"].mean()),
                pre2024_free_text_unavailable=float(pre.loc[pre["categorical"] == 0, "unavailable"].mean()))


def composition(P):
    rows = []
    for ts in sorted(P):
        p = P[ts]; d = p["date"].iloc[0]; c = fc.cells_of(p)
        first_post = p.groupby("drug")["initial_posting"].min()
        rows.append(dict(timestamp=ts, date=str(d.date()), presentations=int(len(p)), drugs=int(p["drug"].nunique()),
                         firms=int(p["firm_unit"].nunique()), cells=int(len(c)), unavailable_cells=int(c["unavailable"].sum()),
                         cell_rate=float(c["unavailable"].mean()), categorical_share=float(p["categorical"].mean()),
                         dropped_uninformative=int(p["unavailable"].isna().sum()),
                         drugs_first_posted_180d=int((first_post >= d - pd.Timedelta(days=180)).sum())))
    return rows


# ------------------------------------------------------------------ (1) recurrence
def _block_perm(y, block, rng):
    order = np.lexsort((np.arange(len(y)), block)); new = np.lexsort((rng.random(len(y)), block))
    yp = np.empty_like(y); yp[order] = y[new]; return yp


def _disp(y, g, keep):
    n = np.bincount(g); s = np.bincount(g, weights=y); sh = (s / np.maximum(n, 1))[keep]
    return sh.var(ddof=1)


def recurrence_one(c, rng, n_perm=N_PERM_DISP):
    """Dispersion ratios with the size-stratified conditional permutation p-values and null means (firm and drug)."""
    y = c["unavailable"].values.astype(float)
    f = c["firm_unit"].astype("category").cat.codes.values; d = c["drug"].astype("category").cat.codes.values
    sz = size_class(c["n_presentations"].values)
    dz = pd.factorize(pd.Series(d).astype(str) + "|" + pd.Series(sz).astype(str))[0]
    fz = pd.factorize(pd.Series(f).astype(str) + "|" + pd.Series(sz).astype(str))[0]
    out = dict(n_cells=int(len(c)), n_unavailable=int(y.sum()), n_firms=int(c["firm_unit"].nunique()), n_drugs=int(c["drug"].nunique()))
    p = y.mean()
    for key, g, blk in (("firm", f, dz), ("drug", d, fz)):
        n = np.bincount(g); keep = n >= 2
        obs = _disp(y, g, keep); expct = np.mean(p * (1 - p) / n[keep])
        null = np.array([_disp(_block_perm(y, blk, rng), g, keep) for _ in range(n_perm)])
        out[f"{key}_ratio"] = float(obs / expct); out[f"{key}_null_ratio"] = float(null.mean() / expct)
        out[f"{key}_p_size"] = float((1 + (null >= obs).sum()) / (1 + n_perm)); out[f"{key}_groups"] = int(keep.sum())
    return out


# ------------------------------------------------------------------ (2), (3) intervals
def interval_frame(W, grid):
    rows = []
    for a, b in zip(grid[:-1], grid[1:]):
        m = W[a].merge(W[b], on=["firm_unit", "drug"], suffixes=("_a", "_b"))
        m["transition"] = f"{a[:8]}-{b[:8]}"; m["start"] = pd.Timestamp(a[:8]); m["days"] = (pd.Timestamp(b[:8]) - pd.Timestamp(a[:8])).days
        rows.append(m)
    D = pd.concat(rows, ignore_index=True)
    sw = ((D["categorical_a"] == 0) & (D["categorical_b"] == 1)).groupby(D["transition"]).mean()
    D["switch_share"] = D["transition"].map(sw)
    D["migration"] = (D["switch_share"] > MIGRATION_SHARE).astype(int)
    return D


def _codes(D):
    fcode = (D["firm_unit"] + "|" + D["transition"]).astype("category").cat.codes.values
    dcode = (D["drug"] + "|" + D["transition"]).astype("category").cat.codes.values
    return fcode, dcode


def transition_test_frame(D, rng, n_perm=N_PERM, by_size=False, start=None, yb=None, ya=None):
    """panel.transition_test on a prebuilt interval frame (any number of intervals); yb/ya override the statuses (for
    simulations)."""
    fcode, dcode = _codes(D)
    PF, _ = panel._pairs(fcode, dcode); I, J = PF
    ya = D["unavailable_a"].values.astype(float) if ya is None else ya
    yb = D["unavailable_b"].values.astype(float) if yb is None else yb
    tr = pd.factorize(D["transition"])[0]; cnt = np.bincount(tr)
    def centre(y):
        return y - (np.bincount(tr, weights=y) / cnt)[tr]
    ca = centre(ya)
    def stat(y):
        cb = centre(y); return 0.5 * ((ca[I] * cb[J]).mean() + (cb[I] * ca[J]).mean())
    key = D["drug"] + "|" + D["transition"] + "|" + pd.Series(ya.astype(int), index=D.index).astype(str)
    if by_size:
        key = key + "|" + pd.Series(size_class(D["n_presentations_b"].values), index=D.index).astype(str)
    if start is not None:
        fixed = ya.astype(int) != start
        key = key.where(~fixed, "fixed|" + pd.Series(np.arange(len(D)), index=D.index).astype(str))
    block = key.astype("category").cat.codes.values
    obs = stat(yb); null = np.array([stat(_block_perm(yb, block, rng)) for _ in range(n_perm)])
    return dict(n_cells=int(len(D)), n_intervals=int(D["transition"].nunique()), n_pairs=int(len(I)), cross_cov=float(obs),
                cross_null_mean=float(null.mean()), p=float((1 + (null >= obs).sum()) / (1 + n_perm)),
                stay_unavailable=float(yb[ya == 1].mean()), stay_available=float(1 - yb[ya == 0].mean()))


def change_tests(D, rng, n_perm=N_PERM, yb=None, ya=None):
    """Firm-sharing covariance of the centred status changes, against (a) permutation of the changes within drug x
    interval and (b) permutation of the later status within drug x interval x earlier status."""
    fcode, dcode = _codes(D); PF, PD = panel._pairs(fcode, dcode)
    ya = D["unavailable_a"].values.astype(float) if ya is None else ya
    yb = D["unavailable_b"].values.astype(float) if yb is None else yb
    tr = pd.factorize(D["transition"])[0]; cnt = np.bincount(tr)
    def delta(y):
        dl = y - ya; return dl - (np.bincount(tr, weights=dl) / cnt)[tr]
    dl = delta(yb)
    p_a = panel._perm_p(dl, PF, dcode, rng, n_perm)
    blk = (D["drug"] + "|" + D["transition"] + "|" + pd.Series(ya.astype(int), index=D.index).astype(str)).astype("category").cat.codes.values
    obs = panel._cov(dl, PF); null = np.array([panel._cov(delta(_block_perm(yb, blk, rng)), PF) for _ in range(n_perm)])
    return dict(n_changes=int((yb != ya).sum()), cov_firm=float(obs), p_within_drug=p_a, p_within_start=float((1 + (null >= obs).sum()) / (1 + n_perm)),
                p_drug=panel._perm_p(dl, PD, fcode, rng, n_perm))


def interval_tests(D, rng, label, n_perm=N_PERM):
    out = dict(transition=transition_test_frame(D, rng, n_perm), transition_size=transition_test_frame(D, rng, n_perm, by_size=True),
               stay=transition_test_frame(D, rng, n_perm, start=1), become=transition_test_frame(D, rng, n_perm, start=0),
               change=change_tests(D, rng, n_perm))
    out["n_onsets"] = int(((D["unavailable_a"] == 0) & (D["unavailable_b"] == 1)).sum())
    out["n_recoveries"] = int(((D["unavailable_a"] == 1) & (D["unavailable_b"] == 0)).sum())
    print(label, {k: round(v["p"], 4) for k, v in out.items() if isinstance(v, dict) and "p" in v},
          {k: round(out["change"][k], 4) for k in ("p_within_drug", "p_within_start")}, "changes", out["change"]["n_changes"])
    return out


def null_simulation(D, rng, n_sim=200, n_perm=400):
    """Size of the transition test and of the two change tests under firm-independent transitions: intervals in time
    order, each cell's earlier status taken from the simulation when the cell was simulated at the previous interval's
    later date (else observed), and its later status drawn with the observed rate of its drug x interval x earlier status
    (pooled over drugs when that block is empty)."""
    D = D.sort_values(["start"]).reset_index(drop=True)
    rate = D.groupby(["drug", "transition", "unavailable_a"])["unavailable_b"].mean().to_dict()
    pooled = D.groupby(["transition", "unavailable_a"])["unavailable_b"].mean().to_dict()
    trs = list(dict.fromkeys(D["transition"])); idx = {t: np.flatnonzero(D["transition"].values == t) for t in trs}
    keys = list(zip(D["firm_unit"], D["drug"])); drugs = D["drug"].values; rej = dict(transition=0, change_within_drug=0, change_within_start=0)
    for _ in range(n_sim):
        ya = D["unavailable_a"].values.astype(float).copy(); yb = np.empty(len(D)); last = {}; prev_end = None
        for t in trs:
            ii = idx[t]
            if prev_end is not None and t.split("-")[0] != prev_end:
                last = {}                                # an excluded interval lies between: start again from the observed status
            prev_end = t.split("-")[1]
            for i in ii:
                if keys[i] in last: ya[i] = last[keys[i]]
                r = rate.get((drugs[i], t, ya[i]), pooled.get((t, ya[i]), 0.5))
                yb[i] = float(rng.random() < r)
            last = {keys[i]: yb[i] for i in ii}
        rej["transition"] += transition_test_frame(D, rng, n_perm, yb=yb, ya=ya)["p"] < 0.05
        ct = change_tests(D, rng, n_perm, yb=yb, ya=ya)
        rej["change_within_drug"] += ct["p_within_drug"] < 0.05; rej["change_within_start"] += ct["p_within_start"] < 0.05
    out = {k: v / n_sim for k, v in rej.items()}; out.update(n_sim=n_sim, n_perm=n_perm)
    print("null simulation", out)
    return out


def top_firms(D, k=3):
    pairs = D.groupby(["transition", "firm_unit"]).size(); pairs = (pairs * (pairs - 1) / 2).groupby("firm_unit").sum().sort_values(ascending=False)
    return list(pairs.index[:k]), float(pairs.iloc[:k].sum() / pairs.sum())


# ------------------------------------------------------------------ (4) first listings
def _first_listing_rows(W, grid, P, lag_days, min_gap_days):
    """New cells (first appearance on the grid; no package NDC seen in any capture at or before s) with the firm's record at s."""
    caps = sorted(P); seen_cells = set(zip(W[grid[0]]["firm_unit"], W[grid[0]]["drug"])); rows = []
    dates = {ts: pd.Timestamp(ts[:8]) for ts in grid}
    ndc_first = {}
    for ts in caps:
        for k in P[ts]["package_ndc"].dropna().unique():
            ndc_first.setdefault(k, ts)
    for t in grid[1:]:
        C = W[t]; dt = dates[t]
        cand = [s for s in grid if (dt - dates[s]).days >= min_gap_days]
        if cand:
            s = min(cand, key=lambda x: abs((dt - dates[x]).days - lag_days))
            A = W[s]
            fs = A.groupby("firm_unit").agg(share=("unavailable", "mean"), n=("unavailable", "size"), inj=("injectable", "mean"))
            fam = A.assign(fam=A["drug"].map(ingredient_family)).groupby("firm_unit")["fam"].agg(set)
            dsum = C.groupby("drug")["unavailable"].agg(["sum", "size"])
            for r in C.itertuples(index=False):
                if (r.firm_unit, r.drug) in seen_cells or r.firm_unit not in fs.index:
                    continue
                if any(ndc_first.get(k, t) <= s for k in r.ndcs):
                    continue                                     # an NDC of the cell was listed at or before s, under some name
                do = dsum.loc[r.drug, "size"] - 1
                prev = grid[grid.index(t) - 1]
                rows.append(dict(t=t, s=s, gap=(dt - dates[s]).days, firm_unit=r.firm_unit, drug=r.drug, unavailable=int(r.unavailable),
                                 categorical=int(r.categorical), prev_days=int((dt - dates[prev]).days),
                                 fs=float(fs.loc[r.firm_unit, "share"]), f_n=int(fs.loc[r.firm_unit, "n"]), f_inj=float(fs.loc[r.firm_unit, "inj"]),
                                 related=int(ingredient_family(r.drug) in fam.get(r.firm_unit, set())), injectable=int(r.injectable),
                                 ds=float((dsum.loc[r.drug, "sum"] - r.unavailable) / do) if do > 0 else np.nan,
                                 log_pres=float(np.log(r.n_presentations)), year=int(dt.year)))
        seen_cells |= set(zip(C["firm_unit"], C["drug"]))
    E = pd.DataFrame(rows)
    E["ds_missing"] = E["ds"].isna().astype(int); E["ds"] = E["ds"].fillna(0.0)
    return E


def first_listings(W, grid, P, rng, lag_days=365, min_gap_days=270, n_perm=2000, n_boot=N_BOOT, controls=(), subset=None, label="first listings"):
    """Logistic odds ratio of the firm's earlier unavailable share for the status of new cells (see module docstring)."""
    E = _first_listing_rows(W, grid, P, lag_days, min_gap_days)
    if subset is not None:
        E = E[subset(E)].reset_index(drop=True)
    if "log_firm_size" in controls:
        E["log_firm_size"] = np.log(E["f_n"])
    years = sorted(E["year"].unique())
    def X(d, fsv=None):
        cols = [np.ones(len(d)), (d["fs"] if fsv is None else fsv).values.astype(float), d["ds"].values, d["ds_missing"].values, d["log_pres"].values]
        cols += [d[c].values.astype(float) for c in controls]
        cols += [(d["year"] == y).values.astype(float) for y in years[1:]]
        return np.column_stack(cols)
    y = E["unavailable"].values.astype(float)
    b = panel.fit_logit(X(E), y)
    out = dict(n_new=int(len(E)), n_unavailable=int(y.sum()), n_firms=int(E["firm_unit"].nunique()), n_dates=int(E["t"].nunique()),
               median_gap_days=float(E["gap"].median()), or_fs=float(np.exp(b[1])), n_related=int(E["related"].sum()),
               n_before_2024=int((E["t"] < "20240301").sum()),
               rate_fs_high=float(y[E["fs"] > 0.5].mean()), n_fs_high=int((E["fs"] > 0.5).sum()),
               rate_fs_low=float(y[E["fs"] <= 0.5].mean()), n_fs_low=int((E["fs"] <= 0.5).sum()))
    # share of new cells unavailable by the firm's earlier share, in four bins, with firm-cluster bootstrap intervals
    edges = [(-0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    out["bins"] = [dict(lo=max(lo, 0.0), hi=hi, n=int(((E["fs"] > lo) & (E["fs"] <= hi)).sum()), rate=float(y[(E["fs"] > lo) & (E["fs"] <= hi)].mean()))
                   for lo, hi in edges]
    groups = {f: g for f, g in E.groupby("firm_unit")}; firms = list(groups); bs = []; bb = []
    for _ in range(n_boot):
        sb = pd.concat([groups[f] for f in rng.choice(firms, len(firms), replace=True)])
        bb.append([sb.loc[(sb["fs"] > lo) & (sb["fs"] <= hi), "unavailable"].mean() for lo, hi in edges])
        if sb["unavailable"].nunique() < 2: continue
        bs.append(panel.fit_logit(X(sb), sb["unavailable"].values.astype(float))[1])
    bb = np.array(bb, float)
    for k, d_ in enumerate(out["bins"]):
        d_["lo95"], d_["hi95"] = float(np.nanpercentile(bb[:, k], 2.5)), float(np.nanpercentile(bb[:, k], 97.5))
    bs = np.array(bs); bs = bs[np.isfinite(bs) & (np.abs(bs) < 50)]
    out["or_lo"], out["or_hi"] = float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))
    out["n_boot"] = int(len(bs)); out["p_boot_gt1"] = float((bs > 0).mean())
    # global firm relabelling: one permutation of the firms' labels for every date; each new cell takes its partner firm's
    # share at the same earlier date (the mean share at that date if the partner was not listed then)
    share = {s: W[s].groupby("firm_unit")["unavailable"].mean() for s in E["s"].unique()}
    allf = sorted(set().union(*[set(v.index) for v in share.values()]))
    null = np.empty(n_perm)
    for k in range(n_perm):
        pi = dict(zip(allf, rng.permutation(allf)))
        fsp = pd.Series([share[s].get(pi[f], share[s].mean()) for s, f in zip(E["s"], E["firm_unit"])], index=E.index)
        null[k] = panel.fit_logit(X(E, fsp), y)[1]
    out["perm_p"] = float((1 + (null >= b[1]).sum()) / (1 + n_perm)); out["n_perm"] = n_perm
    # secondary scheme: within each date, the earlier shares permuted among the firms with a new listing on that date (no
    # imputation, but firm shares are not moved together across dates)
    fsh = E.groupby(["t", "firm_unit"])["fs"].first(); null2 = np.empty(n_perm)
    for k in range(n_perm):
        perm = {}
        for t_, sub in fsh.groupby(level=0):
            perm.update(dict(zip(sub.index, sub.values[rng.permutation(len(sub))])))
        null2[k] = panel.fit_logit(X(E, pd.Series([perm[(t_, f)] for t_, f in zip(E["t"], E["firm_unit"])], index=E.index)), y)[1]
    out["perm_p_within_date"] = float((1 + (null2 >= b[1]).sum()) / (1 + n_perm))
    print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    return out


# ------------------------------------------------------------------ (5) lag covariance on first listings
def first_listing_lag(W, grid, P, rng, n_boot=N_BOOT, min_lag_days=270):
    """Covariance of a firm's cells at s with its cells first listed at t (t - s >= min_lag_days), over the same-date
    firm-sharing covariance of the same firms; by lag; firm-cluster bootstrap resampling numerator and denominator."""
    E = _first_listing_rows(W, grid, P, 365, min_lag_days)       # the new cells (their s is not used here)
    newc = {(t, f, d): u for t, f, d, u in zip(E["t"], E["firm_unit"], E["drug"], E["unavailable"])}
    mean = {ts: W[ts]["unavailable"].mean() for ts in grid}
    # same-date firm-sharing sums per firm and date (pairs of a firm's cells in different drugs, centred at the date mean)
    same = {}
    for ts in grid:
        c = W[ts]; yc = c["unavailable"].values - mean[ts]
        for f, idx in c.groupby("firm_unit").indices.items():
            v = yc[idx]; n = len(v)
            if n >= 2: same[(ts, f)] = ((v.sum() ** 2 - (v ** 2).sum()) / 2, n * (n - 1) / 2)
    rows = []
    dates = {ts: pd.Timestamp(ts[:8]) for ts in grid}
    for (t, f, d), u in newc.items():
        for s in grid:
            lag = (dates[t] - dates[s]).days
            if lag < min_lag_days: continue
            A = W[s]; a = A[A["firm_unit"] == f]
            if not len(a): continue
            prod = float(((a["unavailable"].values - mean[s]) * (u - mean[t])).sum())
            ss, sc = same.get((s, f), (0.0, 0.0)); ts_, tc = same.get((t, f), (0.0, 0.0))
            rows.append(dict(firm=f, lag=lag, prod=prod, n=len(a), same_sum=0.5 * (ss + ts_), same_n=0.5 * (sc + tc)))
    R = pd.DataFrame(rows); R["lag_years"] = np.minimum((R["lag"] / 365.25).astype(int), 5)
    def ratio(x):
        den = x["same_sum"].sum() / max(x["same_n"].sum(), 1e-12)
        return (x["prod"].sum() / x["n"].sum()) / den if den else float("nan")
    out = dict(n_new_cells=int(len(newc)), n_pairs=int(R["n"].sum()), n_firms=int(R["firm"].nunique()), ratio=float(ratio(R)),
               new_cell_rate=float(np.mean(list(newc.values()))))
    byf = {f: x for f, x in R.groupby("firm")}; firms = list(byf); boots = {L: [] for L in sorted(R["lag_years"].unique())}; tot = []
    for _ in range(n_boot):
        sb = pd.concat([byf[f] for f in rng.choice(firms, len(firms), replace=True)])
        tot.append(ratio(sb))
        for L, x in sb.groupby("lag_years"):
            boots[L].append(ratio(x))
    out["ratio_lo"], out["ratio_hi"] = float(np.nanpercentile(tot, 2.5)), float(np.nanpercentile(tot, 97.5))
    out["by_lag_years"] = {}
    for L, x in R.groupby("lag_years"):
        arr = np.array(boots[L], float); arr = arr[np.isfinite(arr)]
        out["by_lag_years"][int(L)] = dict(n_pairs=int(x["n"].sum()), ratio=float(ratio(x)), ratio_lo=float(np.percentile(arr, 2.5)), ratio_hi=float(np.percentile(arr, 97.5)))
    print("first-listing lag", {k: v for k, v in out.items() if k != "by_lag_years"}); print(out["by_lag_years"])
    return out


# ------------------------------------------------------------------ (5b) checks of alternative explanations (domain referee, round 5)
LVP_PN = (panel.IV_FLUID_PATTERN + r"|amino acid|lipid|fat emulsion|potassium|calcium gluconate|calcium chloride|magnesium sulfate|"
          r"sodium bicarbonate|sodium acetate|sodium phosphate|trace element|multivitamin|cysteine|selenious|zinc")


def flag_long_unavailable(P, grid, days):
    """Mark, at each grid date, presentations (firm, package NDC) that were unavailable at every grid date of the preceding
    `days` and at the last grid date before that window: SKUs left on the list as unavailable for years, which may be
    de facto discontinued. Adds the outcome column unavailable_nolong{days} (such presentations treated as not listed)."""
    dates = {ts: pd.Timestamp(ts[:8]) for ts in grid}
    unav = {ts: set(zip(P[ts].loc[P[ts]["unavailable"] == 1, "firm_unit"], P[ts].loc[P[ts]["unavailable"] == 1, "package_ndc"])) for ts in grid}
    col = f"unavailable_nolong{days}"; n_flag = {}
    for ts in P:
        P[ts][col] = P[ts]["unavailable"]
    for t in grid:
        win = [g for g in grid if dates[t] - pd.Timedelta(days=days) <= dates[g] <= dates[t]]
        before = [g for g in grid if dates[g] < dates[t] - pd.Timedelta(days=days)]
        if not before:
            continue
        keep = set.intersection(*[unav[g] for g in win + [before[-1]]])
        q = P[t]; f = [(a, b) in keep for a, b in zip(q["firm_unit"], q["package_ndc"])]
        P[t].loc[f, col] = np.nan; n_flag[t] = int(np.sum(f))
    return n_flag


def fractional_snapshot(c, rng, n_perm=N_PERM_DISP):
    """Presentation-level outcome on one grid date: firm-level dispersion of the share of a cell's presentations that are
    unavailable, against permutation of the cell shares within drug and within drug x size class."""
    y = c["share"].values.astype(float)
    f = c["firm_unit"].astype("category").cat.codes.values; d = c["drug"].astype("category").cat.codes.values
    sz = size_class(c["n_presentations"].values); dz = pd.factorize(pd.Series(d).astype(str) + "|" + pd.Series(sz).astype(str))[0]
    n = np.bincount(f); keep = n >= 2; obs = _disp(y, f, keep); out = dict(n_cells=int(len(c)))
    for lab, blk in (("p", d), ("p_size", dz)):
        null = np.array([_disp(_block_perm(y, blk, rng), f, keep) for _ in range(n_perm)])
        out[lab] = float((1 + (null >= obs).sum()) / (1 + n_perm))
    return out


def top3_by_size(c, top3):
    sz = size_class(c["n_presentations"].values); t = c["firm_unit"].isin(top3).values; y = c["unavailable"].values
    return {str(k): dict(top3=float(y[(sz == k) & t].mean()), other=float(y[(sz == k) & ~t].mean()), n_top3=int(((sz == k) & t).sum()),
                         n_other=int(((sz == k) & ~t).sum())) for k in (1, 2, 3)}


def firm_propensity_persistence(D, early_end="2022-01-01", late_start="2024-03-01", min_cells=10):
    """Do the same firms have more onsets than expected early and late? For each firm, observed minus expected onsets
    (expected: the onset rate of the cell's drug x interval among cells available at the earlier date) per available
    cell-interval, separately for intervals starting before early_end and after late_start; Spearman correlation over firms
    with at least min_cells available cell-intervals in both periods. Likewise for recoveries."""
    from scipy import stats
    out = {}
    for kind, start, col in (("onset", 0, "unavailable_b"), ("recovery", 1, None)):
        X = D[D["unavailable_a"] == start].copy()
        X["event"] = X["unavailable_b"] if start == 0 else 1 - X["unavailable_b"]
        X["expected"] = X.groupby(["drug", "transition"])["event"].transform("mean")
        X["excess"] = X["event"] - X["expected"]
        e = X[X["start"] < pd.Timestamp(early_end)].groupby("firm_unit")["excess"].agg(["mean", "size"])
        l = X[X["start"] >= pd.Timestamp(late_start)].groupby("firm_unit")["excess"].agg(["mean", "size"])
        j = e.join(l, lsuffix="_e", rsuffix="_l", how="inner"); j = j[(j["size_e"] >= min_cells) & (j["size_l"] >= min_cells)]
        rho, p = stats.spearmanr(j["mean_e"], j["mean_l"])
        out[kind] = dict(n_firms=int(len(j)), spearman=float(rho), p=float(p))
    return out


def onset_by_share(D, W, rng, n_boot=N_BOOT):
    """Purchaser's quantity: among cells available at the earlier date of an interval, the share unavailable at the later
    date, by the firm's unavailable share over its other cells at the earlier date (four bins; firm-cluster bootstrap)."""
    rows = []
    for tr, g in D.groupby("transition"):
        a = [k for k in W if k.startswith(tr.split("-")[0])][0]; A = W[a]
        fs = A.groupby("firm_unit")["unavailable"].agg(["sum", "size"])
        for r in g[g["unavailable_a"] == 0].itertuples(index=False):
            s_, n_ = fs.loc[r.firm_unit]
            if n_ >= 2:
                rows.append(dict(firm=r.firm_unit, share=(s_ - 0) / (n_ - 1), y=int(r.unavailable_b)))
    E = pd.DataFrame(rows); edges = [(-0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    out = [dict(lo=max(lo, 0.0), hi=hi, n=int(((E["share"] > lo) & (E["share"] <= hi)).sum()), rate=float(E.loc[(E["share"] > lo) & (E["share"] <= hi), "y"].mean()))
           for lo, hi in edges]
    groups = {f: x for f, x in E.groupby("firm")}; firms = list(groups); bb = []
    for _ in range(n_boot):
        sb = pd.concat([groups[f] for f in rng.choice(firms, len(firms), replace=True)])
        bb.append([sb.loc[(sb["share"] > lo) & (sb["share"] <= hi), "y"].mean() for lo, hi in edges])
    bb = np.array(bb, float)
    for k, d_ in enumerate(out):
        d_["lo95"], d_["hi95"] = float(np.nanpercentile(bb[:, k], 2.5)), float(np.nanpercentile(bb[:, k], 97.5))
    return dict(n=int(len(E)), n_firms=int(E["firm"].nunique()), rate=float(E["y"].mean()), bins=out)


# ------------------------------------------------------------------ (6) records and events
def update_stamps(P, grid):
    """Share of a firm's cells (firms with at least three cells) that carry the firm's most common update date."""
    rows = []
    for ts in grid:
        p = P[ts][P[ts]["unavailable"].notna()]
        c = p.groupby(["firm_unit", "drug"])["update_date"].max().reset_index()
        for f, g in c.groupby("firm_unit"):
            if len(g) >= 3:
                rows.append(dict(ts=ts, share=float(g["update_date"].value_counts(normalize=True).iloc[0])))
    R = pd.DataFrame(rows)
    return dict(median_share=float(R["share"].median()), mean_share=float(R["share"].mean()),
                mean_share_2025_2026=float(R.loc[R["ts"] >= "20250101", "share"].mean()))


def firm_exits(W, grid):
    """Cells that leave the list while their drug stays listed, per interval, by earlier status."""
    rows = []
    for a, b in zip(grid[:-1], grid[1:]):
        A, B = W[a], W[b]; dl = set(B["drug"]); listed = set(zip(B["firm_unit"], B["drug"]))
        A = A[A["drug"].isin(dl)]
        ex = np.array([(f, d) not in listed for f, d in zip(A["firm_unit"], A["drug"])], dtype=bool)
        rows.append(dict(n_avail=int((A["unavailable"] == 0).sum()), ex_avail=int((ex & (A["unavailable"].values == 0)).sum()),
                         n_unav=int((A["unavailable"] == 1).sum()), ex_unav=int((ex & (A["unavailable"].values == 1)).sum())))
    R = pd.DataFrame(rows)
    return dict(rate_available=float(R["ex_avail"].sum() / R["n_avail"].sum()), rate_unavailable=float(R["ex_unav"].sum() / R["n_unav"].sum()))


def events(P):
    """Hurricane Helene (late September 2024; Baxter's North Cove plant) seen in the captures around it, and Akorn's
    shutdown (February 2023): intravenous-fluid cells by firm before and after, and Akorn's last listing."""
    caps = sorted(P)
    def near(day):
        return min(caps, key=lambda t: abs((pd.Timestamp(t[:8]) - pd.Timestamp(day)).days))
    a, b = near("2024-09-27"), near("2024-11-05")
    out = dict(helene_before=a, helene_after=b, firms={})
    for ts, key in ((a, "before"), (b, "after")):
        c = fc.cells_of(P[ts]); c = c[c["drug"].str.contains(panel.IV_FLUID_PATTERN, case=False, regex=True)]
        for f, g in c.groupby("firm_unit"):
            out["firms"].setdefault(f, {})[key] = [int(g["unavailable"].sum()), int(len(g))]
    out["all_firms"] = {k: [sum(v.get(k, [0, 0])[0] for v in out["firms"].values()), sum(v.get(k, [0, 0])[1] for v in out["firms"].values())]
                        for k in ("before", "after")}
    ak = [t for t in caps if (P[t]["firm_unit"] == "AKORN").any()]
    if ak:
        c = fc.cells_of(P[ak[-1]]).query("firm_unit == 'AKORN'")
        out.update(akorn_last=ak[-1], akorn_cells_last=int(len(c)), akorn_unavailable_last=int(c["unavailable"].sum()))
    return out


def main():
    rng = np.random.default_rng(SEED)
    P, grid, info = load()
    out = dict(info=info, grid=grid)
    out["coding"] = coding_summary(P); print("coding", out["coding"])
    out["migration_agreement"] = migration_agreement(P); print("migration agreement", {k: v for k, v in out["migration_agreement"].items() if k != "pairs"})
    out["validation"] = validation(P); print("validation", out["validation"])
    out["composition"] = composition(P)
    W = grid_cells(P, grid)
    out["recurrence"] = {}
    for ts in grid:
        out["recurrence"][ts] = recurrence_one(W[ts], rng); print("recurrence", ts, {k: round(v, 3) if isinstance(v, float) else v for k, v in out["recurrence"][ts].items()})
    a0, a1 = W[grid[0]], W[grid[-1]]
    out["overlap_first_last"] = dict(share_of_last=float(len(a1.merge(a0[["firm_unit", "drug"]])) / len(a1)),
                                     drugs_share_of_last=float(len(set(a1["drug"]) & set(a0["drug"])) / a1["drug"].nunique()))
    Dall = interval_frame(W, grid)
    out["intervals"] = dict(n=int(Dall["transition"].nunique()), days=Dall.groupby("transition")["days"].first().tolist(),
                            migration=sorted(Dall.loc[Dall["migration"] == 1, "transition"].unique()),
                            switch_share=Dall.groupby("transition")["switch_share"].first().to_dict())
    D = Dall[Dall["migration"] == 0].reset_index(drop=True)
    top3, share = top_firms(D); out["top3"] = dict(firms=top3, pair_share=share)
    T = out["tests"] = {}
    T["all"] = interval_tests(D, rng, "all non-migration intervals")
    T["with_migration"] = interval_tests(Dall, rng, "with migration intervals")
    T["no_top3"] = interval_tests(D[~D["firm_unit"].isin(top3)].reset_index(drop=True), rng, "no top3")
    for lab, lo, hi in (("pre_2022", "2019", "2022"), ("2022_2024", "2022", "2024-03-01"), ("post_migration", "2024-03-01", "2027")):
        sub = D[(D["start"] >= pd.Timestamp(lo)) & (D["start"] < pd.Timestamp(hi))].reset_index(drop=True)
        if sub["transition"].nunique() > 0:
            T[lab] = interval_tests(sub, rng, lab)
    variants = dict(labeler=dict(unit="labeler"), marketer=dict(unit="marketer"), group=dict(unit="group"),
                    harmonised=dict(outcome="unavailable_harmonised"), no_stale=dict(outcome="unavailable_nostale"), no_discontinued=dict(nodisc=True))
    for lab, kw in variants.items():
        Dv = interval_frame(grid_cells(P, grid, **kw), grid); Dv = Dv[Dv["migration"] == 0].reset_index(drop=True)
        T[lab] = interval_tests(Dv, rng, lab)
    out["per_interval"] = {tr: dict(days=int(g["days"].iloc[0]), migration=int(g["migration"].iloc[0]), switch_share=float(g["switch_share"].iloc[0]),
                                    n_changes=int((g["unavailable_a"] != g["unavailable_b"]).sum()),
                                    transition_p=transition_test_frame(g.reset_index(drop=True), rng, 2000)["p"],
                                    **{k: v for k, v in change_tests(g.reset_index(drop=True), rng, 2000).items() if k.startswith("p_")})
                           for tr, g in Dall.groupby("transition")}
    T["categorical_both"] = interval_tests(D[(D["categorical_a"] == 1) & (D["categorical_b"] == 1)].reset_index(drop=True), rng, "both dates categorical")
    T["no_long_gap"] = interval_tests(D[D["days"] <= 200].reset_index(drop=True), rng, "intervals of at most 200 days")
    lvp = {d for w in W.values() for d in w["drug"] if pd.notna(d) and __import__("re").search(LVP_PN, d, flags=__import__("re").I)}
    T["no_lvp"] = interval_tests(D[~D["drug"].isin(lvp)].reset_index(drop=True), rng, "without IV fluids, electrolytes and parenteral nutrition")
    nflag = {d_: flag_long_unavailable(P, grid, d_) for d_ in (730, 365)}
    for d_ in (730, 365):
        Dv = interval_frame(grid_cells(P, grid, outcome=f"unavailable_nolong{d_}"), grid); Dv = Dv[Dv["migration"] == 0].reset_index(drop=True)
        T[f"no_long{d_}"] = interval_tests(Dv, rng, f"long-unavailable {d_} d treated as not listed")
    out["long_unavailable"] = dict(flagged_last=nflag[730].get(grid[-1], 0), flagged_last_1y=nflag[365].get(grid[-1], 0))
    last = grid[-1]
    out["snapshot_checks"] = dict(
        no_long730=recurrence_one(cells(P[last], outcome="unavailable_nolong730"), rng),
        no_long365=recurrence_one(cells(P[last], outcome="unavailable_nolong365"), rng),
        no_lvp=recurrence_one(W[last][~W[last]["drug"].isin(lvp)].reset_index(drop=True), rng),
        fractional=fractional_snapshot(W[last], rng), fractional_nodisc=fractional_snapshot(cells(P[last], nodisc=True), rng),
        top3_by_size=top3_by_size(W[last], top3))
    out["lvp_recurrence"] = {ts: recurrence_one(W[ts][~W[ts]["drug"].isin(lvp)].reset_index(drop=True), rng, n_perm=5000) for ts in grid}
    out["lvp_drugs"] = sorted(lvp)
    cat_dates = [ts for ts in grid if np.mean([c["categorical_share"] for c in out["composition"] if c["timestamp"] == ts] or [0]) > 0.8]
    out["recurrence_unavailable_only"] = {ts: recurrence_one(cells(P[ts], outcome="unavailable_only"), rng, n_perm=5000) for ts in cat_dates}
    out["onset_by_share_post"] = onset_by_share(D[D["start"] >= pd.Timestamp("2024-03-01")].reset_index(drop=True), W, rng)
    print("checks", out["snapshot_checks"], out["onset_by_share_post"])
    out["propensity_persistence"] = firm_propensity_persistence(D); print("propensity persistence", out["propensity_persistence"])
    out["null_simulation"] = null_simulation(D, rng)
    F = out["first_listings"] = {}
    F["main"] = first_listings(W, grid, P, rng)
    F["lag2y"] = first_listings(W, grid, P, rng, lag_days=730, min_gap_days=600, label="first listings, two years")
    F["no_top3"] = first_listings({k: v[~v["firm_unit"].isin(top3)].reset_index(drop=True) for k, v in W.items()}, grid, P, rng, label="first listings, no top3")
    F["firm_size"] = first_listings(W, grid, P, rng, controls=("log_firm_size",), label="first listings, firm size")
    F["firm_size_injectable"] = first_listings(W, grid, P, rng, controls=("log_firm_size", "injectable", "f_inj"), label="first listings, firm size and injectable")
    F["unrelated"] = first_listings(W, grid, P, rng, subset=lambda E: E["related"] == 0, label="first listings, no related product")
    F["pre_2024"] = first_listings(W, grid, P, rng, subset=lambda E: E["t"] < "20240301", label="first listings, before 2024")
    F["families"] = first_listings(grid_cells(P, grid, drug_key="family"), grid, P, rng, label="first listings, ingredient families")
    F["no_discontinued"] = first_listings(grid_cells(P, grid, nodisc=True), grid, P, rng, label="first listings, no discontinued")
    F["harmonised"] = first_listings(grid_cells(P, grid, outcome="unavailable_harmonised"), grid, P, rng, label="first listings, harmonised")
    F["marketer"] = first_listings(grid_cells(P, grid, unit="marketer"), grid, P, rng, label="first listings, marketer")
    F["group"] = first_listings(grid_cells(P, grid, unit="group"), grid, P, rng, label="first listings, group")
    F["categorical"] = first_listings(W, grid, P, rng, subset=lambda E: E["categorical"] == 1, label="first listings, categorically coded new cells")
    F["no_long_gap"] = first_listings(W, grid, P, rng, subset=lambda E: E["prev_days"] <= 200, label="first listings, after intervals of at most 200 days")
    F["no_long730"] = first_listings(grid_cells(P, grid, outcome="unavailable_nolong730"), grid, P, rng, label="first listings, long-unavailable not listed")
    out["first_listing_lag"] = first_listing_lag(W, grid, P, rng)
    out["update_stamps"] = update_stamps(P, grid); print("update stamps", out["update_stamps"])
    out["firm_exits"] = firm_exits(W, grid); print("firm exits", out["firm_exits"])
    out["events"] = events(P); print("events", out["events"])
    with open(os.path.join(RESULTS, "long_panel.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
