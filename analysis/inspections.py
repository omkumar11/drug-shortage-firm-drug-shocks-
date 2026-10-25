"""
inspections.py -- FDA inspection classifications as a firm-level quality signal (Section 3.5; asked for by the domain
referee in paper-v3-om rounds 2-3 and paper-v4 round 2).

Data: the FDA Data Dashboard "Entire Inspections Dataset" (data/raw/InspectionsDataset.xlsx; legal name, FEI number,
product type, project area, final classification NAI/VAI/OAI, inspection end date). The direct URL
(https://datadashboard.fda.gov/InspectionsDataset.xlsx) refuses requests made outside a browser session, so the file is
downloaded by hand and kept with the raw exports; this script writes a filtered extract of the drug inspections
(data/raw/inspections_drugs_extract.csv.gz) and reads the extract when the workbook is absent.

Linkage: each firm unit's normalised name is matched to the normalised legal names of inspected establishments by exact
match, by its corporate-group parent (analysis/common.py), or by a word-prefix match (EUGIA / EUGIA SPECIALITIES; MYLAN
INSTITUTIONAL / MYLAN). Inspections are restricted to the Drug Quality Assurance project area (the GMP surveillance
inspections) ending in the five years before the snapshot, the window used for the recall covariate.

Outputs (results/inspections.json): matched shares, cell unavailability by inspection status, logistic regressions with
"any OAI inspection" as a firm covariate (two-way clustered standard errors and a firm-cluster bootstrap), the association
between OAI status and the firm's posterior shock probability and unavailable share, and the model-free firm tests
without the OAI firms.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, normalise_company, CORPORATE_GROUP, RESULTS, PROC  # noqa: E402
from analysis.regressions import twoway_cluster_logit  # noqa: E402
from analysis.outcome_variants import model_free_row, SEED  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
XLSX = os.path.join(RAW, "InspectionsDataset.xlsx")
EXTRACT = os.path.join(RAW, "inspections_drugs_extract.csv.gz")
WINDOW_YEARS = 5
MIN_PREFIX_LEN = 4           # a word-prefix match needs at least four letters (so that short names match only exactly)
RULESETS = {"curated": "curated", "name": ("exact", "parent", "prefix")}
OUTSOURCING_503B = r"^NEPHRON SC\b|COMPOUNDING"       # section 503B outsourcing facilities among the linked establishments (sensitivity)
WINDOW_YEARS_SHORT = 4                                  # window starting after the pandemic pause of foreign inspections (sensitivity)   # primary: hand-curated establishment list; sensitivity: automatic name rules

# Hand-curated linkage (primary): for each reporting firm unit, regular expressions on the establishment's legal name as it
# appears in the FDA dataset, upper-cased with punctuation removed (so "Dr. Reddy's" is "DR REDDY S"). Built on 18 Oct 2026
# from the list of candidate establishments sharing a name token with the firm (scratch output kept in the hand-off), with
# documented ownership: Accord Healthcare's products are made by Intas; Eugia is Aurobindo's injectables arm but has its own
# establishments; the Mylan units are one group (Viatris); Prinston and Solco sell Zhejiang Huahai's products; Rhodes is
# affiliated with Purdue; SpecGx is Mallinckrodt's generics arm; Par Health is an Endo subsidiary; Novadoz is MSN's US arm;
# Teva's sites include Actavis (Barr, IVAX and Assia appear only in older records); Sun's US plant is Ohm Laboratories;
# Avet was Heritage Pharmaceuticals until 2020. Establishments of a different company
# that happen to share a token are excluded (Epic Laboratories of Tampa is not Epic Pharma of Laurelton; Gland Chemicals is not
# Gland Pharma; Fresenius Medical Care and SGS Institut Fresenius are not Fresenius Kabi; Zydus Hospira is a joint venture).
# Firms with no establishment in the dataset (marketers without plants, or plants under other names) get an empty list.
CURATED = {
    "ACCORD": [r"^ACCORD ", r"^INTAS PHARMACEUTICALS"], "AILEX": [r"^AILEX PHARMACEUTICALS"], "ALEMBIC": [r"^ALEMBIC PHARMACEUTICALS"],
    "ALVOGEN": [r"^NORWICH PHARMACEUTICALS"], "AMERICAN REGENT": [r"^AMERICAN REGENT"], "AMNEAL": [r"^AMNEAL "],
    "AMPHASTAR": [r"^AMPHASTAR", r"^INTERNATIONAL MEDICATION SYSTEMS"], "APOTEX": [r"^APOTEX "], "ASPIRO": [r"^ASPIRO PHARMA"],
    "AUROBINDO": [r"^AUROBINDO PHARMA"], "AVET": [r"^AVET PHARMACEUTICALS", r"^HERITAGE PHARMACEUTICALS"], "AZURITY": [r"^AZURITY"], "B BRAUN": [r"^B BRAUN"], "BAXTER": [r"^BAXTER ", r"^LABORATORIOS BAXTER"],
    "BRISTOL MYERS SQUIBB": [r"^BRISTOL MYERS SQUIBB"], "CAPLIN STERILES": [r"^CAPLIN STERILES"], "CHARTWELL MOLECULAR HOLDINGS": [r"^CHARTWELL PHARMACEUTICALS"],
    "CURIUM": [r"^CURIUM"], "DR REDDY S": [r"DR REDDY S"], "ELITE": [r"^ELITE LABORATORIES"], "EPIC": [r"^EPIC PHARMA "], "ESTEVE S A": [r"^ESTEVE "],
    "EUGIA": [r"^EUGIA "], "EXELA SCIENCES": [r"^EXELA PHARMA SCIENCES"], "FERRING": [r"^FERRING"], "FRESENIUS KABI": [r"^FRESENIUS KABI"],
    "GLAND": [r"^GLAND PHARMA"], "GRANULES": [r"^GRANULES"], "HAINAN POLY PHARM": [r"^HAINAN POLY PHARM"], "HIKMA": [r"^HIKMA"],
    "HOSPIRA": [r"^HOSPIRA"], "HUONS": [r"^HUONS"], "INFORLIFE SA": [r"^INFORLIFE"], "JANSSEN": [r"^JANSSEN (?!COSMETICS|VACCINES)"],
    "JIANGSU HENGRUI": [r"^JIANGSU HENGRUI"], "KINDOS": [r"^KINDOS"], "LANNETT": [r"^LANNETT"], "LONG GROVE": [r"^LONG GROVE PHARMACEUTICALS"],
    "LUPIN": [r"^LUPIN ", r"^NOVEL LABORATORIES INC D B A LUPIN"], "MEDEFIL": [r"^MEDEFIL"], "MICRO": [r"^MICRO LABS"],
    "MYLAN": [r"^MYLAN ", r"^VIATRIS", r"TRADING AS MYLAN"], "MYLAN INSTITUTIONAL": [r"^MYLAN ", r"^VIATRIS", r"TRADING AS MYLAN"],
    "MYLAN SPECIALTY": [r"^MYLAN ", r"^VIATRIS", r"TRADING AS MYLAN"], "NEPHRON": [r"^NEPHRON"], "NOVADOZ": [r"^MSN LABORATORIES", r"^MSN LIFE SCIENCES"],
    "NOVEN": [r"^NOVEN"], "NOVO NORDISK": [r"^NOVO NORDISK"], "ONESOURCE SPECIALTY PTE": [r"^ONESOURCE SPECIALTY"], "OTSUKA ICU": [r"^OTSUKA ICU MEDICAL"],
    "PADAGIS": [r"^PADAGIS"], "PAR HEALTH": [r"^PAR HEALTH", r"^PAR PHARMACEUTICAL", r"^PAR FORMULATIONS", r"^PAR ACTIVE", r"^PAR STERILE", r"^ENDO "],
    "PFIZER": [r"^PFIZER", r"^WYETH PHARMACEUTICALS"], "PIRAMAL CRITICAL CARE": [r"^PIRAMAL"], "PRINSTON": [r"^PRINSTON", r"^ZHEJIANG HUAHAI"],
    "RHODES L P": [r"^RHODES PHARMACEUTICALS", r"^RHODES TECHNOLOGIES", r"^PURDUE PHARMA"], "RISING HOLDINGS": [r"^RISING PHARMA"],
    "RUBICON RESEARCH PRIVATE": [r"^RUBICON RESEARCH"], "SAGENT": [r"^SAGENT"], "SANDOZ": [r"^SANDOZ"], "SANOFI AVENTIS U S": [r"^SANOFI", r"^AVENTIS "],
    "SOLCO": [r"^SOLCO", r"^ZHEJIANG HUAHAI"], "SOMERSET": [r"^SOMERSET THERAPEUTICS"], "SPECGX": [r"^SPECGX", r"^MALLINCKRODT"],
    "SPECTRA DEVICES": [r"^SPECTRA MEDICAL DEVICES"], "STERIMAX": [r"^STERIMAX"], "SUN": [r"^SUN PHARMA", r"^OHM LABORATORIES"],
    "SUNRISE": [r"^SUNRISE PHARMACEUTICAL"], "TAKEDA": [r"^TAKEDA"], "TEVA": [r"\bTEVA\b", r"^ACTAVIS"], "TEYRO": [r"^TEYRO"],
    "VERTICAL": [r"^VERTICAL PHARMACEUTICALS"], "XGEN DJB": [r"^XGEN PHARMACEUTICALS"],
}
KEEP = ["FEI Number", "Legal Name", "City", "State", "Country/Area", "Fiscal Year", "Inspection ID", "Inspection End Date",
        "Classification", "Project Area", "Product Type"]


def load_drug_inspections():
    """All inspections with product type 'Drugs', from the workbook if present (rewriting the extract) or from the extract."""
    if os.path.exists(XLSX):
        df = pd.read_excel(XLSX)
        n_total = int(len(df))
        d = df.loc[df["Product Type"] == "Drugs", KEEP].copy()
        d["Inspection End Date"] = pd.to_datetime(d["Inspection End Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        d.insert(0, "workbook_rows_total", n_total)
        d.to_csv(EXTRACT, index=False, compression="gzip")
    elif os.path.exists(EXTRACT):
        d = pd.read_csv(EXTRACT, dtype={"FEI Number": str, "Inspection ID": str})
        n_total = int(d["workbook_rows_total"].iloc[0])
    else:
        raise FileNotFoundError("neither InspectionsDataset.xlsx nor inspections_drugs_extract.csv.gz is in data/raw")
    d["end"] = pd.to_datetime(d["Inspection End Date"], errors="coerce")
    d["norm"] = d["Legal Name"].astype(str).map(normalise_company)
    d["raw"] = d["Legal Name"].astype(str).str.upper().str.replace(r"[^A-Z0-9 &]", " ", regex=True).str.split().str.join(" ")
    d["class3"] = d["Classification"].astype(str).str.extract(r"\((NAI|VAI|OAI)\)")[0]
    return d, n_total


def match_firm(norm, inspected, group_map=CORPORATE_GROUP, rules=RULESETS["name"]):
    """Inspected legal names (normalised) that belong to the firm unit `norm`, with the rules that matched them.
    The three rules are applied as a union (a firm can match through all of them): exact name; the firm's corporate-group
    parent (exact or as a word prefix), so that Accord's establishments and Intas's are one firm; and a word prefix either
    way (the firm name is the first word(s) of the legal name, or the legal name is the first word(s) of the firm name),
    only when the shorter side has at least MIN_PREFIX_LEN letters."""
    if not norm:
        return set(), None
    hits, used = set(), []
    if "exact" in rules and norm in inspected:
        hits.add(norm); used.append("exact")
    parent = group_map.get(norm)
    if "parent" in rules and parent:
        ph = {x for x in inspected if x == parent or x.startswith(parent + " ")}
        if ph:
            hits |= ph; used.append("parent")
    if "prefix" in rules and len(norm) >= MIN_PREFIX_LEN:
        pf = {x for x in inspected if len(x) >= MIN_PREFIX_LEN and (x.startswith(norm + " ") or norm.startswith(x + " "))}
        if pf:
            hits |= pf; used.append("prefix")
    return (hits, "+".join(used)) if hits else (set(), None)


def match_firm_first_hit(norm, inspected, group_map=CORPORATE_GROUP):
    """The precedence version used in an earlier draft (first rule that hits wins); kept to report how many firms differ."""
    if not norm:
        return set()
    if norm in inspected:
        return {norm}
    parent = group_map.get(norm)
    if parent:
        ph = {x for x in inspected if x == parent or x.startswith(parent + " ")}
        if ph:
            return ph
    if len(norm) >= MIN_PREFIX_LEN:
        return {x for x in inspected if len(x) >= MIN_PREFIX_LEN and (x.startswith(norm + " ") or norm.startswith(x + " "))}
    return set()


def firm_status(cells, insp, snapshot, rules=RULESETS["curated"], exclude=None, window_years=WINDOW_YEARS):
    """One row per firm unit: matched legal names, drug-quality inspections in the window, OAI / VAI counts, FEIs, the
    date of the first OAI in the window, and whether the union rule changed the match relative to the first-hit rule."""
    lo = snapshot - pd.DateOffset(years=window_years)
    dqa = insp[insp["Project Area"] == "Drug Quality Assurance"]
    if exclude:
        dqa = dqa[~dqa["raw"].str.contains(exclude, regex=True)]
    w = dqa[(dqa["end"] > lo) & (dqa["end"] <= snapshot)]
    names_all = set(insp["norm"]); rows = []
    by_name = w.groupby("norm"); by_name_all = dqa.groupby("norm")
    raw_all = dqa["raw"].unique()
    for f in sorted(cells["firm_unit"].unique()):
        key = str(f)
        if rules == "curated":
            pats = CURATED.get(key, [])
            hits = {r for r in raw_all if any(re.search(p_, r) for p_ in pats)}
            rule = "curated" if hits else ("curated-nomatch" if pats else "")
            first = hits
            sub = w[w["raw"].isin(hits)]; sub_all = dqa[dqa["raw"].isin(hits)]
        else:
            hits, rule = match_firm(key, names_all, rules=rules)
            first = match_firm_first_hit(key, names_all)
            parts = [by_name.get_group(h) for h in hits if h in by_name.groups]
            sub = pd.concat(parts) if parts else w.iloc[0:0]
            parts_all = [by_name_all.get_group(h) for h in hits if h in by_name_all.groups]
            sub_all = pd.concat(parts_all) if parts_all else dqa.iloc[0:0]          # every drug-quality inspection, any date
        oai_dates = sub.loc[sub["class3"] == "OAI", "end"]
        rows.append(dict(firm_unit=f, matched=int(bool(hits)), match_rule=rule or "", n_legal_names=len(hits),
                         legal_names="; ".join(sorted(hits)), n_insp=int(len(sub)), n_fei=int(sub["FEI Number"].nunique()) if len(sub) else 0,
                         n_oai=int((sub["class3"] == "OAI").sum()), n_vai=int((sub["class3"] == "VAI").sum()), n_nai=int((sub["class3"] == "NAI").sum()),
                         n_foreign=int((sub["Country/Area"] != "United States").sum()) if len(sub) else 0,
                         first_oai=str(oai_dates.min().date()) if len(oai_dates) else "",
                         oai_dates_all=";".join(sorted(str(d.date()) for d in sub_all.loc[sub_all["class3"] == "OAI", "end"])),
                         changed_vs_first_hit=int(hits != first)))
    fs = pd.DataFrame(rows)
    fs["inspected_any"] = (fs["n_insp"] > 0).astype(int)
    fs["oai_any"] = (fs["n_oai"] > 0).astype(int)
    fs["vai_oai_any"] = ((fs["n_oai"] + fs["n_vai"]) > 0).astype(int)
    fs["oai_share"] = np.where(fs["n_insp"] > 0, fs["n_oai"] / fs["n_insp"].clip(lower=1), 0.0)
    fs["log1p_insp"] = np.log1p(fs["n_insp"]); fs["log1p_fei"] = np.log1p(fs["n_fei"])
    return fs, w


def coefs(r, se, X):
    return {c: dict(coef=float(r.params.iloc[i]), se=float(se[i]), or_=float(np.exp(r.params.iloc[i])),
                    or_lo=float(np.exp(r.params.iloc[i] - 1.96 * se[i])), or_hi=float(np.exp(r.params.iloc[i] + 1.96 * se[i])),
                    p=float(2 * (1 - stats.norm.cdf(abs(r.params.iloc[i] / se[i]))))) for i, c in enumerate(X.columns)}


def regressions(d, specs, rng, boot_var="oai_any", n_boot=500, y_col="unavailable"):
    out = {}
    for name, cols in specs.items():
        X = sm.add_constant(d[cols].astype(float)); y = d[y_col].astype(float)
        r, se, min_eig = twoway_cluster_logit(y, X, d["firm_unit"].values, d["drug"].values)
        out[name] = dict(coefs=coefs(r, se, X), n=int(r.nobs), min_eigenvalue=min_eig, n_firms=int(d["firm_unit"].nunique()), n_drugs=int(d["drug"].nunique()))
    cols = specs[next(iter(specs))]; firms_u = d["firm_unit"].unique(); boots = []
    for _ in range(n_boot):
        pick = rng.choice(firms_u, len(firms_u), replace=True)
        db = pd.concat([d[d["firm_unit"] == f] for f in pick])
        if db[boot_var].nunique() < 2:
            continue
        try:
            rb = sm.Logit(db[y_col].astype(float), sm.add_constant(db[cols].astype(float))).fit(disp=0, maxiter=200)
            boots.append(float(rb.params[boot_var]))
        except Exception:
            pass
    boots = np.array(boots); boots = boots[np.isfinite(boots)]
    if len(boots):
        out["bootstrap"] = dict(variable=boot_var, n_boot=int(len(boots)), or_lo=float(np.exp(np.percentile(boots, 2.5))), or_hi=float(np.exp(np.percentile(boots, 97.5))))
    return out


def firm_permutation_p(d, cols, var, rng, n_perm=2000, strata=None):
    """Firm-level permutation test of a firm covariate's logistic coefficient: the covariate's firm values are reshuffled
    across firms (every cell of a firm keeps the same value), the plain logistic regression is refitted, and the two-sided
    p-value is the share of permuted |coefficients| at least as large as the observed one. This respects the small number of
    treated firms, which the cluster-robust intervals do not."""
    X = sm.add_constant(d[cols].astype(float)); y = d["unavailable"].astype(float)
    obs = abs(float(sm.Logit(y, X).fit(disp=0, maxiter=200).params[var]))
    fv = d.groupby("firm_unit")[var].first(); firms_u = fv.index.values; vals = fv.values.astype(float); cnt = 0; done = 0
    st = d.groupby("firm_unit")[strata].first().reindex(firms_u).values if strata else np.zeros(len(firms_u))
    idx_by_stratum = [np.flatnonzero(st == u) for u in np.unique(st)]
    for _ in range(n_perm):
        pv = vals.copy()
        for ix in idx_by_stratum:                                       # reshuffle the firm values within each stratum
            pv[ix] = vals[ix][rng.permutation(len(ix))]
        perm = dict(zip(firms_u, pv))
        Xp = X.copy(); Xp[var] = d["firm_unit"].map(perm).values
        try:
            b = abs(float(sm.Logit(y, Xp).fit(disp=0, maxiter=200).params[var])); done += 1; cnt += int(b >= obs - 1e-12)
        except Exception:
            pass
    return dict(p=float((cnt + 1) / (done + 1)), n_perm=int(done), observed_abs_coef=obs)


SUFFIX_WORDS = r"\b(LIMITED|LTD|INC|INCORPORATED|LLC|L L C|CORPORATION|CORP|CO|COMPANY|PRIVATE|PVT|GMBH|AG|SA|S A|S R L|SRL|SPA|NV|N V|B V|BV|PTY|ULC|UNLIMITED|USA|US|LP|L P|D O O|AKTIENGESELLSCHAFT)\b"


def write_oai_tables(fs, d, label):
    """Supplementary tables of the OAI firms with their establishments, split in two halves so that each fits a page; names
    are shown as in the FDA dataset with legal-form words removed and duplicates collapsed, plus the establishment count."""
    rows = []
    for _, r_ in fs[fs["oai_any"] == 1].sort_values(["n_oai", "firm_unit"], ascending=[False, True]).iterrows():
        g = d[d["firm_unit"] == r_["firm_unit"]]
        names = []
        for nm in str(r_["legal_names"]).split("; "):
            short = " ".join(re.sub(SUFFIX_WORDS, " ", nm).split()).title()
            if short and short not in names:
                names.append(short)
        names_ = "; ".join(names).replace("&", "\\&")
        rows.append(f"{str(r_['firm_unit']).title()} & {names_} & {r_['n_insp']} & {r_['n_fei']} & {r_['n_oai']} & {len(g)} & {int(g['unavailable'].sum())}\\\\")
    half = (len(rows) + 1) // 2
    for part, chunk in (("a", rows[:half]), ("b", rows[half:])):
        with open(os.path.join(RESULTS, f"oai_firms_table_{label}_{part}.tex"), "w") as f:
            f.write("\n".join(chunk) + "\n\\bottomrule\n")
    with open(os.path.join(RESULTS, f"oai_firms_table_{label}.tex"), "w") as f:      # the whole list in one file as well
        f.write("\n".join(rows) + "\n\\bottomrule\n")


def tables_only():
    """Rewrite the supplementary OAI tables from the saved firm tables without rerunning the analysis."""
    cells, _, _, _ = load_cells("company")
    for label in RULESETS:
        fs = pd.read_csv(os.path.join(RESULTS, f"firm_inspections_{label}.csv"), keep_default_na=False)
        write_oai_tables(fs, cells.merge(fs, on="firm_unit", how="left"), label)
    print("tables rewritten")


def set_permutation_p(d, cols, var, rng, n_perm=2000):
    """Like firm_permutation_p, but the unit reshuffled is the establishment set: firms that share the same linked
    establishments (the three Mylan units; Prinston and Solco) move together."""
    X = sm.add_constant(d[cols].astype(float)); y = d["unavailable"].astype(float)
    obs = abs(float(sm.Logit(y, X).fit(disp=0, maxiter=200).params[var]))
    sv = d.groupby("estab_set")[var].first(); units = sv.index.values; vals = sv.values.astype(float); cnt = 0; done = 0
    for _ in range(n_perm):
        perm = dict(zip(units, vals[rng.permutation(len(vals))]))
        Xp = X.copy(); Xp[var] = d["estab_set"].map(perm).values
        try:
            b = abs(float(sm.Logit(y, Xp).fit(disp=0, maxiter=200).params[var])); done += 1; cnt += int(b >= obs - 1e-12)
        except Exception:
            pass
    return dict(p=float((cnt + 1) / (done + 1)), n_perm=int(done), n_units=int(len(units)))


def analyse(cells, pres, firms, fs, snapshot, rng, fp, label):
    """Everything reported for one linkage rule set: status rates, regressions, firm-level comparisons, clustering within
    and without the OAI firms, timing, concentration. Returns a dict."""
    out = {}
    d = cells.merge(fs, on="firm_unit", how="left")
    cell_n = d.groupby("firm_unit")["unavailable"].size()
    for key, mask in (("matched", fs["matched"] == 1), ("inspected", fs["inspected_any"] == 1), ("oai", fs["oai_any"] == 1), ("vai_oai", fs["vai_oai_any"] == 1)):
        ff = fs.loc[mask, "firm_unit"]
        out[f"firms_{key}"] = int(mask.sum()); out[f"cells_{key}"] = int(cell_n.reindex(ff).fillna(0).sum())
    out["match_rules"] = fs.loc[fs["matched"] == 1, "match_rule"].value_counts().to_dict()
    out["firms_matched_not_inspected_in_window"] = int(((fs["matched"] == 1) & (fs["inspected_any"] == 0)).sum())
    out["firms_oai_list"] = fs.loc[fs["oai_any"] == 1, ["firm_unit", "n_oai", "n_insp", "n_fei", "match_rule", "legal_names"]].to_dict("records")
    out["unmatched_firms_by_cells"] = d.loc[d["matched"] == 0].groupby("firm_unit").size().sort_values(ascending=False).head(12).to_dict()
    out["firms_changed_by_union_rule"] = int(fs["changed_vs_first_hit"].sum())
    out["firms_curated_no_match"] = int((fs["match_rule"] == "curated-nomatch").sum()); out["firms_curated_no_pattern"] = int((fs["match_rule"] == "").sum())
    out["oai_firms_mean_cells"] = float(cell_n.reindex(fs.loc[fs["oai_any"] == 1, "firm_unit"]).mean()); out["other_firms_mean_cells"] = float(cell_n.reindex(fs.loc[fs["oai_any"] == 0, "firm_unit"]).mean())
    out["oai_firms_fei_max"] = int(fs.loc[fs["oai_any"] == 1, "n_fei"].max()) if out["firms_oai"] else 0
    # cell unavailability by firm inspection status, for the main, strict, Unavailable-only and no-discontinuation outcomes
    status = np.where(d["oai_any"] == 1, "oai", np.where(d["inspected_any"] == 1, "inspected_no_oai", np.where(d["matched"] == 1, "matched_no_window_inspection", "unmatched")))
    d["insp_status"] = status
    out["cell_rate_by_status"] = {k: dict(n=int(len(g)), n_unavailable=int(g["unavailable"].sum()), rate=float(g["unavailable"].mean()),
                                          n_firms=int(g["firm_unit"].nunique()),
                                          firm_weighted_rate=float(g.groupby("firm_unit")["unavailable"].mean().mean()))   # mean of the firms' own shares
                                  for k, g in d.groupby("insp_status")}
    out["cell_strict_rate_by_status"] = {k: float(g["all_unavailable"].mean()) for k, g in d.groupby("insp_status")}
    out["cell_uo_rate_by_status"] = {k: float(g["unavailable_only"].mean()) for k, g in d.groupby("insp_status")}
    out["cell_nodisc_rate_by_status"] = {k: float(g["unavailable_nodisc"].mean()) for k, g in d.groupby("insp_status")}
    # concentration of the OAI cells in a few firms, and the gradient of "any OAI" in the number of inspections
    oc = d[d["oai_any"] == 1].groupby("firm_unit").size().sort_values(ascending=False)
    out["oai_cells_top3"] = dict(firms=list(oc.index[:3]), cells=int(oc.iloc[:3].sum()), of=int(oc.sum()))
    fi = fs[fs["inspected_any"] == 1]
    if len(fi) >= 6:
        terc = pd.qcut(fi["n_insp"].rank(method="first"), 3, labels=["low", "mid", "high"])
        out["oai_rate_by_insp_tercile"] = {str(k): dict(n_firms=int(len(g)), oai_share=float(g["oai_any"].mean()), median_insp=float(g["n_insp"].median())) for k, g in fi.groupby(terc, observed=True)}
    # timing: an OAI inspection ending before the cell's first posting (and within five years before it), the analogue of the
    # pre-posting recall variant; and how the OAI firms' unavailable cells sit relative to their first OAI in the window
    first_post = pd.to_datetime(d["first_posting"], errors="coerce")
    dates = fs.set_index("firm_unit")["oai_dates_all"].map(lambda x: [pd.Timestamp(t) for t in str(x).split(";") if t and t != "nan"])
    def pre_posting(f, t0):
        if pd.isna(t0):
            return 0
        return int(any((t < t0) and (t >= t0 - pd.DateOffset(years=WINDOW_YEARS)) for t in dates.get(f, [])))
    d["oai_pre_posting"] = [pre_posting(f, t0) for f, t0 in zip(d["firm_unit"], first_post)]
    out["cells_oai_pre_posting"] = int(d["oai_pre_posting"].sum()); out["cells_oai_pre_posting_unavail"] = int(d.loc[d["oai_pre_posting"] == 1, "unavailable"].sum())
    out["firms_oai_pre_posting"] = int(d.loc[d["oai_pre_posting"] == 1, "firm_unit"].nunique())
    out["rate_oai_pre_posting"] = float(d.loc[d["oai_pre_posting"] == 1, "unavailable"].mean()) if out["cells_oai_pre_posting"] else float("nan")
    out["rate_not_oai_pre_posting"] = float(d.loc[d["oai_pre_posting"] == 0, "unavailable"].mean())
    fo = fs.set_index("firm_unit")["first_oai"].map(lambda x: pd.Timestamp(x) if x else pd.NaT)
    u = d[(d["oai_any"] == 1) & (d["unavailable"] == 1)]
    fp_u = pd.to_datetime(u["first_posting"], errors="coerce"); first_oai_u = u["firm_unit"].map(fo)
    lo = snapshot - pd.DateOffset(years=WINDOW_YEARS)
    out["oai_unavail_cells"] = int(len(u)); out["oai_unavail_cells_posted_before_first_oai"] = int((fp_u < first_oai_u).sum())
    out["oai_unavail_cells_posted_before_window"] = int((fp_u < lo).sum())
    # regressions: the firm specification of Table S5 column (2) with OAI in place of, and alongside, the recall indicator;
    # with the number of inspections, the OAI share, the pre-posting indicator, listed drugs as the size control; and the
    # strict, Unavailable-only and no-discontinuation outcomes
    d["log_apps"] = np.log(d["n_apps_in_market"].clip(lower=1)); d["log_pres"] = np.log(d["n_presentations"])
    d["log_firm_markets"] = np.log(d["firm_n_markets"].clip(lower=1)); d["recall_any"] = (d["firm_recall_events_5y"] > 0).astype(int)
    d["log_listed_drugs"] = np.log(d.groupby("firm_unit")["drug"].transform("size"))
    d["insp_stratum"] = np.where(d["n_insp"] == 0, 0, np.where(d["n_insp"] <= 3, 1, np.where(d["n_insp"] <= 9, 2, 3)))
    dr = d.dropna(subset=["log_apps"]).copy()
    base = ["injectable", "controlled", "log_apps", "log_pres"]
    specs = {"m11_oai": base + ["oai_any", "log_firm_markets"],
             "m12_oai_inspected": base + ["oai_any", "inspected_any", "log_firm_markets"],
             "m13_oai_recall": base + ["oai_any", "recall_any", "log_firm_markets"],
             "m14_vai_oai": base + ["vai_oai_any", "log_firm_markets"],
             "m16_oai_share": base + ["oai_share", "log1p_insp", "log_firm_markets"],
             "m17_oai_pre_posting": base + ["oai_pre_posting", "log_firm_markets"],
             "m21_oai_listed": base + ["oai_any", "log_listed_drugs"],
             "m22_oai_fei": base + ["oai_any", "log1p_fei", "log_firm_markets"]}
    rg = regressions(dr, specs, rng)
    rg.update(regressions(dr, {"m15_oai_ninsp": base + ["oai_any", "log1p_insp", "log_firm_markets"]}, rng, n_boot=0))
    for y_col, nm in (("all_unavailable", "m18_oai_strict"), ("unavailable_only", "m19_oai_uo"), ("unavailable_nodisc", "m20_oai_nodisc")):
        rg[nm] = regressions(dr, {nm: specs["m11_oai"]}, rng, n_boot=0, y_col=y_col)[nm]
    rg["n"] = int(len(dr)); rg["n_firms_oai"] = int(dr.loc[dr["oai_any"] == 1, "firm_unit"].nunique()); rg["n_cells_oai"] = int((dr["oai_any"] == 1).sum())
    rg["firm_perm_oai"] = firm_permutation_p(dr, specs["m11_oai"], "oai_any", rng)
    rg["firm_perm_oai_strat"] = firm_permutation_p(dr, specs["m11_oai"], "oai_any", rng, strata="insp_stratum")   # within inspection-count strata
    # firms sharing one establishment set (the Mylan units; Prinston and Solco) are one unit for the permutation
    sets = fs.set_index("firm_unit")["legal_names"].astype(str)
    dr["estab_set"] = dr["firm_unit"].map(sets).where(dr["firm_unit"].map(sets) != "", dr["firm_unit"])
    rg["firm_perm_oai_sets"] = set_permutation_p(dr, specs["m11_oai"], "oai_any", rng)
    out["n_establishment_sets_oai"] = int(fs.loc[fs["oai_any"] == 1, "legal_names"].nunique()); out["n_establishment_sets"] = int(fs.loc[fs["matched"] == 1, "legal_names"].nunique())
    rg["firm_perm_recall"] = firm_permutation_p(dr, base + ["recall_any", "log_firm_markets"], "recall_any", rng)
    out["regressions"] = rg
    out["recall_oai_overlap"] = dict(firms_both=int(((fs["oai_any"] == 1) & (fs["firm_unit"].isin(firms.loc[firms["firm_recall_events_5y"] > 0, "firm_unit"]))).sum()),
                                     firms_recall=int((firms["firm_recall_events_5y"] > 0).sum()))
    # firm level: unavailable share (and the model's shock posterior, which is a monotone function of the firm's unavailable count
    # and therefore restates the share comparison) by OAI status
    fl = fs.merge(fp.rename(columns={"firm": "firm_unit"}), on="firm_unit", how="left")
    fl["share_unavail"] = fl["n_unavailable"] / fl["n_drugs"]
    a = fl[fl["oai_any"] == 1]; b = fl[(fl["oai_any"] == 0)]
    def mw(x, y):
        x = x.dropna(); y = y.dropna()
        if len(x) == 0 or len(y) == 0:
            return dict(n_a=int(len(x)), n_b=int(len(y)))
        uu, p = stats.mannwhitneyu(x, y, alternative="greater")
        return dict(n_a=int(len(x)), n_b=int(len(y)), mean_a=float(x.mean()), mean_b=float(y.mean()), median_a=float(x.median()), median_b=float(y.median()),
                    auc=float(uu / (len(x) * len(y))), p_one_sided=float(p))
    out["firm_zpost_by_oai"] = mw(a["z_post"], b["z_post"]); out["firm_share_by_oai"] = mw(a["share_unavail"], b["share_unavail"])
    out["firm_share_by_oai_multi"] = mw(a.loc[a["n_drugs"] >= 2, "share_unavail"], b.loc[b["n_drugs"] >= 2, "share_unavail"])
    # clustering within the OAI firms and within the other firms (the comparison the "without OAI firms" test cannot make),
    # and the model-free firm tests without the OAI firms and without every firm with a VAI or OAI
    for key, mask in (("within_oai_firms", fs["oai_any"] == 1), ("within_non_oai_firms", fs["oai_any"] == 0),
                      ("no_oai_firms", fs["oai_any"] == 0), ("no_vai_oai_firms", fs["vai_oai_any"] == 0)):
        keep = set(fs.loc[mask, "firm_unit"]); sub = cells[cells["firm_unit"].isin(keep)].reset_index(drop=True)
        if key.startswith("within") and (sub["firm_unit"].nunique() < 3 or sub["unavailable"].nunique() < 2):
            out[key] = dict(n_cells=int(len(sub))); continue
        out[key] = model_free_row(sub, sub["unavailable"].values, rng)
        out[key].update(n_firms=int(sub["firm_unit"].nunique()), n_firms_dropped=int(len(fs) - len(keep)))
    out["no_oai_firms"] = out["within_non_oai_firms"]                      # the same subset, named for the text
    write_oai_tables(fs, d, label)
    fs.to_csv(os.path.join(RESULTS, f"firm_inspections_{label}.csv"), index=False)
    return out


def main():
    rng = np.random.default_rng(SEED + 1300)
    build = json.load(open(os.path.join(PROC, "build_summary.json")))
    snapshot = pd.Timestamp(build["shortage_snapshot_date"])
    insp, n_total = load_drug_inspections()
    out = dict(workbook_rows_total=n_total, drug_rows=int(len(insp)),
               drug_rows_dqa=int((insp["Project Area"] == "Drug Quality Assurance").sum()),
               end_date_min=str(insp["end"].min().date()), end_date_max=str(insp["end"].max().date()),
               fiscal_year_min=int(insp["Fiscal Year"].min()), fiscal_year_max=int(insp["Fiscal Year"].max()),
               window_start=str((snapshot - pd.DateOffset(years=WINDOW_YEARS)).date()), window_end=str(snapshot.date()), window_years=WINDOW_YEARS)
    cells, pres, drugs, firms = load_cells("company")
    fp = pd.read_csv(os.path.join(RESULTS, "firm_posteriors.csv"))
    out["n_firms"] = int(cells["firm_unit"].nunique()); out["n_cells"] = int(len(cells))
    for label, rules in RULESETS.items():
        fs, w = firm_status(cells, insp, snapshot, rules)
        if label == "curated":
            out["window_rows"] = int(len(w)); out["window_legal_names"] = int(w["norm"].nunique()); out["window_fei"] = int(w["FEI Number"].nunique())
            out["window_oai_rows"] = int((w["class3"] == "OAI").sum()); out["window_vai_rows"] = int((w["class3"] == "VAI").sum())
            out["window_share_foreign"] = float((w["Country/Area"] != "United States").mean())
        out[label] = analyse(cells, pres, firms, fs, snapshot, rng, fp, label)
        out[label]["rules"] = list(rules) if isinstance(rules, tuple) else [rules]
        if label == "curated":
            for key, kw in (("no_503b", dict(exclude=OUTSOURCING_503B)), ("short_window", dict(window_years=WINDOW_YEARS_SHORT))):
                fs2, w2 = firm_status(cells, insp, snapshot, rules, **kw)
                d2 = cells.merge(fs2, on="firm_unit", how="left")
                d2["log_apps"] = np.log(d2["n_apps_in_market"].clip(lower=1)); d2["log_pres"] = np.log(d2["n_presentations"]); d2["log_firm_markets"] = np.log(d2["firm_n_markets"].clip(lower=1))
                d2r = d2.dropna(subset=["log_apps"]).copy()
                X = sm.add_constant(d2r[["injectable", "controlled", "log_apps", "log_pres", "oai_any", "log_firm_markets"]].astype(float))
                r, se, me = twoway_cluster_logit(d2r["unavailable"].astype(float), X, d2r["firm_unit"].values, d2r["drug"].values)
                st = d2.groupby(d2["oai_any"].map({1: "oai", 0: "other"}))["unavailable"].mean()
                out[label][key] = dict(firms_oai=int(fs2["oai_any"].sum()), cells_oai=int(d2.loc[d2["oai_any"] == 1].shape[0]), firms_inspected=int(fs2["inspected_any"].sum()),
                                       window_rows=int(len(w2)), rate_oai=float(st.get("oai", float("nan"))), rate_other=float(st.get("other", float("nan"))),
                                       m11_oai=dict(coefs=coefs(r, se, X), n=int(r.nobs), min_eigenvalue=me),
                                       excluded_establishments=int(insp["raw"].str.contains(OUTSOURCING_503B, regex=True).sum()) if key == "no_503b" else 0)
        # marketer of record as the firm unit: matched share and the OAI regression
        mk, _, _, _ = load_cells("marketer")
        fsm, _ = firm_status(mk, insp, snapshot, rules)
        dm = mk.merge(fsm, on="firm_unit", how="left"); cell_nm = dm.groupby("firm_unit")["unavailable"].size()
        mkt = dict(n_firms=int(len(fsm)), firms_matched=int(fsm["matched"].sum()), firms_oai=int(fsm["oai_any"].sum()),
                   cells_matched=int(cell_nm.reindex(fsm.loc[fsm["matched"] == 1, "firm_unit"]).fillna(0).sum()),
                   cells_oai=int(cell_nm.reindex(fsm.loc[fsm["oai_any"] == 1, "firm_unit"]).fillna(0).sum()), n_cells=int(len(dm)))
        dm["log_apps"] = np.log(dm["n_apps_in_market"].clip(lower=1)); dm["log_pres"] = np.log(dm["n_presentations"]); dm["log_firm_markets"] = np.log(dm["firm_n_markets"].clip(lower=1))
        dmr = dm.dropna(subset=["log_apps"]).copy()
        X = sm.add_constant(dmr[["injectable", "controlled", "log_apps", "log_pres", "oai_any", "log_firm_markets"]].astype(float))
        r, se, me = twoway_cluster_logit(dmr["unavailable"].astype(float), X, dmr["firm_unit"].values, dmr["drug"].values)
        mkt["m11_oai"] = dict(coefs=coefs(r, se, X), n=int(r.nobs), min_eigenvalue=me)
        out[label]["marketer"] = mkt
    # the first-hit precedence rule of the earlier draft, for the record (how many firms the union rule changed)
    with open(os.path.join(RESULTS, "inspections.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for label in RULESETS:
        o = out[label]; rg = o["regressions"]
        print(f"[{label}] matched {o['firms_matched']}/{out['n_firms']} firms, OAI firms {o['firms_oai']} ({o['cells_oai']} cells); rates",
              {k: round(v["rate"], 3) for k, v in o["cell_rate_by_status"].items()})
        for k in ("m11_oai", "m15_oai_ninsp", "m21_oai_listed", "m17_oai_pre_posting", "m18_oai_strict", "m19_oai_uo", "m20_oai_nodisc"):
            v = next((c for c in rg[k]["coefs"] if c.startswith("oai")), None); c = rg[k]["coefs"][v]
            print(f"   {k}: OR {c['or_']:.2f} ({c['or_lo']:.2f}-{c['or_hi']:.2f})")
        print("   firm-permutation p (OAI, recall):", round(rg["firm_perm_oai"]["p"], 3), round(rg["firm_perm_recall"]["p"], 3), "| bootstrap", {k: round(v, 2) for k, v in rg["bootstrap"].items() if isinstance(v, float)})
        print("   share by OAI:", {k: round(v, 3) for k, v in o["firm_share_by_oai"].items() if isinstance(v, float)})
        for k in ("within_oai_firms", "within_non_oai_firms", "no_vai_oai_firms"):
            print(f"   {k}:", {kk: round(o[k][kk], 3) for kk in ("n_cells", "disp_firm_ratio", "perm_firm_p_size", "cov_firm", "perm_cov_firm_p_size") if kk in o[k]})
        print("   timing: unavailable OAI cells", o["oai_unavail_cells"], "posted before first OAI", o["oai_unavail_cells_posted_before_first_oai"], "before window", o["oai_unavail_cells_posted_before_window"],
              "| pre-posting OAI cells", o["cells_oai_pre_posting"], "rate", round(o["rate_oai_pre_posting"], 3), "vs", round(o["rate_not_oai_pre_posting"], 3))
        print("   top3", o["oai_cells_top3"], "| marketer OAI firms", o["marketer"]["firms_oai"], "OR", round(o["marketer"]["m11_oai"]["coefs"]["oai_any"]["or_"], 2))


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        tables_only()
    else:
        main()
