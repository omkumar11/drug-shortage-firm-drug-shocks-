"""Shared loaders and settings for the analysis scripts."""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed"); RESULTS = os.path.join(ROOT, "results"); FIGS = os.path.join(ROOT, "figures")

# Primary firm unit: the reporting company named in the FDA record, normalised to a corporate name.
# Robustness: FIRM_UNIT = "labeler" uses the NDC labeler code (finer); FIRM_UNIT = "marketer" uses the normalised NDC
# labeler name, i.e. the company whose NDC the presentation carries, with all of its labeler codes merged (the entity a
# purchaser contracts with); FIRM_UNIT = "group" merges subsidiaries into their parent groups (coarser) using the
# hand-made map below.
FIRM_UNIT = os.environ.get("FIRM_UNIT", "company")

# Subsidiary -> parent corporate group, for reporting companies on the list whose parent is a different listed or
# well-known company. Only widely documented ownership relations are included; everything else keeps its own name.
CORPORATE_GROUP = {
    "HOSPIRA": "PFIZER",                      # Hospira acquired by Pfizer (2015)
    "EUGIA": "AUROBINDO",                     # Eugia Pharma is Aurobindo's injectables arm
    "MYLAN": "VIATRIS", "MYLAN INSTITUTIONAL": "VIATRIS", "MYLAN SPECIALTY": "VIATRIS",
    "PAR HEALTH": "ENDO",                     # Par Pharmaceutical / Par Health is an Endo subsidiary
    "SPECGX": "MALLINCKRODT",                 # SpecGx is Mallinckrodt's generics subsidiary
    "ACCORD": "INTAS",                        # Accord Healthcare is Intas's US/EU arm
    "RHODES L P": "PURDUE",                   # Rhodes Pharmaceuticals is affiliated with Purdue
    "SOLCO": "HUAHAI", "PRINSTON": "HUAHAI",  # Solco and Prinston are Zhejiang Huahai subsidiaries
    "SAGENT": "NICHI-IKO",                    # Sagent acquired by Nichi-Iko (2016)
    "GLAND": "FOSUN",                         # Gland Pharma majority-owned by Fosun Pharma
}

# Manufacturer-reported reasons grouped a priori (used only for validation, never for fitting)
FIRM_REASONS = {"Requirements related to complying with good manufacturing practices",
                "Discontinuation of the manufacture of the drug", "Delay in shipping of the drug",
                "Shortage of an inactive ingredient component", "Regulatory delay"}
MARKET_REASONS = {"Demand increase for the drug", "Shortage of an active ingredient"}


def normalise_company(name):
    n = name.upper()
    n = re.sub(r",?\s*A (PFIZER|VIATRIS) COMPANY", "", n)
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"[^A-Z ]", " ", n)
    n = re.sub(r"\b(INC|LLC|LTD|CORPORATION|CORP|CO|USA|US|LIMITED|PHARMACEUTICALS?|PHARMA|HEALTHCARE|MEDICAL|LABORATORIES|LABS|THERAPEUTICS|INDUSTRIES|COMPANY|AMERICA|NORTH)\b", " ", n)
    return " ".join(n.split())


# notes that describe an exit without the discontinuation reason code ("Currently not marketed", "Not currently
# manufactured", "No longer manufactured", "Discontinued product"); a presentation is discontinued if either says so
DISCONTINUED_NOTE_PATTERN = r"not (?:currently )?marketed|not currently manufactured|no longer manufactured|discontinued product"


def load_tbd():
    """The 'To Be Discontinued' records of listed drugs (pipeline/build_data.py), with the firm unit of load_cells()."""
    t = pd.read_csv(os.path.join(PROC, "presentations_tbd.csv"))
    t["company_norm"] = t["company_name"].map(normalise_company)
    t["company_group"] = t["company_norm"].map(lambda n: CORPORATE_GROUP.get(n, n))
    t["firm_unit"] = {"company": t["company_norm"], "labeler": t["firm"].astype(str), "group": t["company_group"], "raw": t["company_name"]}[FIRM_UNIT]
    return t


def load_cells(firm_unit=None):
    """Presentations aggregated to firm-drug cells. firm_unit overrides FIRM_UNIT ("company", "labeler", "marketer",
    "group" or "raw")."""
    firm_unit = firm_unit or FIRM_UNIT
    pres = pd.read_csv(os.path.join(PROC, "presentations.csv"))
    pres["shortage_reason"] = pres["shortage_reason"].fillna("")
    pres["company_norm"] = pres["company_name"].map(normalise_company)
    pres["company_group"] = pres["company_norm"].map(lambda n: CORPORATE_GROUP.get(n, n))
    pres["marketer_norm"] = pres["firm_name"].fillna(pres["company_name"]).astype(str).map(normalise_company)
    pres["firm_unit"] = {"company": pres["company_norm"], "labeler": pres["firm"].astype(str), "marketer": pres["marketer_norm"],
                         "group": pres["company_group"], "raw": pres["company_name"]}[firm_unit]
    # outcome variant: an Available presentation whose note says "existing customers only" / allocation counts as limited
    pres["unavailable_alloc"] = ((pres["unavailable"] == 1) | (pres["allocation_note"] == 1)).astype(int)
    pres["unavailable_note"] = ((pres["unavailable"] == 1) | (pres["constrained_note"] == 1)).astype(int)
    # outcome variant: an Available presentation whose note says "check wholesalers for inventory" counts as limited
    pres["wholesaler_note"] = pres["related_info"].fillna("").astype(str).str.contains("wholesal", case=False).astype(int)
    pres["unavailable_wholesaler"] = ((pres["unavailable"] == 1) | (pres["wholesaler_note"] == 1)).astype(int)
    # outcome variant: only "Unavailable" counts (Limited Availability treated as available)
    pres["unavailable_only"] = (pres["availability"].astype(str).str.strip().str.lower() == "unavailable").astype(int)
    # outcome variant: presentations whose reported reason is a discontinuation are treated as not listed (a permanent
    # market exit, not an availability state), so a cell is unavailable only through its other presentations
    pres["related_info"] = pres["related_info"].fillna("")
    pres["discontinued_reason"] = pres["shortage_reason"].str.contains("Discontinuation", case=False).astype(int)
    pres["discontinued_note"] = pres["related_info"].str.contains(DISCONTINUED_NOTE_PATTERN, case=False, regex=True).astype(int)
    pres["discontinued"] = ((pres["discontinued_reason"] == 1) | (pres["discontinued_note"] == 1)).astype(int)
    pres["unavailable_nodisc"] = ((pres["unavailable"] == 1) & (pres["discontinued"] == 0)).astype(int)
    # extended discontinuation (domain referee, paper-v5 round 3): also a presentation whose product NDC is no longer in the
    # NDC Directory (delisted), which many records describe only as "Estimated recovery: TBD"
    pres["discontinued_ext"] = ((pres["discontinued"] == 1) | (pres["in_ndc_directory"] == 0)).astype(int)
    pres["unavailable_nodisc_ext"] = ((pres["unavailable"] == 1) & (pres["discontinued_ext"] == 0)).astype(int)
    cells = (pres.groupby(["firm_unit", "drug"], as_index=False)
             .agg(firm=("firm", "first"), company_name=("company_name", "first"), company_group=("company_group", "first"),
                  n_presentations=("presentation", "size"), n_unavailable=("unavailable", "sum"),
                  unavailable=("unavailable", "max"), all_unavailable=("unavailable", "min"), unavailable_alloc=("unavailable_alloc", "max"), unavailable_note=("unavailable_note", "max"), unavailable_wholesaler=("unavailable_wholesaler", "max"), unavailable_only=("unavailable_only", "max"), unavailable_nodisc=("unavailable_nodisc", "max"), all_discontinued=("discontinued", "min"), unavailable_nodisc_ext=("unavailable_nodisc_ext", "max"), all_discontinued_ext=("discontinued_ext", "min"), left_censored=("recall_window_left_censored", "max"),
                  first_posting=("initial_posting_date", "min"),
                  injectable=("injectable", "max"), controlled=("controlled", "max"),
                  n_firms_in_market=("n_firms_in_market", "max"), n_apps_in_market=("n_apps_in_market", "max"), first_approval_year=("first_approval_year", "min"),
                  firm_n_markets=("firm_n_markets", "max"), firm_recall_events_5y=("firm_recall_events_5y", "max"),
                  firm_recall_events_pre_posting=("firm_recall_events_pre_posting", "max"),
                  in_ndc_directory=("in_ndc_directory", "max")))
    # extended recall measure (sensitivity; Section 2.7): Class I/II events in the five years before the snapshot linked to
    # any of the firm unit's labeler codes or to its normalised recalling-firm name, united across all of the unit's
    # presentations. The "other drugs" version drops the events whose product description names any active ingredient
    # of the cell (full ingredient names from the NDC Directory; for presentations without an NDC match, the FDA
    # generic name without its dosage-form words).
    def evset(x):
        return set(str(x).split(";")) - {"", "nan"}
    desc = pd.read_csv(os.path.join(PROC, "recall_events.csv"), dtype=str).fillna("")
    desc = dict(zip(desc["event_id"], desc["description"].str.lower()))
    form_words = r"\b(injection|tablet|tablets|capsule|capsules|solution|suspension|oral|extended|release|chewable|film|for|irrigant|sterile|usp|powder|cream|ointment|emulsion|kit|vial|concentrate)\b"
    def names(row):
        if isinstance(row["ingredients"], str) and row["ingredients"]:
            return {n.strip().lower() for n in row["ingredients"].split(";") if n.strip()}
        nm = re.sub(form_words, " ", row["drug"].lower()); nm = re.sub(r"[^a-z ]", " ", nm)
        return {" ".join(nm.split())} - {""}
    pres["_ext"] = pres["recall_events_ext"].map(evset); pres["_names"] = pres.apply(names, axis=1)
    firm_ext = pres.groupby("firm_unit")["_ext"].agg(lambda s: set().union(*s))
    cell_names = pres.groupby(["firm_unit", "drug"])["_names"].agg(lambda s: set().union(*s)).reset_index()
    cell_names["recall_ext_events"] = cell_names["firm_unit"].map(firm_ext).map(len)
    cell_names["recall_ext_other_events"] = [sum(1 for e in firm_ext[f] if not any(n and n in desc.get(e, "") for n in nm))
                                             for f, nm in zip(cell_names["firm_unit"], cell_names["_names"])]
    cells = cells.merge(cell_names[["firm_unit", "drug", "recall_ext_events", "recall_ext_other_events"]], on=["firm_unit", "drug"])
    # reason type of the cell: firm-type if any presentation reports a firm-type reason, market-type if any
    # reports a market-type reason (and none firm-type), else none
    def reason_type(s):
        s = set(x for x in s if x)
        if s & FIRM_REASONS: return "firm"
        if s & MARKET_REASONS: return "market"
        if "Other" in s: return "other"
        return "none"
    def reason_type_market_first(s):
        s = set(x for x in s if x)
        if s & MARKET_REASONS: return "market"
        if s & FIRM_REASONS: return "firm"
        if "Other" in s: return "other"
        return "none"
    rt = pres.groupby(["firm_unit", "drug"])["shortage_reason"].agg(reason_type).rename("reason_type").reset_index()
    rt2 = pres.groupby(["firm_unit", "drug"])["shortage_reason"].agg(reason_type_market_first).rename("reason_type_market_first").reset_index()
    cells = cells.merge(rt, on=["firm_unit", "drug"]).merge(rt2, on=["firm_unit", "drug"])
    # display name for figures: the most frequent original company name of the unit, lightly shortened
    disp = (pres.groupby("firm_unit")["company_name"].agg(lambda s: s.value_counts().index[0])
            .map(lambda n: re.sub(r",?\s*(Inc\.?|LLC|Ltd\.?|L\.P\.|Corporation|Corp\.?|Co\.?|USA|U\.S\.|a Pfizer Company|a Viatris Company)\b\.?", "", n).strip(" ,")).rename("display_name"))
    cells = cells.merge(disp, left_on="firm_unit", right_index=True, how="left")
    drugs = (cells.groupby("drug", as_index=False)
             .agg(n_firms_listed=("firm_unit", "nunique"), n_cells_unavailable=("unavailable", "sum"),
                  injectable=("injectable", "max"), controlled=("controlled", "max"),
                  n_firms_in_market=("n_firms_in_market", "max"), n_apps_in_market=("n_apps_in_market", "max"), first_approval_year=("first_approval_year", "min"),
                  first_posting=("first_posting", "min")))
    firms = (cells.groupby("firm_unit", as_index=False)
             .agg(n_drugs_listed=("drug", "nunique"), n_drugs_unavailable=("unavailable", "sum"),
                  firm_n_markets=("firm_n_markets", "max"), firm_recall_events_5y=("firm_recall_events_5y", "max")))
    return cells, pres, drugs, firms
