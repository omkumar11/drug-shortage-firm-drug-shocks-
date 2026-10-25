"""
build_data.py -- turn the raw public FDA files in data/raw/ into the analysis tables in data/processed/.

Inputs (all downloaded by download_data.py, URLs and dates in data/raw/download_log.txt):
  drug-shortages-0001-of-0001.json.zip   openFDA drug shortages (FDA Drug Shortage Database, daily export)
  drug-ndc-0001-of-0001.json.zip         openFDA NDC Directory
  drug-enforcement-0001-of-0001.json.zip openFDA drug recall (enforcement) reports
  EOBZIP_2026_09.zip                     FDA Orange Book data files (products.txt)

Outputs:
  data/processed/presentations.csv   one row per presentation on the Current shortage list
  data/processed/cells.csv           one row per (labeler code, drug) cell on the Current shortage list; the manuscript's
                                     primary firm unit (the normalised reporting company) is built from presentations.csv
                                     by analysis/common.py, which re-aggregates the cells
  data/processed/drugs.csv           one row per drug (FDA generic_name) on the Current shortage list
  data/processed/firms.csv           one row per firm (NDC labeler code) that appears on the list
  data/processed/build_summary.json  counts used in the manuscript

Definitions (these are the definitions stated in the manuscript, Section 2):
  firm   = NDC labeler code (first segment of the NDC) of the presentation.
  drug   = the FDA shortage database's generic_name (active ingredient(s) + dosage form), the unit at which FDA
           declares a shortage and under which it lists every reporting manufacturer's presentations.
  unavailable presentation = availability field is "Unavailable" or "Limited Availability" (case/typo tolerant).
  cell (firm, drug) unavailable = at least one of the firm's presentations of the drug is unavailable.
"""
import collections
import csv
import datetime as dt
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

FORM_FAMILY = {
    # NDC dosage_form first token -> coarse family used to count competitors in a drug market
    "INJECTION": "INJECTABLE", "INJECTABLE": "INJECTABLE",
    "TABLET": "TABLET", "CAPSULE": "CAPSULE", "SOLUTION": "SOLUTION", "SUSPENSION": "SUSPENSION",
    "POWDER": "POWDER", "KIT": "KIT", "SPRAY": "SPRAY", "OINTMENT": "OINTMENT", "CREAM": "CREAM",
    "FILM": "FILM", "IRRIGANT": "IRRIGANT", "CONCENTRATE": "CONCENTRATE", "SUPPOSITORY": "SUPPOSITORY",
    "EMULSION": "EMULSION", "GEL": "GEL", "LIQUID": "LIQUID", "AEROSOL": "AEROSOL", "PATCH": "PATCH",
    "GRANULE": "GRANULE", "SYRUP": "SYRUP", "LOTION": "LOTION", "IMPLANT": "IMPLANT", "INSERT": "INSERT",
}


def load_zip_json(name):
    z = zipfile.ZipFile(os.path.join(RAW, name))
    return json.loads(z.read(z.namelist()[0]))


def parse_date(s, fmt):
    try:
        return dt.datetime.strptime(s, fmt).date()
    except Exception:
        return None


def form_family(dosage_form):
    if not dosage_form:
        return "OTHER"
    tok = dosage_form.split(",")[0].strip().upper()
    return FORM_FAMILY.get(tok, "OTHER")


def norm_avail(a):
    a = (a or "").strip().lower()
    if a.startswith("unavail") or a.startswith("unvail"):
        return "Unavailable"
    if a.startswith("limited"):
        return "Limited Availability"
    if a.startswith("avail"):
        return "Available"
    return "Unknown"


def main():
    log = {}
    # ---------------------------------------------------------------- NDC directory
    ndc_meta = load_zip_json("drug-ndc-0001-of-0001.json.zip")
    ndc = ndc_meta["results"]
    log["ndc_export_date"] = ndc_meta["meta"].get("last_updated")
    ndc_by_product = {r["product_ndc"]: r for r in ndc}
    labeler_names = collections.defaultdict(collections.Counter)
    for r in ndc:
        labeler_names[r["product_ndc"].split("-")[0]][r["labeler_name"]] += 1
    labeler_name = {c: n.most_common(1)[0][0] for c, n in labeler_names.items()}

    # approved, finished, human prescription products define the market universe
    rx = [r for r in ndc if r.get("product_type") == "HUMAN PRESCRIPTION DRUG" and r.get("finished")
          and r.get("marketing_category") in ("ANDA", "NDA", "NDA AUTHORIZED GENERIC", "BLA")]

    def market_key(r):
        ai = tuple(sorted({a["name"].upper().strip() for a in r.get("active_ingredients", [])}))
        return (ai, form_family(r.get("dosage_form")), tuple(sorted(r.get("route") or [])))

    market_firms = collections.defaultdict(set)      # market key -> labeler codes (includes repackagers/relabelers)
    market_apps = collections.defaultdict(set)       # market key -> distinct approved application numbers
    market_products = collections.Counter()
    firm_markets = collections.defaultdict(set)      # labeler code -> market keys
    for r in rx:
        k = market_key(r)
        code = r["product_ndc"].split("-")[0]
        market_firms[k].add(code)
        if r.get("application_number"):
            market_apps[k].add(r["application_number"].upper().replace(" ", ""))
        market_products[k] += 1
        firm_markets[code].add(k)
    log["ndc_products_total"] = len(ndc)
    log["ndc_rx_approved_finished"] = len(rx)
    log["ndc_markets"] = len(market_firms)
    log["ndc_firm_market_cells"] = sum(len(v) for v in market_firms.values())

    # ---------------------------------------------------------------- Orange Book (approval dates)
    z = zipfile.ZipFile(os.path.join(RAW, "EOBZIP_2026_09.zip"))
    ob_lines = z.read("products.txt").decode("latin-1").splitlines()
    ob_header = ob_lines[0].split("~")
    ob = [dict(zip(ob_header, l.split("~"))) for l in ob_lines[1:] if l.strip()]
    ob_first_approval = {}   # (ingredient string upper, OB form family) -> earliest approval date
    ob_anda_holders = collections.defaultdict(set)
    for r in ob:
        ing = r["Ingredient"].upper().strip()
        fam = r["DF;Route"].split(";")[0].split(",")[0].strip().upper()
        fam = "INJECTABLE" if fam.startswith("INJECT") else fam
        d = parse_date(r["Approval_Date"], "%b %d, %Y")
        key = (ing, fam)
        if d is not None and (key not in ob_first_approval or d < ob_first_approval[key]):
            ob_first_approval[key] = d
        if r["Appl_Type"] == "A" and r["Type"] == "RX":
            ob_anda_holders[key].add(r["Applicant"])
    log["orange_book_rows"] = len(ob)

    # ---------------------------------------------------------------- recalls -> labeler codes
    enf_meta = load_zip_json("drug-enforcement-0001-of-0001.json.zip"); enf = enf_meta["results"]
    log["enforcement_export_date"] = enf_meta["meta"].get("last_updated")
    ndc_re = re.compile(r"\b(\d{4,5})-\d{3,4}-\d{1,2}\b|\bNDC[:\s#]*(\d{4,5})-\d{3,4}\b", re.I)
    recall_by_code = collections.defaultdict(list)   # labeler code -> list of (initiation date, class, event_id)
    # extended linkage (sensitivity): also match the normalised recalling-firm name to the normalised reporting company,
    # and keep the product description so that recalls of the same drug can be excluded
    sys.path.insert(0, os.path.dirname(HERE)); from analysis.common import normalise_company  # noqa: E402
    recall_by_name = collections.defaultdict(list)    # normalised firm name -> list of (date, class, event_id, description)
    recall_desc = {}                                   # event_id -> concatenated lower-case product descriptions
    for r in enf:
        codes = set()
        for m in ndc_re.finditer(r.get("product_description", "")):
            codes.add(m.group(1) or m.group(2))
        for p in (r.get("openfda") or {}).get("product_ndc", []) or []:
            codes.add(p.split("-")[0])
        d = parse_date(r.get("recall_initiation_date", ""), "%Y%m%d")
        if d is None:
            continue
        for c in codes:
            recall_by_code[c].append((d, r.get("classification"), r.get("event_id")))
        nm = normalise_company(r.get("recalling_firm", "") or "")
        if nm:
            recall_by_name[nm].append((d, r.get("classification"), r.get("event_id")))
        recall_desc[r.get("event_id")] = recall_desc.get(r.get("event_id"), "") + " " + (r.get("product_description", "") or "").lower()
    log["recall_reports"] = len(enf)

    # ---------------------------------------------------------------- shortage list
    sh_meta = load_zip_json("drug-shortages-0001-of-0001.json.zip")
    snapshot = sh_meta["meta"].get("last_updated")
    log["shortage_snapshot_date"] = snapshot
    snapshot_date = dt.date.fromisoformat(snapshot)
    sh = sh_meta["results"]
    log["shortage_records_all"] = len(sh)
    log["shortage_status_counts"] = dict(collections.Counter(r["status"] for r in sh))
    cur = [r for r in sh if r["status"] == "Current"]
    cur_drugs = {r["generic_name"].strip() for r in cur}
    tbd = [r for r in sh if r["status"] == "To Be Discontinued"]
    log["tbd_records_in_listed_drugs"] = sum(1 for r in tbd if (r.get("generic_name") or "").strip() in cur_drugs)
    log["tbd_records_total"] = len(tbd)

    rows = []
    for r in cur:
        pkg = r.get("package_ndc", "")
        parts = pkg.split("-")
        code = parts[0] if parts else ""
        product_ndc = "-".join(parts[:2]) if len(parts) >= 2 else ""
        nd = ndc_by_product.get(product_ndc)
        mk = market_key(nd) if nd else None
        ing = ";".join(mk[0]) if mk else ""
        fam = mk[1] if mk else form_family(r.get("dosage_form"))
        ob_key = (mk[0][0], fam) if (mk and len(mk[0]) == 1) else None
        first_appr = ob_first_approval.get(ob_key) if ob_key else None
        ipd = parse_date(r.get("initial_posting_date", ""), "%m/%d/%Y")
        upd = parse_date(r.get("update_date", ""), "%m/%d/%Y")
        recalls = recall_by_code.get(code, [])
        rel = (r.get("related_info") or "").lower()
        pre_events = {e for d, c, e in recalls if ipd and c in ("Class I", "Class II") and 0 <= (ipd - d).days <= 5 * 365}
        # extended: union of code-linked and name-linked Class I/II events in the five years before the snapshot
        ext = {e for d, c, e in recalls + recall_by_name.get(normalise_company(r.get("company_name", "")), [])
               if (snapshot_date - d).days <= 5 * 365 and c in ("Class I", "Class II")}

        rows.append(dict(
            drug=r["generic_name"].strip(),
            company_name=r.get("company_name", "").strip(),
            firm=code,
            firm_name=labeler_name.get(code, r.get("company_name", "")),
            package_ndc=pkg,
            product_ndc=product_ndc,
            in_ndc_directory=int(nd is not None),
            presentation=r.get("presentation", ""),
            availability=norm_avail(r.get("availability")),
            unavailable=int(norm_avail(r.get("availability")) in ("Unavailable", "Limited Availability")),
            shortage_reason=r.get("shortage_reason") or "",
            initial_posting_date=ipd.isoformat() if ipd else "",
            update_date=upd.isoformat() if upd else "",
            update_type=r.get("update_type", ""),
            dosage_form=r.get("dosage_form", ""),
            form_family=fam,
            injectable=int(fam == "INJECTABLE"),
            route=";".join(nd.get("route") or []) if nd else "",
            dea_schedule=(nd.get("dea_schedule") or "") if nd else "",
            controlled=int(bool(nd and nd.get("dea_schedule"))),
            marketing_category=(nd.get("marketing_category") or "") if nd else "",
            application_number=(nd.get("application_number") or "") if nd else "",
            therapeutic_category=";".join(r.get("therapeutic_category") or []),
            ingredients=ing,
            market_key=repr(mk) if mk else "",
            n_firms_in_market=len(market_firms[mk]) if mk else np.nan,
            n_apps_in_market=len(market_apps[mk]) if mk else np.nan,
            allocation_note=int("existing customer" in rel or "allocat" in rel),
            constrained_note=int("existing customer" in rel or "allocat" in rel or "additional lots" in rel or "limited" in rel or "as released" in rel),
            recall_window_left_censored=int(bool(ipd) and ipd.year - 5 < 2012),
            related_info=r.get("related_info") or "",
            n_products_in_market=market_products[mk] if mk else np.nan,
            first_approval_year=first_appr.year if first_appr else np.nan,
            n_anda_holders=len(ob_anda_holders.get(ob_key, ())) if ob_key else np.nan,
            firm_n_markets=len(firm_markets.get(code, ())),
            firm_recalls_5y=sum(1 for d, c, e in recalls if (snapshot_date - d).days <= 5 * 365 and c in ("Class I", "Class II")),
            firm_recall_events_5y=len({e for d, c, e in recalls if (snapshot_date - d).days <= 5 * 365 and c in ("Class I", "Class II")}),
            firm_recall_events_pre_posting=len(pre_events),
            recall_events_ext=";".join(sorted(str(e) for e in ext)),
        ))
    pres = pd.DataFrame(rows)
    # the "To Be Discontinued" records of listed drugs (not shortage entries; used only by the exit sensitivity of Table 4)
    tbd_rows = []
    for r in tbd:
        if (r.get("generic_name") or "").strip() not in cur_drugs: continue
        pkg = r.get("package_ndc", "") or ""; parts = pkg.split("-"); code = parts[0] if parts else ""
        product_ndc = "-".join(parts[:2]) if len(parts) >= 2 else ""; nd = ndc_by_product.get(product_ndc); mk = market_key(nd) if nd else None
        fam = mk[1] if mk else form_family(r.get("dosage_form"))
        tbd_rows.append(dict(drug=r["generic_name"].strip(), company_name=(r.get("company_name") or "").strip(), firm=code, package_ndc=pkg,
                             presentation=r.get("presentation", ""), status=r.get("status", ""), related_info=r.get("related_info") or "",
                             initial_posting_date=(parse_date(r.get("initial_posting_date", ""), "%m/%d/%Y") or "").isoformat() if parse_date(r.get("initial_posting_date", ""), "%m/%d/%Y") else "",
                             injectable=int(fam == "INJECTABLE")))
    pd.DataFrame(tbd_rows).to_csv(os.path.join(OUT, "presentations_tbd.csv"), index=False)
    log["tbd_rows_written"] = len(tbd_rows)
    pres.to_csv(os.path.join(OUT, "presentations.csv"), index=False)
    # product descriptions of the recall events (used to exclude recalls of a cell's own drug; see analysis/common.py)
    pd.DataFrame(sorted(recall_desc.items()), columns=["event_id", "description"]).to_csv(os.path.join(OUT, "recall_events.csv"), index=False)

    # ---------------------------------------------------------------- cells
    def agg_reason(s):
        s = [x for x in s if x]
        return collections.Counter(s).most_common(1)[0][0] if s else ""

    cells = (pres.groupby(["firm", "drug"], as_index=False)
             .agg(firm_name=("firm_name", "first"), company_name=("company_name", "first"),
                  n_presentations=("presentation", "size"), n_unavailable=("unavailable", "sum"),
                  unavailable=("unavailable", "max"),
                  first_posting=("initial_posting_date", "min"),
                  reason=("shortage_reason", agg_reason),
                  injectable=("injectable", "max"), controlled=("controlled", "max"),
                  n_firms_in_market=("n_firms_in_market", "max"), n_apps_in_market=("n_apps_in_market", "max"), first_approval_year=("first_approval_year", "min"),
                  n_anda_holders=("n_anda_holders", "max"),
                  firm_n_markets=("firm_n_markets", "max"), firm_recalls_5y=("firm_recalls_5y", "max"),
                  firm_recall_events_5y=("firm_recall_events_5y", "max"), firm_recall_events_pre_posting=("firm_recall_events_pre_posting", "max"),
                  in_ndc_directory=("in_ndc_directory", "max")))
    cells["share_unavailable"] = cells["n_unavailable"] / cells["n_presentations"]
    cells.to_csv(os.path.join(OUT, "cells.csv"), index=False)

    drugs = (cells.groupby("drug", as_index=False)
             .agg(n_firms_listed=("firm", "nunique"), n_cells_unavailable=("unavailable", "sum"),
                  n_presentations=("n_presentations", "sum"), injectable=("injectable", "max"),
                  controlled=("controlled", "max"), n_firms_in_market=("n_firms_in_market", "max"),
                  first_approval_year=("first_approval_year", "min"), first_posting=("first_posting", "min")))
    drugs["share_firms_unavailable"] = drugs["n_cells_unavailable"] / drugs["n_firms_listed"]
    drugs.to_csv(os.path.join(OUT, "drugs.csv"), index=False)

    firms = (cells.groupby("firm", as_index=False)
             .agg(firm_name=("firm_name", "first"), n_drugs_listed=("drug", "nunique"),
                  n_drugs_unavailable=("unavailable", "sum"), firm_n_markets=("firm_n_markets", "max"),
                  firm_recalls_5y=("firm_recalls_5y", "max"), firm_recall_events_5y=("firm_recall_events_5y", "max")))
    firms["share_drugs_unavailable"] = firms["n_drugs_unavailable"] / firms["n_drugs_listed"]
    firms.to_csv(os.path.join(OUT, "firms.csv"), index=False)

    log.update(dict(
        current_presentations=len(pres),
        current_presentations_in_ndc=int(pres["in_ndc_directory"].sum()),
        current_unavailable_presentations=int(pres["unavailable"].sum()),
        availability_counts=dict(pres["availability"].value_counts()),
        cells=len(cells), cells_unavailable=int(cells["unavailable"].sum()),
        drugs=len(drugs), firms=len(firms),
        drugs_injectable=int(drugs["injectable"].sum()),
        available_with_reason=int(((pres["availability"] == "Available") & (pres["shortage_reason"] != "")).sum()),
        available_related_info=dict(blank=int(((pres["availability"] == "Available") & (pres["related_info"] == "")).sum()),
                                    wholesalers=int(((pres["availability"] == "Available") & pres["related_info"].str.lower().str.contains("wholesaler")).sum()),
                                    allocation=int(((pres["availability"] == "Available") & (pres["allocation_note"] == 1)).sum())),
        labeler_codes_with_multiple_companies=int((pres.groupby("firm")["company_name"].nunique() > 1).sum()),
        companies_with_multiple_labelers=int((pres.groupby("company_name")["firm"].nunique() > 1).sum()),
        reason_counts=dict(pres["shortage_reason"].replace("", "(none)").value_counts()),
    ))
    with open(os.path.join(OUT, "build_summary.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
