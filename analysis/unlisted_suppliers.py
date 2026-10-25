"""
unlisted_suppliers.py -- firm-level listing selection (Section 3.2; round-1 domain referee of paper-v4). The FDA list names
the presentations of the firms that report to it; approved manufacturers of a listed drug that are not on the list are not
observed. If some firms list only their unavailable products while others are absent altogether, firm clustering could be
inflated by who is listed. This script bounds that effect: for every listed drug it finds the approved applications in the
drug's NDC-Directory product market(s) that have no listed presentation, takes each application's holder from the Orange Book
(so repackagers and relabelers, which carry another firm's application, are not counted), keeps only applications whose
Orange Book products are not all marked discontinued, and adds one *available* cell per unlisted holder and drug, with the
firm defined as the marketer of record (normalised name; analysis/common.py). The model-free tests of Table 4 are rerun on
the augmented graph. Note (methods referee, v4 round 2): setting every added cell available creates all-available rows for
new firms and so inflates firm clustering; it is NOT a conservative bound. The paper reports the conservative versions in
analysis/robustness_v4.py (added cells only for firms already listed; outcomes imputed at the drug's listed rate).
Writes results/unlisted_suppliers.json.
"""
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, normalise_company, RESULTS, CORPORATE_GROUP  # noqa: E402
from analysis.outcome_variants import model_free_row, SEED  # noqa: E402
from pipeline.build_data import load_zip_json, form_family  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")


def market_key(r):
    """The product-market key of pipeline/build_data.py (active ingredient set, dosage-form family, route)."""
    ai = tuple(sorted({a["name"].upper().strip() for a in r.get("active_ingredients", [])}))
    return (ai, form_family(r.get("dosage_form")), tuple(sorted(r.get("route") or [])))


def orange_book_holders():
    """Application ('ANDA123456' style) -> (normalised applicant name, True if any product is not discontinued)."""
    z = zipfile.ZipFile(os.path.join(RAW, "EOBZIP_2026_09.zip"))
    lines = z.read("products.txt").decode("latin-1").splitlines()
    h = lines[0].split("~"); out = {}
    for ln in lines[1:]:
        r = dict(zip(h, ln.split("~")))
        app = ("ANDA" if r["Appl_Type"] == "A" else "NDA") + r["Appl_No"].lstrip("0").rjust(6, "0")
        active = r["Type"] != "DISCN"
        name = normalise_company(r["Applicant_Full_Name"] or r["Applicant"])
        prev = out.get(app)
        out[app] = (name, active or (prev[1] if prev else False))
    return out


def norm_app(a):
    a = str(a).upper().replace(" ", "")
    for p in ("NDAAUTHORIZEDGENERIC", "ANDA", "NDA", "BLA"):
        if a.startswith(p):
            num = a[len(p):].lstrip("0")
            return ("ANDA" if p == "ANDA" else "NDA") + num.rjust(6, "0") if num.isdigit() else None
    return None


def unlisted_cells(cells, pres, stats=None):
    """One available cell (firm_unit, drug) per Orange Book holder of an approved, not-discontinued application in a listed
    drug's NDC-Directory market with no listed presentation (each such application has a finished, listed NDC product by
    construction of the market universe), excluding holders that are a listed firm of the drug or of its corporate group.
    cells and pres are the marketer-unit tables of load_cells("marketer"). If stats is a dict, screening counts go into it."""
    ndc = load_zip_json("drug-ndc-0001-of-0001.json.zip")["results"]
    rx = [r for r in ndc if r.get("product_type") == "HUMAN PRESCRIPTION DRUG" and r.get("finished")
          and r.get("marketing_category") in ("ANDA", "NDA", "NDA AUTHORIZED GENERIC", "BLA")]
    market_apps = defaultdict(set)
    for r in rx:
        a = norm_app(r.get("application_number") or "")
        if a: market_apps[repr(market_key(r))].add(a)
    ob = orange_book_holders()
    added = []; n_apps_unlisted = n_no_ob = n_discn = n_same_firm = 0
    for drug, g in pres.groupby("drug"):
        keys = set(g["market_key"].dropna()) - {""}
        listed_apps = {norm_app(a) for a in g["application_number"].dropna()} - {None}
        listed_firms = set(cells.loc[cells["drug"] == drug, "firm_unit"])
        listed_groups = {CORPORATE_GROUP.get(f, f) for f in listed_firms}
        holders = set()
        for k in keys:
            for a in market_apps.get(k, ()):
                if a in listed_apps: continue
                n_apps_unlisted += 1
                if a not in ob: n_no_ob += 1; continue
                name, active = ob[a]
                if not active: n_discn += 1; continue
                if name in listed_firms or CORPORATE_GROUP.get(name, name) in listed_groups: n_same_firm += 1; continue
                holders.add(name)
        for h in sorted(holders):
            added.append(dict(firm_unit=h, drug=drug))
    if stats is not None:
        stats.update(n_unlisted_applications=n_apps_unlisted, n_not_in_orange_book=n_no_ob, n_discontinued=n_discn, n_held_by_listed_firm=n_same_firm)
    return pd.DataFrame(added)


def main():
    cells, pres, _, _ = load_cells("marketer")
    out = dict(n_drugs=int(cells["drug"].nunique()))
    # presentations whose marketer of record differs from the reporting company (domain referee, v4 round 1)
    moved = pres["marketer_norm"] != pres["company_norm"]
    out["n_pres_reassigned"] = int(moved.sum()); out["n_unavail_pres_reassigned"] = int((moved & (pres["unavailable"] == 1)).sum())
    cc, _, _, _ = load_cells("company")
    out["n_cells_company"] = int(len(cc)); out["n_unavail_cells_company"] = int(cc["unavailable"].sum())
    out["n_cells_marketer"] = int(len(cells)); out["n_unavail_cells_marketer"] = int(cells["unavailable"].sum())
    st = {}; add = unlisted_cells(cells, pres, st); out.update(st)
    out.update(n_added_cells=int(len(add)), n_added_firms=int(add["firm_unit"].nunique()),
               n_added_firms_new=int(len(set(add["firm_unit"]) - set(cells["firm_unit"]))),
               n_drugs_with_added=int(add["drug"].nunique()),
               top_added_firms=add["firm_unit"].value_counts().head(8).to_dict())
    aug = pd.concat([cells[["firm_unit", "drug", "n_presentations", "unavailable", "unavailable_only"]],
                     add.assign(n_presentations=1, unavailable=0, unavailable_only=0)], ignore_index=True)
    rng = np.random.default_rng(SEED + 77)
    for name, col in (("main", "unavailable"), ("unavailable_only", "unavailable_only")):
        out[name] = model_free_row(aug, aug[col].values, rng); out[name]["n_firms"] = int(aug["firm_unit"].nunique())
        print(name, {k: round(v, 3) for k, v in out[name].items() if isinstance(v, (int, float))})
    print({k: v for k, v in out.items() if k not in ("main", "unavailable_only")})
    with open(os.path.join(RESULTS, "unlisted_suppliers.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
