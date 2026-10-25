"""
fda_csv.py -- read the archived copies of the FDA Drug Shortage Database CSV download (download_archive_csv.py;
data/raw/archive_csv/) into presentation and cell tables with the definitions of the main analysis.

The CSV is the agency's own "download current drug shortages" file, one row per presentation (Generic Name, Company
Name, Presentation with its NDC, Availability Information, Related Information, Reason for Shortage, Status, Initial
Posting Date, ...). The capture of 16 October 2026 has the same 1,141 current presentations and 69 drugs as the openFDA
export of 13 October 2026 used in the main analysis; analysis/long_panel.py checks the agreement at all three openFDA dates.

Availability. Since an FDA system change the field holds one of "Available", "Unavailable", "Limited Availability"; in
older captures it is free text written by the firm ("Next Delivery: October 2019; Estimated Recovery: ...", "Inventory is
currently available.", "Backordered. Next release ..."). The main outcome needs only whether a presentation is
available without constraint, so free text is coded by two published keyword lists (CONSTRAINT, AVAILABLE): any
constraint word (backorder, unavailable, next delivery, estimated recovery, allocation, limited, discontinued, TBD, ...)
makes the presentation unavailable or limited; otherwise an availability word (available, in stock, shipping, inventory)
makes it available; text matching neither is dropped and counted. The coding of every distinct string is written to
results/fda_csv_availability_coding.csv so that it can be audited. The Unavailable-only outcome is defined only for the
categorical values.
"""
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import normalise_company, DISCONTINUED_NOTE_PATTERN  # noqa: E402

CSV_DIR = os.path.join(ROOT, "data", "raw", "archive_csv")

CONSTRAINT = re.compile(
    r"unavail|unvail|not (?:currently |yet )?available|no (?:stock|product|current supply|supply|inventory)|back[- ]?order|\bbo\b|"
    r"out of stock|stock(?:ed)? ?out|limited|allocat|intermittent|constrain|shortage|delay|discontinu|no longer|not (?:currently )?marketed|"
    r"next (?:delivery|release|shipment|lot)|recover|re-?suppl|\btbd\b|to be determined|not sooner|expected availability|"
    r"estimated availability|availability is unknown|temporar|contracted customers|existing customers|current customers|"
    r"primary (?:customers|accounts)|can only|cannot|on hold|interruption|unknown|pending|release date|expected (?:to|in)|anticipated|"
    r"deplet|low inventory|inventory is low|low stock|tentative|sold out|estimated delivery|short[- ]dat|restricted distribution|"
    r"no disruption to existing patients")
AVAILABLE = re.compile(r"\bavailable\b|in stock|shipping|inventory|\bavail\b|\bavailble\b")
# text that opens by saying the product is available and names no restriction (only a future release or resupply) is
# available: "Stock available. Additional resupply June 2020", "Available; additional units expected in Nov 2021"
AVAILABLE_LEAD = re.compile(r"^(?:this presentation is |stock |product |inventory |supply )?(?:is )?(?:currently )?(?:available|in stock)\b")
STRONG = re.compile(r"unavail|not (?:currently )?available|no (?:stock|product|supply)|back[- ]?order|\bbo\b|out of stock|stock(?:ed)? ?out|"
                    r"limited|allocat|intermittent|constrain|shortage|discontinu|no longer|not (?:currently )?marketed|cannot|can only|"
                    r"contracted customers|existing customers|current customers|primary (?:customers|accounts)|on hold|short[- ]dat|deplet|low inventory|"
                    r"inventory is low|\btbd\b|sold out")
# available only at a future date ("Available in February 2024", "available by end of October 2023"): not available now
FUTURE = re.compile(r"^(?:back in stock|(?:product |stock |supply |inventory )?(?:will be |to be |is expected to be )?(?:available|availability)) (?:in|by|from|starting|beginning|week|wk|"
                    r"early|mid|late|end|after|on|around|approximately)\b(?=.*(?:january|february|march|april|may|june|july|august|september|"
                    r"october|november|december|\bq[1-4]\b|quarter|20\d\d|\d{1,2}/\d{1,2}))")
# not (yet, or no longer) on the market: launches, first releases and products the firm does not distribute or market
NOT_MARKETED = re.compile(r"not (?:currently )?marketing|estimated release|^release:|resumes")
# entries for presentations that were never part of supply (not yet launched, not distributed, not commercialised): the
# presentation is treated as not listed and counted
PRELAUNCH = re.compile(r"not (?:currently )?distribut|do not distribute|not commerciali[sz]|not currently offered|\blaunch|not yet launched|"
                       r"targeted for|planned for|initial availability")
# entries that say nothing about availability
UNINFORMATIVE = re.compile(r"^(?:pending information|information pending|tbd|undetermined|to be determined|n/?a|unknown|pending)$")
# constraints that concern only how supply is distributed (allocation, limited quantities, customer restrictions); firms
# mapped many of these to "Available" at the 2024 change to categorical values, so a sensitivity codes them as available
ALLOCATION_ONLY = re.compile(r"allocat|limited|intermittent|contracted customers|existing customers|current customers|primary (?:customers|accounts)|"
                             r"can only|constrain|low inventory")
HARD = re.compile(r"unavail|unvail|not (?:currently |yet )?available|no (?:stock|product|current supply|supply|inventory)|back[- ]?order|\bbo\b|"
                  r"out of stock|stock(?:ed)? ?out|discontinu|no longer|not (?:currently )?market|deplet|next (?:delivery|release|shipment|lot)|"
                  r"recover|re-?suppl|\btbd\b|to be determined|temporar|on hold|interruption|release date")
CATEGORICAL = {"available": 0, "unavailable": 1, "limited availability": 1}
# reporting companies renamed within the archived period (same company, new name), mapped to the latest name; the
# variants were found by listing every normalised name with its first and last capture
RENAMES = {"ICU": "OTSUKA ICU", "PAR": "PAR HEALTH", "AUROMEDICS": "EUGIA", "EUGIA SPECIALTIES": "EUGIA", "EUGIA MFD BY EUGIA SPECIALTIES": "EUGIA",
           "AMERICAN REGENT LUITPOLD": "AMERICAN REGENT", "HERITAGE": "AVET", "HERITAGE D B A AVET": "AVET", "FRESENIUS KASBI": "FRESENIUS KABI",
           "VIATRIS": "MYLAN"}
MONTHS = {m: i + 1 for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                                          "november", "december"))}


def code_availability(text, harmonise=False):
    """(main outcome 0/1 or None if uninformative, Unavailable-only outcome 0/1 or None, categorical flag). harmonise=True:
    free text that signals only allocation or limited quantities (no backorder, outage, recovery date or discontinuation)
    counts as available."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".").strip()
    if t in CATEGORICAL:
        return CATEGORICAL[t], int(t == "unavailable"), True
    if not t or UNINFORMATIVE.match(t) or PRELAUNCH.search(t):
        return None, None, False
    if harmonise and ALLOCATION_ONLY.search(t) and not HARD.search(t):
        return 0, None, False
    if FUTURE.search(t):
        return 1, None, False
    if AVAILABLE_LEAD.search(t) and not STRONG.search(t):
        return 0, None, False
    if CONSTRAINT.search(t) or NOT_MARKETED.search(t):
        return 1, None, False
    if AVAILABLE.search(t):
        return 0, None, False
    return None, None, False


def stale(text, capture_date, days=31):
    """True if a free-text entry gives delivery or recovery dates ("October 2022") and every one of them is more than
    `days` before the capture date: the entry was not revised after the date it announced."""
    t = (text or "").lower()
    found = [(int(y), MONTHS[m]) for m, y in re.findall(r"\b(" + "|".join(MONTHS) + r")\.?,? (?:\d{1,2},? )?(20\d\d)\b", t)]
    if not found:
        return False
    last = max(pd.Timestamp(year=y, month=m, day=28) for y, m in found)
    return bool(last < capture_date - pd.Timedelta(days=days))


def read_capture(path, date=None):
    """Current presentations of one CSV capture (date: the capture date, for the stale-entry flag)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    head = [h.strip().lstrip("﻿") for h in rows[0]]
    recs = []
    ip = head.index("Presentation")
    for r in rows[1:]:
        if len(r) > len(head):                       # an unquoted comma split the presentation field
            k = len(r) - len(head); r = r[:ip] + [",".join(r[ip:ip + k + 1])] + r[ip + k + 1:]
        d = dict(zip(head, r))
        if (d.get("Status") or "").strip().lower() != "current":
            continue
        y, yu, cat = code_availability(d.get("Availability Information"))
        yh = code_availability(d.get("Availability Information"), harmonise=True)[0]
        st = int((not cat) and y == 1 and date is not None and stale(d.get("Availability Information"), date))
        pres = d.get("Presentation") or ""
        m = re.search(r"NDC\s*:?\s*(\d{4,5})-(\d{3,4})-(\d{1,2})", pres)
        firm = normalise_company((d.get("Company Name") or "").strip()); firm = RENAMES.get(firm, firm)
        note = d.get("Related Information") or ""; reason = d.get("Reason for Shortage") or ""
        avail_txt = (d.get("Availability Information") or "").lower()
        disc = int("discontinu" in avail_txt or "discontinu" in reason.lower() or "no longer distribut" in avail_txt
                   or bool(re.search(r"not (?:currently )?distribut|do not distribute|not marketing|not (?:currently )?marketed", avail_txt))
                   or bool(re.search(DISCONTINUED_NOTE_PATTERN, note + " " + avail_txt, flags=re.I)))
        recs.append(dict(firm_unit=firm, drug=(d.get("Generic Name") or "").strip(), unavailable=y, unavailable_only=yu,
                         unavailable_harmonised=yh, unavailable_nostale=(None if st else y), stale=st,
                         categorical=int(cat), discontinued=disc, labeler=(m.group(1) if m else None),
                         package_ndc=("-".join(m.groups()) if m else None), availability_text=(d.get("Availability Information") or "").strip(),
                         initial_posting=pd.to_datetime(d.get("Initial Posting Date"), format="%m/%d/%Y", errors="coerce"),
                         update_date=pd.to_datetime(d.get("Date of Update"), format="%m/%d/%Y", errors="coerce")))
    return pd.DataFrame(recs)


def captures():
    """(timestamp, date, path) of every capture on disk, oldest first."""
    out = []
    for p in sorted(glob.glob(os.path.join(CSV_DIR, "Drugshortages-wayback-*.csv"))):
        ts = re.search(r"(\d{14})", p).group(1)
        out.append((ts, pd.Timestamp(ts[:8]), p))
    return out


def cells_of(p, outcome="unavailable", unit="company", nodisc=False):
    """Firm-drug cells of one capture's presentations (presentations with an uninformative outcome dropped)."""
    q = p[p[outcome].notna()]
    if nodisc:
        q = q[q["discontinued"] == 0]
    if unit == "labeler":
        q = q[q["labeler"].notna()].assign(firm_unit=lambda x: x["labeler"])
    elif unit == "marketer":
        q = q[q["marketer"].notna()].assign(firm_unit=lambda x: x["marketer"])
    c = (q.groupby(["firm_unit", "drug"], as_index=False)
         .agg(n_presentations=(outcome, "size"), unavailable=(outcome, "max"), discontinued_all=("discontinued", "min")))
    c["unavailable"] = c["unavailable"].astype(int)
    return c


_NOT_INGREDIENT = {"injection", "injectable", "hydrochloride", "sodium", "tablet", "tablets", "capsule", "capsules", "solution",
                   "suspension", "powder", "oral", "ophthalmic", "nasal", "spray", "acetate", "sulfate", "phosphate", "monohydrate",
                   "bitartrate", "citrate", "potassium", "calcium", "with", "and", "for", "plastic", "container", "lyophilized",
                   "sterile", "irrigation", "irrigant", "inhalation", "inhalational", "extended", "release", "delayed", "grams",
                   "vials", "syringes", "concentrate", "emulsion", "premix", "premixed"}


def _ingredient_words(names):
    return {w for n in names for w in re.findall(r"[a-z]{4,}", n.lower()) if w not in _NOT_INGREDIENT}


def _name_key(n):
    k = re.sub(r"\(.*?\)", " ", n.lower())
    k = re.sub(r"\busp\b|\bmonohydrate\b|\banhydrous\b|[^a-z0-9%. ]", " ", k)
    k = re.sub(r"\b(tablet|capsule|injection|solution|suspension)s\b", r"\1", k)
    return " ".join(k.split())


def canonical_drugs(P):
    """Map every generic name in the captures {timestamp: presentations} to one name per drug across FDA renamings (", USP"
    dropped in 2021, "Dextrose 50% Injection" -> "Dextrose Monohydrate 50% Injection", ...). Two names are the same drug if
    they differ only in style (", USP", a bracketed brand name, "Monohydrate", a plural), or if they share a package NDC in
    different captures, have the same ingredient words and are never listed as two drugs in one capture; the canonical name is the
    latest one, so the most recent capture keeps the names of the main analysis. Returns (mapping, number of merges)."""
    ts = sorted(P)
    names = sorted({n for t in ts for n in P[t]["drug"].unique()})
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    together = set()
    for t in ts:
        u = sorted(P[t]["drug"].unique())
        together.update((a, b) for i, a in enumerate(u) for b in u[i + 1:])
    by_ndc = {}
    for t in ts:
        for n, k in zip(P[t]["drug"], P[t]["package_ndc"]):
            if isinstance(k, str):
                by_ndc.setdefault(k, set()).add(n)
    last_seen = {n: t for t in ts for n in P[t]["drug"].unique()}
    members = {n: {n} for n in names}
    merges = 0
    # names that differ only in style (", USP", a bracketed brand, "Monohydrate", plural forms) are one drug even when a
    # renaming was rolled out over several days and both names appear in one capture
    by_key = {}
    for n in names:
        by_key.setdefault(_name_key(n), []).append(n)
    for grp in by_key.values():
        for a in grp[1:]:
            ra, rb = find(grp[0]), find(a)
            if ra != rb:
                parent[rb] = ra; members[ra] |= members.pop(rb); merges += 1
    for k, ns in sorted(by_ndc.items()):
        ns = sorted(ns)
        for a in ns[1:]:
            ra, rb = find(ns[0]), find(a)
            if ra == rb:
                continue
            # never merge two groups that contain names listed together in one capture, or names without a common
            # ingredient word (a shared NDC between, say, saline and sterile-water irrigation is a data error)
            if any((x, y) in together or (y, x) in together for x in members[ra] for y in members[rb]):
                continue
            if _ingredient_words(members[ra]) != _ingredient_words(members[rb]):
                continue                                 # e.g. cyclopentolate vs cyclopentolate with phenylephrine
            parent[rb] = ra; members[ra] |= members.pop(rb); merges += 1
    mapping = {}
    for n in names:
        grp = members[find(n)]
        mapping[n] = max(grp, key=lambda x: (last_seen[x], x))
    return mapping, merges


def marketer_names():
    """NDC labeler code -> normalised marketer name: the labeler name in the NDC Directory export (as for the marketer of
    record in the main analysis), else the openFDA manufacturer name recorded for that code in the three openFDA shortage
    exports; codes found in neither keep the reporting company (marketer = None here)."""
    from pipeline.build_data import load_zip_json
    from analysis import panel
    import json as _json, zipfile as _zip
    names = {}
    for path in [p for _, p in panel.WAVES]:
        z = _zip.ZipFile(path); d = _json.loads(z.read(z.namelist()[0]))
        for r in d["results"]:
            mn = (r.get("openfda") or {}).get("manufacturer_name") or []
            code = (r.get("package_ndc") or "").split("-")[0]
            if mn and code:
                names.setdefault(code, normalise_company(mn[0]))
    for r in load_zip_json("drug-ndc-0001-of-0001.json.zip")["results"]:
        code = (r.get("product_ndc") or "").split("-")[0]
        if code and r.get("labeler_name"):
            names[code] = normalise_company(r["labeler_name"])
    return names


def add_marketer(p, names):
    """Marketer column: the normalised labeler name of the presentation's NDC labeler code, else the reporting company."""
    return p.assign(marketer=[names.get(c, f) if isinstance(c, str) else f for c, f in zip(p["labeler"], p["firm_unit"])])
