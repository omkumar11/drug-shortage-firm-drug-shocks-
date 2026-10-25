"""
download_archive_csv.py -- fetch every Internet Archive capture of the FDA Drug Shortage Database CSV download
(https://www.accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm) into data/raw/archive_csv/.

The capture list is pinned in data/raw/archive_csv/manifest.csv (Wayback timestamp, Wayback digest, SHA-256 of the bytes
received). If the manifest is missing it is built from the Wayback CDX index (status 200 captures only) and written; later
runs fetch exactly the captures it lists and check each file's SHA-256 against it. The "id_" form of the Wayback URL
returns the archived bytes unmodified. Files already present are kept.
"""
import csv
import datetime as dt
import hashlib
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(RAW, "archive_csv")
MANIFEST = os.path.join(OUT, "manifest.csv")
TARGET = "https://www.accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm"
CDX = "https://web.archive.org/cdx/search/cdx"


def cdx_list():
    for attempt in range(8):
        try:
            r = requests.get(CDX, params=dict(url=TARGET.replace("https://", ""), output="json",
                                              fl="timestamp,statuscode,mimetype,digest"), timeout=300)
            r.raise_for_status(); break
        except requests.RequestException:
            if attempt == 7: raise
            time.sleep(60 * (attempt + 1))
    rows = r.json()[1:]
    return [dict(timestamp=t, digest=d) for t, s, m, d in rows if s == "200"]


def fetch(ts):
    url = f"https://web.archive.org/web/{ts}id_/{TARGET}"
    for attempt in range(8):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(60 * (attempt + 1))   # the Wayback Machine refuses connections for a while after bursts
    raise RuntimeError(f"could not fetch {url}")


def main():
    os.makedirs(OUT, exist_ok=True)
    have_manifest = os.path.exists(MANIFEST)
    seed = os.path.join(OUT, "capture_list.csv")   # CDX listing saved on 21 Oct 2026 (the CDX server refuses bursts)
    entries = (list(csv.DictReader(open(MANIFEST))) if have_manifest else
               list(csv.DictReader(open(seed))) if os.path.exists(seed) else cdx_list())
    log = open(os.path.join(HERE, "data", "raw", "download_log.txt"), "a")
    bad = 0
    for e in entries:
        path = os.path.join(OUT, f"Drugshortages-wayback-{e['timestamp']}.csv")
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            b = fetch(e["timestamp"])
            open(path, "wb").write(b)
            log.write(f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')} https://web.archive.org/web/{e['timestamp']}id_/{TARGET}"
                      f" -> {os.path.relpath(path, RAW)} {len(b)} bytes (Internet Archive capture of the FDA Drug Shortage Database CSV)\n")
            log.flush()
            time.sleep(4)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if have_manifest and h != e["sha256"]:
            print("SHA-256 mismatch", path); bad += 1
        e["sha256"] = h
    log.close()
    if not have_manifest:
        with open(MANIFEST, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "digest", "sha256"]); w.writeheader()
            for e in entries: w.writerow({k: e[k] for k in ("timestamp", "digest", "sha256")})
    print(f"{len(entries)} captures in {OUT}; {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
