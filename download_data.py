"""
download_data.py -- fetch the raw public files into data/raw/ and append the URL + UTC date to data/raw/download_log.txt.
Files already present are kept (delete them to force a fresh download; the manuscript's numbers correspond to the
export dates recorded in the log and in data/processed/build_summary.json).
"""
import datetime as dt
import os
import sys
import urllib.request

import requests

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "data", "raw")
os.makedirs(os.path.join(RAW, "archive"), exist_ok=True)

SOURCES = [
    ("https://download.open.fda.gov/drug/shortages/drug-shortages-0001-of-0001.json.zip", "drug-shortages-0001-of-0001.json.zip"),
    ("https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip", "drug-ndc-0001-of-0001.json.zip"),
    ("https://download.open.fda.gov/drug/enforcement/drug-enforcement-0001-of-0001.json.zip", "drug-enforcement-0001-of-0001.json.zip"),
    ("https://www.fda.gov/media/76860/download?attachment", "EOBZIP_2026_09.zip"),   # Orange Book data files (monthly)
    # two earlier copies of the shortage export held by the Internet Archive (the only captures of the file; analysis/panel.py);
    # the "id_" form returns the archived bytes unmodified. SHA-256 of the files used in the paper:
    #   20250629203306: b49d052a0ed95f3c1279ff18a5013e3113e2c2d98199de06017d3fd55343852e
    #   20260128115236: 1b8b00d3bdef7b9cbc1080e710731c179caad2879f57bfe1470fbd5aa3690994
    ("https://web.archive.org/web/20250629203306id_/https://download.open.fda.gov/drug/shortages/drug-shortages-0001-of-0001.json.zip",
     os.path.join("archive", "drug-shortages-wayback-20250629203306.json.zip")),
    ("https://web.archive.org/web/20260128115236id_/https://download.open.fda.gov/drug/shortages/drug-shortages-0001-of-0001.json.zip",
     os.path.join("archive", "drug-shortages-wayback-20260128115236.json.zip")),
]


def main():
    log = open(os.path.join(RAW, "download_log.txt"), "a")
    for url, name in SOURCES:
        path = os.path.join(RAW, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print("kept", name); continue
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        log.write(f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')} {url} -> {name} {len(r.content)} bytes\n")
        print("downloaded", name, len(r.content))
    log.close()
    # The FDA Data Dashboard "Entire Inspections Dataset" (https://datadashboard.fda.gov/InspectionsDataset.xlsx) refuses
    # requests made outside a browser session, so it is downloaded by hand into data/raw/InspectionsDataset.xlsx;
    # analysis/inspections.py writes and can read a filtered extract (inspections_drugs_extract.csv.gz) that ships with the code.
    if not (os.path.exists(os.path.join(RAW, "InspectionsDataset.xlsx")) or os.path.exists(os.path.join(RAW, "inspections_drugs_extract.csv.gz"))):
        print("WARNING: neither InspectionsDataset.xlsx nor inspections_drugs_extract.csv.gz is in data/raw; download the workbook from "
              "https://datadashboard.fda.gov/oii/cd/inspections.htm (Download Dataset > Entire Inspections Dataset) before analysis/inspections.py")


if __name__ == "__main__":
    main()
