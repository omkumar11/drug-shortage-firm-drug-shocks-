"""
coding_sample.py -- a random sample of free-text availability entries for the authors to code by hand (TODO_HUMAN.md), and
Cohen's kappa between their codes and the keyword coding of analysis/fda_csv.py.

    python analysis/coding_sample.py sample      # writes results/coding_sample.csv (200 entries, blank 'hand' column)
    python analysis/coding_sample.py kappa       # reads the filled file and prints agreement and kappa
Each author codes the 'hand' column independently (1 = unavailable or limited now, 0 = available now, blank = cannot tell),
without looking at the 'keyword' column, which is hidden in the sample file and stored in results/coding_sample_key.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import RESULTS  # noqa: E402

N = 200
SEED = 20261023


def sample():
    cod = pd.read_csv(os.path.join(RESULTS, "fda_csv_availability_coding.csv"))
    ft = cod[(cod["categorical"] == 0) & cod["availability_text"].notna()]
    # a simple random sample of the distinct free-text strings
    s = ft.sample(n=N, random_state=SEED).reset_index(drop=True)
    s.insert(0, "id", np.arange(1, N + 1))
    s[["id", "availability_text"]].assign(hand_author1="", hand_author2="").to_csv(os.path.join(RESULTS, "coding_sample.csv"), index=False)
    s[["id", "coded"]].rename(columns={"coded": "keyword"}).to_csv(os.path.join(RESULTS, "coding_sample_key.csv"), index=False)
    print("wrote", N, "entries")


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b); po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in np.union1d(a, b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score():
    s = pd.read_csv(os.path.join(RESULTS, "coding_sample.csv")).merge(pd.read_csv(os.path.join(RESULTS, "coding_sample_key.csv")), on="id")
    for col in ("hand_author1", "hand_author2"):
        m = s[s[col].notna() & s["keyword"].notna()]
        print(col, "n", len(m), "agreement", round((m[col] == m["keyword"]).mean(), 3), "kappa", round(kappa(m[col], m["keyword"]), 3))
    m = s[s["hand_author1"].notna() & s["hand_author2"].notna()]
    print("between authors: n", len(m), "kappa", round(kappa(m["hand_author1"], m["hand_author2"]), 3))
    # summary for make_numbers.py (macros HandCodeN, HandCodeAgreePct, HandCodeKappa use the first filled column)
    out = {}
    for col in ("hand_author1", "hand_author2"):
        mm = s[s[col].notna() & s["keyword"].notna()]
        if len(mm):
            out[col] = {"n": int(len(mm)), "agreement": float((mm[col] == mm["keyword"]).mean()), "kappa": float(kappa(mm[col], mm["keyword"])),
                        "disagree_ids": [int(i) for i in mm.loc[mm[col] != mm["keyword"], "id"]]}
    if len(m):
        out["between"] = {"n": int(len(m)), "kappa": float(kappa(m["hand_author1"], m["hand_author2"]))}
    import json
    json.dump(out, open(os.path.join(RESULTS, "coding_sample_kappa.json"), "w"), indent=1)


if __name__ == "__main__":
    {"sample": sample, "kappa": score}[sys.argv[1]]()
