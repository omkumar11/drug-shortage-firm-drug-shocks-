"""
leave_one_out.py -- how much the headline statistics depend on single firms (editor, paper-v5 round 4). For each of the
five firms listed for the most drugs on the analysis snapshot, dropped one at a time: the firm dispersion ratio, the
firm-sharing covariance, the same-firm conditional co-unavailability (main outcome, Section 3.2), and the first-listing
odds ratio of the quarterly panel (Section 3.3, with a firm-cluster bootstrap interval). Writes results/leave_one_out.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, RESULTS  # noqa: E402
import analysis.descriptives as de  # noqa: E402
from analysis import long_panel as lp  # noqa: E402

SEED = 20261022


def snapshot_stats(cells):
    df = cells[["firm_unit", "drug", "unavailable"]].reset_index(drop=True)
    v, e, _ = de.dispersion_stat(df, "firm_unit")
    sc = de.sharing_covariances(df); co = de.conditional_co_unavailability(df)
    return dict(n_cells=int(len(df)), firm_ratio=v / e, cov_firm=sc["cov_shared_firm"],
                co_same_firm=co["same_firm"]["p_second_given_first"], co_unrelated=co["unrelated"]["p_second_given_first"])


def main():
    rng = np.random.default_rng(SEED)
    cells, _, _, _ = load_cells("company")
    top = list(cells.groupby("firm_unit")["drug"].nunique().sort_values(ascending=False).index[:5])
    P, grid, _ = lp.load(); W = lp.grid_cells(P, grid)
    out = dict(all=snapshot_stats(cells), firms=top, drop={})
    fl = lp.first_listings(W, grid, P, rng, n_perm=1, n_boot=500)
    out["all"].update(fl_or=fl["or_fs"], fl_lo=fl["or_lo"], fl_hi=fl["or_hi"], fl_n=fl["n_new"])
    for f in top:
        r = snapshot_stats(cells[cells["firm_unit"] != f])
        Wf = {k: v[v["firm_unit"] != f].reset_index(drop=True) for k, v in W.items()}
        fl = lp.first_listings(Wf, grid, P, rng, n_perm=1, n_boot=500)
        r.update(fl_or=fl["or_fs"], fl_lo=fl["or_lo"], fl_hi=fl["or_hi"], fl_n=fl["n_new"])
        out["drop"][f] = r; print("without", f, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    with open(os.path.join(RESULTS, "leave_one_out.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
