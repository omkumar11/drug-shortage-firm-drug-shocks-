"""
sbc_prior_draws.py -- where the first SBC batch's top-tenth excess sits (Supplementary Section S1). For a weakly identified
quantity the posterior stays close to the prior, so the SBC rank of the true value is close to the prior quantile of the
true value, and the rank histogram of a batch largely reproduces the histogram of its own prior draws. This script
regenerates the prior draws of every batch from their seeds (analysis/sbc.py, draw_replicates; no sampler is run),
computes the prior quantile of each true value from a large independent prior sample, and records, per batch and quantity,
the number of true values in the top tenth of the prior, its binomial tail, and the correlation between the sampler's rank
(results/sbc.json) and the prior quantile. Writes results/sbc_prior_draws.json.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis import sbc  # noqa: E402
from analysis.common import load_cells, RESULTS  # noqa: E402
from analysis.fit_model import index_graph  # noqa: E402
from model import twotier as tt  # noqa: E402

N_PRIOR = 400_000


def main():
    res_sbc = json.load(open(os.path.join(RESULTS, "sbc.json")))
    cells, _, _, _ = load_cells(); fi, mi, F, M, _, _ = index_graph(cells)
    rng = np.random.default_rng(sbc.SEED + 99)
    prior = sbc.derived({k: rng.beta(1.0, 1.0, N_PRIOR) for k in tt.PARAMS})
    prior_sorted = {k: np.sort(prior[k]) for k in sbc.QUANTITIES}
    out = dict(n_prior=N_PRIOR, batches=[])
    for b, seed in enumerate(sbc.BATCH_SEEDS[:res_sbc["n_batches"]]):
        assert res_sbc["batches"][b]["seed"] == seed, "batch order differs from sbc.BATCH_SEEDS"
        reps = sbc.draw_replicates(fi, mi, F, M, res_sbc["n_sbc"], seed)
        truth = sbc.derived({k: np.array([r["truth"][k] for r in reps]) for k in tt.PARAMS})
        ranks = res_sbc["batches"][b]["ranks"]; n = len(reps); row = dict(seed=seed, n=n, quantities={})
        for k in sbc.QUANTITIES:
            qnt = np.searchsorted(prior_sorted[k], truth[k]) / N_PRIOR          # prior quantile of the true value
            top = int((qnt >= 0.9).sum())
            row["quantities"][k] = dict(prior_top_tenth=top, prior_top_tail=float(stats.binom.sf(top - 1, n, 0.1)),
                                        sampler_top_tenth=int(res_sbc["batches"][b]["top_bin"][k]),
                                        corr_rank_prior_quantile=float(np.corrcoef(ranks[k], qnt)[0, 1]))
        out["batches"].append(row)
    with open(os.path.join(RESULTS, "sbc_prior_draws.json"), "w") as f:
        json.dump(out, f, indent=1)
    for row in out["batches"]:
        print(row["seed"], {k: (v["prior_top_tenth"], v["sampler_top_tenth"], round(v["corr_rank_prior_quantile"], 2)) for k, v in row["quantities"].items()})


if __name__ == "__main__":
    main()
