"""
figures.py -- all manuscript figures from the processed data and results JSONs. Vector PDF + 600 dpi PNG,
grayscale-safe (hatching / marker shapes carry the information, colours are secondary).
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from analysis.common import load_cells, RESULTS, FIGS  # noqa: E402
from model import twotier as tt  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100})
DARK = "#222222"; MID = "#7a7a7a"; LIGHT = "#c8c8c8"


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def short_drug(d):
    for k, v in ((" Injection", " inj."), (" Hydrochloride", " HCl"), ("Monohydrate", ""), ("Sodium Phosphate", "Na phos."), ("Tablet, Chewable", "chew. tab."),
                 ("Tablet, Extended Release", "ER tab."), ("Film, Extended Release", "ER film"),
                 ("Amphetamine Aspartate , Amphetamine Sulfate, Dextroamphetamine Saccharate, Dextroamphetamine Sulfate Tablet", "Mixed amphetamine salts tab."),
                 ("Capsule", "caps."), ("Dimesylate", "dimes.")):
        d = d.replace(k, v)
    return " ".join(d.split())[:42]


def fig1_incidence(cells):
    """Drug x firm incidence matrix for firms with >= 3 listed drugs and drugs with >= 3 listed firms, with three cell
    states: every presentation unavailable (strict outcome), some but not all unavailable, all available."""
    fk = cells.groupby("firm_unit")["drug"].transform("size") >= 3
    dk = cells.groupby("drug")["firm_unit"].transform("size") >= 3
    sub = cells[fk & dk]
    forder = sub.groupby("firm_unit")["unavailable"].agg(["size", "mean"]).sort_values(["size", "mean"], ascending=False).index
    dorder = sub.groupby("drug")["unavailable"].agg(["size", "mean"]).sort_values(["size", "mean"], ascending=False).index
    fidx = {f: j for j, f in enumerate(forder)}; didx = {d: i for i, d in enumerate(dorder)}
    fig, ax = plt.subplots(figsize=(6.8, 0.2 * len(dorder) + 2.2))
    for _, r in sub.iterrows():
        i, j = didx[r.drug], fidx[r.firm_unit]
        if r.all_unavailable == 1: fc, h = DARK, None
        elif r.unavailable == 1: fc, h = LIGHT, None
        else: fc, h = "white", "////"
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor=MID, linewidth=0.5, hatch=h))
    ax.set_xlim(0, len(forder)); ax.set_ylim(len(dorder), 0)
    # normalised firm names, title-cased as in the text and tables
    pretty = lambda n: re.sub(r"\b(Icu|Us|Usa)\b", lambda m: m.group(0).upper(), str(n).title()).replace("B Braun", "B. Braun")
    ax.set_xticks(np.arange(len(forder)) + 0.5); ax.set_xticklabels([pretty(f) for f in forder], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(dorder)) + 0.5); ax.set_yticklabels([short_drug(d) for d in dorder], fontsize=8)
    ax.set_ylabel("Drug (FDA shortage database generic name)"); ax.set_xlabel("Reporting firm")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=DARK, label="every presentation unavailable"))
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=LIGHT, edgecolor=MID, label="some presentations unavailable or limited"))
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor="white", edgecolor=MID, hatch="////", label="all available"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=7)
    save(fig, "fig1_incidence")


def fig2_dispersion(desc):
    """Observed vs permutation-null dispersion for firms and drugs, and the sharing covariances."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    labels = ["Across firms", "Across drugs"]; obs = [desc["dispersion_firm"]["observed_var"], desc["dispersion_drug"]["observed_var"]]
    null = [desc["perm_firm"]["null_mean"], desc["perm_drug"]["null_mean"]]; q95 = [desc["perm_firm"]["null_q95"], desc["perm_drug"]["null_q95"]]
    x = np.arange(2)
    ax.bar(x - 0.18, null, 0.34, color="white", edgecolor=DARK, hatch="////", label="permutation null (mean)")
    ax.bar(x + 0.18, obs, 0.34, color=DARK, label="observed")
    for i in range(2): ax.plot([x[i] - 0.36, x[i] + 0.36], [q95[i]] * 2, color=MID, linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Variance of unavailability share")
    ax.set_ylim(0, max(obs + q95) * 1.45)
    ax.legend(frameon=False, fontsize=7, loc="upper left", title="dashed: null 95th pct.", title_fontsize=7)
    ax.set_title("(a) Dispersion of shares", fontsize=9, loc="left")
    ax = axes[1]
    sc = desc["sharing_cov"]
    vals = [sc["cov_shared_firm"], sc["cov_shared_drug"], sc["cov_unrelated"]]
    bt = desc["sharing_cov_boot"]                                  # two-way pigeonhole bootstrap 95% intervals (Table 2)
    lo = [bt["cov_firm_lo"], bt["cov_drug_lo"], bt["cov_none_lo"]]; hi = [bt["cov_firm_hi"], bt["cov_drug_hi"], bt["cov_none_hi"]]
    ax.bar(np.arange(3), vals, color=[DARK, MID, LIGHT], edgecolor=DARK)
    ax.errorbar(np.arange(3), vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none", ecolor=DARK, capsize=3, linewidth=1)
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["same firm,\ndifferent\ndrug", "same drug,\ndifferent\nfirm", "unrelated"], fontsize=7.5)
    ax.set_ylabel("Mean pairwise covariance")
    ax.set_title("(b) Co-unavailability of cell pairs", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig2_dispersion")


def fig3_posteriors(fit):
    """Posterior distributions of the attributable fractions and transmission parameters (main fit and sensitivities)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.0))
    ax = axes[0]
    rows = [("Main", fit["main"]), ("Strict outcome", fit["sensitivity"]["strict_outcome"]), ("Labeler unit", fit["sensitivity"]["labeler_unit"]), ("Marketer of record", fit["sensitivity"]["marketer_unit"]),
            ("Corporate group", fit["sensitivity"]["corporate_group"]), ("Raw company string", fit["sensitivity"]["raw_company"]),
            ("Allocation = limited", fit["sensitivity"]["allocation_outcome"]), ("Any constraining note = limited", fit["sensitivity"]["note_outcome"]),
            ("Unavailable only", fit["sensitivity"]["unavailable_only"]), ("Shared labelers dropped", fit["sensitivity"]["no_shared_labelers"]), ("Majority outcome", fit["sensitivity"]["majority_outcome"]),
            ("Injectables only", fit["sensitivity"]["injectables_only"]), ("Firms and drugs with $\\geq$ 2 cells", fit["sensitivity"]["shared_cells_only"]), ("Beta(2,2) prior", fit["sensitivity"]["prior_beta22"]),
            ("Discontinued recoded as available", fit["sensitivity"]["no_discontinued"]), ("Per-presentation baseline", fit["sensitivity"]["per_presentation_baseline"])]
    for i, (name, s) in enumerate(rows):
        for k, off, mk, col in (("af_firm", -0.15, "o", DARK), ("af_market", 0.15, "s", "white")):
            ax.plot([s[k]["lo"], s[k]["hi"]], [i + off] * 2, color=DARK, linewidth=1)
            ax.plot(s[k]["median"], i + off, marker=mk, color=DARK, markerfacecolor=col, markersize=5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("Attributable fraction (posterior median, 95% CrI)"); ax.set_xlim(0, 1)
    ax.plot([], [], marker="o", color=DARK, label="firm shocks", linestyle="none"); ax.plot([], [], marker="s", color=DARK, markerfacecolor="white", label="drug-level shocks", linestyle="none")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); ax.set_title("(a) Attributable fractions", fontsize=9, loc="left")
    ax = axes[1]
    s = fit["main"]; names = [("eps", r"$\epsilon$ baseline"), ("a", "$a$ firm transmission"), ("b", "$b$ drug transmission"), ("phi", r"$\phi$ P(firm shock)"), ("mu", r"$\mu$ P(drug shock)")]
    for i, (k, lab) in enumerate(names):
        ax.plot([s[k]["lo"], s[k]["hi"]], [i, i], color=DARK, linewidth=1); ax.plot(s[k]["median"], i, "o", color=DARK, markersize=5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n[1] for n in names], fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Posterior median and 95% CrI"); ax.set_title("(b) Model parameters (main fit)", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig3_posteriors")


def fig4_validation(cells, fit):
    """Posterior firm-shock probability by manufacturer-reported reason type (unavailable cells)."""
    c = pd.read_csv(os.path.join(RESULTS, "cell_attribution.csv"))
    un = c[c["unavailable"] == 1]
    order = ["firm", "market", "other"]; labels = {"firm": "firm-type\nreason", "market": "drug-type\nreason", "other": "'Other'"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=False)
    for ax, col, ttl in ((axes[0], "attr_firm", "(a) Attribution to the firm cause"), (axes[1], "attr_market", "(b) Attribution to the drug cause")):
        data = [un.loc[un["reason_type"] == o, col].values for o in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color="white", linewidth=1.5), showfliers=False)
        for p in bp["boxes"]: p.set_facecolor(DARK)
        for i, dd in enumerate(data):
            ax.plot(np.full(len(dd), i + 1) + np.random.default_rng(1).uniform(-0.18, 0.18, len(dd)), dd, ".", color=MID, markersize=3, alpha=0.7)
        ax.set_xticks(range(1, 4)); ax.set_xticklabels([f"{labels[o]}\n(n={len(d)})" for o, d in zip(order, data)], fontsize=7)
        ax.set_ylim(-0.03, 1.03); ax.set_ylabel("Posterior attribution (Section 2.5)"); ax.set_title(ttl, fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig4_validation")


def fig5_recovery(fit):
    """Parameter-recovery study on the observed graph: coverage and interval width."""
    rec = fit["recovery"]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ks = list(tt.PARAMS); x = np.arange(len(ks))
    ax.bar(x, [rec["coverage"][k] for k in ks], color="white", edgecolor=DARK, hatch="////")
    ax.axhline(0.95, color=MID, linestyle="--", linewidth=1)
    for i, k in enumerate(ks): ax.text(i, rec["coverage"][k] + 0.02, f"w={rec['mean_width'][k]:.2f}\nb={rec['mean_bias'][k]:+.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([r"$\epsilon$", "$a$", "$b$", r"$\phi$", r"$\mu$"]); ax.set_ylim(0, 1.12)
    ax.set_ylabel("95% CrI coverage"); ax.set_title(f"Recovery study ({rec['n']} replicates)", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig5_recovery")


def fig6_schematic():
    """Two-tier model schematic (firms -> cells <- markets) drawn programmatically."""
    fig, ax = plt.subplots(figsize=(5.2, 2.4)); ax.axis("off")
    firms = ["Firm 1", "Firm 2", "Firm 3"]; drugs = ["Drug A", "Drug B"]
    fy = [2.5, 1.5, 0.5]; dy = [2.2, 0.8]
    for i, f in enumerate(firms):
        ax.add_patch(matplotlib.patches.Ellipse((0.6, fy[i]), 0.78, 0.42, facecolor="white", edgecolor=DARK)); ax.text(0.6, fy[i], f, ha="center", va="center", fontsize=7)
    for j, d in enumerate(drugs):
        ax.add_patch(plt.Rectangle((4.0, dy[j] - 0.2), 0.9, 0.4, facecolor="white", edgecolor=DARK)); ax.text(4.45, dy[j], d, ha="center", va="center", fontsize=7)
    cells = [(0, 0), (0, 1), (1, 0), (2, 1), (1, 1)]
    xs = {(0, 0): 2.2, (0, 1): 3.1, (1, 0): 1.85, (2, 1): 2.2, (1, 1): 2.9}    # placed so that no two edges cross
    for (i, j) in cells:
        y = (fy[i] + dy[j]) / 2; x = xs[(i, j)]
        ax.plot([0.99, x - 0.25], [fy[i], y], color=MID, linewidth=0.8); ax.plot([x + 0.25, 4.0], [y, dy[j]], color=MID, linewidth=0.8)
        ax.add_patch(plt.Rectangle((x - 0.25, y - 0.12), 0.5, 0.24, facecolor=LIGHT, edgecolor=DARK, linewidth=0.6))
        ax.text(x, y, r"$y_{%d%s}$" % (i + 1, "AB"[j]), ha="center", va="center", fontsize=6.5)
    ax.text(0.6, 2.95, r"tier 1: firms, $Z_f\sim$Bern($\phi$)", ha="center", fontsize=7); ax.text(2.5, 2.95, r"cells $y_{fm}$ (eq. 1)", ha="center", fontsize=7)
    ax.text(4.45, 2.95, r"tier 2: drugs, $W_m\sim$Bern($\mu$)", ha="center", fontsize=7)
    ax.set_xlim(0.1, 5.2); ax.set_ylim(0.2, 3.2)
    save(fig, "fig6_schematic")


def fig7_timeline(pres):
    """Presentations on the current list by initial posting year, split by availability."""
    d = pres.assign(year=pd.to_datetime(pres["initial_posting_date"]).dt.year)
    t = d.groupby(["year", "unavailable"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    ax.bar(t.index, t.get(0, 0), color="white", edgecolor=DARK, hatch="////", label="available")
    ax.bar(t.index, t.get(1, 0), bottom=t.get(0, 0), color=DARK, label="unavailable / limited")
    ax.set_xlabel("Year the drug's shortage was first posted (presentation count)"); ax.set_ylabel("Presentations")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); save(fig, "fig7_timeline")


def fig8_redundancy(theory):
    """Theorem 2 and Proposition 2 on the posterior draws of the strict-outcome fit, where O_n is a complete outage (no
    listed presentation available): (a) O_n against the independence benchmark M_1^n, with observed shares by number of
    listed firms; (b) joint outage lift of two drugs with three listed firms each, by number of shared firms."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    ax = axes[0]; n = np.arange(1, 9); o = theory["outage_strict"]
    med = np.array([x["median"] for x in o["O"]]); lo = np.array([x["lo"] for x in o["O"]]); hi = np.array([x["hi"] for x in o["O"]])
    ind = np.array([x["median"] for x in o["O_indep"]])
    ax.fill_between(n, lo, hi, color=LIGHT, alpha=0.6, linewidth=0)
    ax.plot(n, med, color=DARK, marker="o", markersize=4, label="model $O_n$ (95% band)")
    ax.plot(n, ind, color=MID, linestyle="--", linewidth=1, label="independence benchmark $M_1^n$")
    obs = o["observed_by_n"][:5]
    pos = [d for d in obs if d["all_unavailable"] > 0]
    ax.plot([d["n"] for d in pos], [d["all_unavailable"] / d["drugs"] for d in pos], linestyle="none", marker="x", color=DARK, markersize=6, label="observed share")
    zero = [d for d in o["observed_by_n"] if d["all_unavailable"] == 0]     # zero shares drawn as triangles on the axis floor
    ax.plot([d["n"] for d in zero], [1.15e-3] * len(zero), linestyle="none", marker="v", markerfacecolor="white", color=DARK, markersize=4,
            clip_on=False, label="observed share of zero (floor)")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.0); ax.set_xticks(n); ax.set_xlim(0.5, 8.5); ax.set_xlabel("Listed firms $n$ (observed: 6 = six or more)")
    ax.set_ylabel("$O_n$ = P(no listed firm available)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2); ax.set_title("(a) Redundancy within a drug", fontsize=9, loc="left")
    ax = axes[1]; k = np.arange(4); L = theory["outage_strict"]["lift_33"]
    for i, x in enumerate(L):
        ax.plot([i] * 2, [x["lo"], x["hi"]], color=DARK, linewidth=1)
    ax.plot(k, [x["median"] for x in L], marker="o", linestyle="none", color=DARK, markersize=5)
    ax.axhline(1, color=MID, linestyle=":", linewidth=1)
    ax.set_xticks(k); ax.set_xlabel("Listed firms shared by two drugs ($n=n'=3$)"); ax.set_ylabel("$J_k / (O_3)^2$")
    ax.set_title("(b) Shared suppliers couple outages", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig8_redundancy")


def fig9_long_panel(lp):
    """The panel 2019-2026: (a) firm and drug dispersion ratios at each grid date, filled when the size-stratified conditional
    permutation test gives p < 0.05, with the permutation null means; (b) the share of a firm's first listings that are
    unavailable, by the firm's unavailable share about a year earlier (firm-cluster bootstrap intervals)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]; rec = lp["recurrence"]
    t = [pd.Timestamp(k[:8]) for k in rec]
    for key, col, mk, lab in (("firm", DARK, "o", "firms"), ("drug", MID, "s", "drugs")):
        y = [rec[k][f"{key}_ratio"] for k in rec]; sig = [rec[k][f"{key}_p_size"] < 0.05 for k in rec]
        ax.plot(t, y, color=col, linewidth=0.8)
        ax.plot(t, [rec[k][f"{key}_null_ratio"] for k in rec], color=col, linewidth=0.6, linestyle="--")
        ax.plot([a for a, s_ in zip(t, sig) if s_], [b for b, s_ in zip(y, sig) if s_], linestyle="none", marker=mk, color=col, markersize=4, label=f"{lab}, $p_s<0.05$")
        ax.plot([a for a, s_ in zip(t, sig) if not s_], [b for b, s_ in zip(y, sig) if not s_], linestyle="none", marker=mk, markerfacecolor="white", color=col, markersize=4, label=f"{lab}, $p_s\\geq0.05$")
    ax.axhline(1, color=MID, linestyle=":", linewidth=1)
    ax.set_ylabel("Dispersion ratio"); ax.set_title("(a) Clustering at each grid date", fontsize=9, loc="left")
    ax.set_xlim(pd.Timestamp("2019-10-01"), pd.Timestamp("2026-12-31"))
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), handletextpad=0.3, columnspacing=0.8)
    ax = axes[1]; bins = lp["first_listings"]["main"]["bins"]
    x = np.arange(len(bins))
    for xi, b_ in zip(x, bins):
        ax.plot([xi, xi], [b_["lo95"], b_["hi95"]], color=DARK, linewidth=1)
        ax.text(xi, b_["lo95"] - 0.07, f"n={b_['n']}", ha="center", fontsize=6.5)
    ax.plot(x, [b_["rate"] for b_ in bins], marker="o", linestyle="none", color=DARK, markersize=4)
    ax.set_xticks(x); ax.set_xticklabels([f"{b_['lo']:.2f}\u2013{b_['hi']:.2f}" for b_ in bins], fontsize=7)
    ax.set_ylim(0, 1); ax.set_xlabel("Firm's unavailable share about a year earlier"); ax.set_ylabel("New listings unavailable")
    ax.set_title("(b) Firm record and its new listings", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig9_long_panel")


def main():
    cells, pres, drugs, firms = load_cells()
    desc = json.load(open(os.path.join(RESULTS, "descriptives.json")))
    fit = json.load(open(os.path.join(RESULTS, "model_fit.json")))
    fig6_schematic(); fig1_incidence(cells); fig2_dispersion(desc); fig3_posteriors(fit); fig4_validation(cells, fit); fig5_recovery(fit); fig7_timeline(pres)
    fig8_redundancy(json.load(open(os.path.join(RESULTS, "theory.json"))))
    if os.path.exists(os.path.join(RESULTS, "long_panel.json")):
        fig9_long_panel(json.load(open(os.path.join(RESULTS, "long_panel.json"))))
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
