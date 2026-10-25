"""
make_numbers.py -- collect every number the manuscript reports into results/results.json and write LaTeX macros
to results/numbers.tex (and copy it to ../manuscript/numbers.tex). The manuscript never types a result by hand.
"""
import datetime as dt
import importlib.metadata
import json
import numpy as np
import os
import platform
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); PROC = os.path.join(HERE, "data", "processed")
MANU = os.path.join(os.path.dirname(HERE), "manuscript")


def pretty_firm(name):
    """Title-case a normalised firm name for the text, keeping common initialisms upper-case."""
    out = str(name).title()
    for w in ("Icu", "Usa", "Llc", "Inc", "Gmbh", "Sa", "Pte", "Djb", "Btg", "Hq", "Wg"):
        out = re.sub(rf"\b{w}\b", w.upper(), out)
    return out.replace("B Braun", "B. Braun")


def and_join(names):
    """'A, B and C' for a list of firm names in running text."""
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def load(name):
    return json.load(open(os.path.join(RES, name)))


def uniformity_test_counts(ranks, L, n_bins=10):
    """Chi-square p-value of rank uniformity with the binning of analysis/sbc.py (for pooled subsets of batches)."""
    from scipy import stats
    ranks = np.asarray(ranks); n_vals = L + 1
    cuts = [int(np.floor(j * n_vals / n_bins)) for j in range(n_bins + 1)]
    counts = np.array([np.sum((ranks >= cuts[j]) & (ranks < cuts[j + 1])) for j in range(n_bins)])
    expected = np.array([(cuts[j + 1] - cuts[j]) / n_vals * len(ranks) for j in range(n_bins)])
    return float(stats.chisquare(counts, f_exp=expected).pvalue)


def main():
    desc = load("descriptives.json"); fit = load("model_fit.json"); reg = load("regressions.json")
    build = json.load(open(os.path.join(PROC, "build_summary.json")))
    R = {}
    # sample
    R["SnapshotDate"] = build["shortage_snapshot_date"]
    R["ShortageRecordsAll"] = build["shortage_records_all"]
    R["StatusCurrent"] = build["shortage_status_counts"]["Current"]; R["StatusDiscontinued"] = build["shortage_status_counts"]["To Be Discontinued"]; R["StatusResolved"] = build["shortage_status_counts"]["Resolved"]
    R["TBDInListedDrugs"] = build["tbd_records_in_listed_drugs"]
    R["NDCProducts"] = build["ndc_products_total"]; R["NDCRx"] = build["ndc_rx_approved_finished"]; R["NDCMarkets"] = build["ndc_markets"]; R["RecallReports"] = build["recall_reports"]; R["OrangeBookRows"] = build["orange_book_rows"]
    R["NPres"] = desc["n_presentations"]; R["NPresUnavail"] = desc["n_presentations_unavailable"]; R["NPresInNDC"] = build["current_presentations_in_ndc"]
    R["AvailAvailable"] = int(build["availability_counts"]["Available"]); R["AvailUnavailable"] = int(build["availability_counts"]["Unavailable"]); R["AvailLimited"] = int(build["availability_counts"]["Limited Availability"])
    R["NCells"] = desc["n_cells"]; R["NCellsUnavail"] = desc["n_cells_unavailable"]; R["CellRatePct"] = 100 * desc["cell_rate"]
    R["NDrugs"] = desc["n_drugs"]; R["NFirms"] = desc["n_firms"]; R["NLabelers"] = desc["n_labeler_codes"]; R["NDrugsInjectable"] = desc["n_drugs_injectable"]
    R["PresInjectablePct"] = 100 * desc["share_presentations_injectable"]
    R["FirmsTwoPlus"] = desc["firms_with_2plus_drugs"]; R["DrugsTwoPlus"] = desc["drugs_with_2plus_firms"]; R["DrugsSingleFirm"] = desc["drugs_single_firm"]
    R["MedianFirmsPerDrug"] = desc["median_firms_per_drug"]; R["MaxFirmsPerDrug"] = desc["max_firms_per_drug"]; R["MedianDrugsPerFirm"] = desc["median_drugs_per_firm"]
    R["TopFiveFirmsCellPct"] = 100 * desc["share_cells_top5_firms"]
    for i, t in enumerate(desc["top_firms"][:5]):
        R[f"TopFirm{'ABCDE'[i]}Name"] = t["firm"].title(); R[f"TopFirm{'ABCDE'[i]}N"] = t["n_drugs"]; R[f"TopFirm{'ABCDE'[i]}K"] = t["n_unavailable"]
    ru = desc["reason_counts_unavailable"]
    R["UReasonOther"] = ru.get("Other", 0); R["UReasonDemand"] = ru.get("Demand increase for the drug", 0); R["UReasonAPI"] = ru.get("Shortage of an active ingredient", 0)
    R["UReasonDiscont"] = ru.get("Discontinuation of the manufacture of the drug", 0); R["UReasonShipping"] = ru.get("Delay in shipping of the drug", 0); R["UReasonGMP"] = ru.get("Requirements related to complying with good manufacturing practices", 0)
    rc = desc["reason_counts"]; R["ReasonNone"] = rc.get("(none)", 0); R["ReasonOther"] = rc.get("Other", 0); R["ReasonDemand"] = rc.get("Demand increase for the drug", 0)
    R["ReasonAPI"] = rc.get("Shortage of an active ingredient", 0); R["ReasonDiscont"] = rc.get("Discontinuation of the manufacture of the drug", 0); R["ReasonShipping"] = rc.get("Delay in shipping of the drug", 0)
    R["ReasonGMP"] = rc.get("Requirements related to complying with good manufacturing practices", 0); R["ReasonInactive"] = rc.get("Shortage of an inactive ingredient component", 0); R["ReasonRegulatory"] = rc.get("Regulatory delay", 0)
    R["ReasonNoneAvailPct"] = 100 * (1 - desc["reason_unavailable_rate"].get("(none)", 0))
    R["MedianAppsInMarket"] = desc["median_applications_in_market"]; R["MedianFirmsListed"] = desc["median_firms_listed"]
    R["PostingMin"] = desc["first_posting_min"]; R["PostingMax"] = desc["first_posting_max"]; R["PresBeforeTwentyOnePct"] = 100 * desc["share_presentations_posted_before_2021"]
    R["PostingSpreadZeroPct"] = 100 * desc["posting_spread"]["share_zero"]; R["PostingSpreadN"] = desc["posting_spread"]["n_drugs"]; R["PostingSpreadNonzeroN"] = desc["posting_spread"]["n_nonzero"]
    R["PostingSpreadMedianDays"] = desc["posting_spread"]["median_days_nonzero"]
    # model-free
    for nm, key in (("Firm", "firm"), ("Drug", "drug")):
        R[f"Disp{nm}Obs"] = desc[f"dispersion_{key}"]["observed_var"]; R[f"Disp{nm}Binom"] = desc[f"dispersion_{key}"]["binomial_var"]; R[f"Disp{nm}Ratio"] = desc[f"dispersion_{key}"]["ratio"]; R[f"Disp{nm}Groups"] = desc[f"dispersion_{key}"]["n_groups"]
        R[f"Perm{nm}NullMean"] = desc[f"perm_{key}"]["null_mean"]; R[f"Perm{nm}P"] = desc[f"perm_{key}"]["p_value"]; R[f"Perm{nm}Q"] = desc[f"perm_{key}"]["null_q95"]
    R["ChiFirm"] = desc["firm_homogeneity_chi2"]["chi2"]; R["ChiFirmDof"] = desc["firm_homogeneity_chi2"]["dof"]; R["ChiFirmP"] = desc["firm_homogeneity_chi2"]["p"]
    R["ChiDrug"] = desc["drug_homogeneity_chi2"]["chi2"]; R["ChiDrugDof"] = desc["drug_homogeneity_chi2"]["dof"]; R["ChiDrugP"] = desc["drug_homogeneity_chi2"]["p"]
    sc = desc["sharing_cov"]; R["CovFirm"] = sc["cov_shared_firm"]; R["CovDrug"] = sc["cov_shared_drug"]; R["CovNone"] = sc["cov_unrelated"]; R["PairsFirm"] = sc["n_pairs_firm"]; R["PairsDrug"] = sc["n_pairs_drug"]; R["PairsNone"] = sc["n_pairs_unrelated"]; R["VarY"] = sc["var_y"]
    R["CovRatio"] = sc["cov_shared_firm"] / sc["cov_shared_drug"] if sc["cov_shared_drug"] > 0 else float("nan")
    R["ExcessCovFirm"] = sc["cov_shared_firm"] - sc["cov_unrelated"]; R["ExcessCovDrug"] = sc["cov_shared_drug"] - sc["cov_unrelated"]
    bt = desc["sharing_cov_boot"]
    for k, nm in (("cov_firm", "CovFirm"), ("cov_drug", "CovDrug"), ("cov_none", "CovNone"), ("excess_firm", "ExcessCovFirm"), ("excess_drug", "ExcessCovDrug")):
        R[f"{nm}BLo"] = bt[f"{k}_lo"]; R[f"{nm}BHi"] = bt[f"{k}_hi"]
    R["BootN"] = bt["n_boot"]; R["PExcessDrugLeZero"] = bt["p_excess_drug_le_0"]; R["PCovFirmGtDrug"] = bt["p_firm_gt_drug"]
    for k, nm in (("perm_cov_firm", "PermCovFirm"), ("perm_cov_drug", "PermCovDrug")):
        R[f"{nm}NullMean"] = desc[k]["null_mean"]; R[f"{nm}Q"] = desc[k]["null_q95"]; R[f"{nm}P"] = desc[k]["p_value"]
    R["CovExclTopThree"] = desc["cov_excl_top3"]["cov_shared_firm"]; R["PairsExclTopThree"] = desc["cov_excl_top3"]["n_pairs_firm"]
    R["UnavailWithoutReason"] = desc["unavailable_without_reason"]; R["CarboplatinReasons"] = desc["carboplatin_reasons"]; R["CarboplatinFirmsUnavail"] = desc["carboplatin_firms_unavailable"]
    R["DrugsTwoPlusUnavail"] = desc["drugs_2plus_unavail_firms"]; R["DrugsMixedReasons"] = desc["drugs_mixed_reason_types"]; R["PostedSentinel"] = desc["presentations_posted_2012_01_01"]
    R["EnforcementExportDate"] = build.get("enforcement_export_date", ""); R["NDCExportDate"] = build.get("ndc_export_date", "")
    R["AvailWithReason"] = desc["available_with_reason"]; ar = desc["available_related_info"]
    R["AvailRelBlank"] = ar["blank"]; R["AvailRelWholesalers"] = ar["wholesalers"]; R["AvailRelAllocation"] = ar["allocation"]; R["AvailRelOther"] = ar["other"]
    R["CellsChangedAlloc"] = desc["cells_changed_by_allocation"]; R["LabelersMultiCompany"] = desc["labeler_codes_multi_company"]; R["CompaniesMultiLabeler"] = desc["companies_multi_labeler"]
    R["MedianLabelersInMarket"] = desc["median_labelers_in_market"]; R["RecallPreShareCellsPct"] = 100 * desc["recall_pre_share_cells"]
    h = desc["drug_share_hist"]; R["DrugShareAllAvail"] = h["all_available"]; R["DrugSharePartial"] = h["partial"]; R["DrugShareAllUnavail"] = h["all_unavailable"]
    for k, nm in (("injectable", "Inj"), ("controlled", "Ctrl"), ("recall", "Recall")):
        d = desc[f"unavailable_by_{k}"]
        for lvl, lab in (("0", "No"), ("1", "Yes"), ("False", "No"), ("True", "Yes")):
            if lvl in d: R[f"Rate{nm}{lab}Pct"] = 100 * d[lvl]["rate"]; R[f"N{nm}{lab}"] = d[lvl]["n"]
    # model
    m = fit["main"]
    for p, nm in (("eps", "Eps"), ("a", "A"), ("b", "B"), ("phi", "Phi"), ("mu", "Mu"), ("af_firm", "AFF"), ("af_market", "AFM"), ("marginal_rate", "Marg"), ("cov_shared_firm", "ModCovF"), ("cov_shared_market", "ModCovM")):
        R[f"{nm}Med"] = m[p]["median"]; R[f"{nm}Lo"] = m[p]["lo"]; R[f"{nm}Hi"] = m[p]["hi"]
    R["PFirmGtMarket"] = m["p_firm_gt_market"]; R["RhatMax"] = f"{max(m['rhat'].values()):.3f}"; R["AcceptRate"] = m["accept_rate"]
    for p, nm in (("nonfirm", "NonFirm"), ("unshocked_rate", "Unsh"), ("mu_b", "MuB"), ("phi_a", "PhiA")):
        R[f"{nm}Med"] = m[p]["median"]; R[f"{nm}Lo"] = m[p]["lo"]; R[f"{nm}Hi"] = m[p]["hi"]
    R["SPriorNonFirm"] = fit["sensitivity"]["prior_beta22"]["nonfirm"]["median"]; R["SPriorNonFirmLo"] = fit["sensitivity"]["prior_beta22"]["nonfirm"]["lo"]; R["SPriorNonFirmHi"] = fit["sensitivity"]["prior_beta22"]["nonfirm"]["hi"]
    R["NChains"] = fit["n_chains"]; R["NIter"] = fit["n_iter"]; R["Burn"] = fit["burn"]; R["Thin"] = fit["thin"]
    R["NFirmsZHalf"] = fit["n_firms_zpost_gt_0.5"]; R["NMarketsWHalf"] = fit["n_markets_wpost_gt_0.5"]
    pp = fit["ppc"]; R["PPCFirmP"] = pp["ppc_firm_p"]; R["PPCDrugP"] = pp["ppc_drug_p"]; R["PPCFirmVar"] = pp["ppc_firm_var_mean"]; R["PPCDrugVar"] = pp["ppc_drug_var_mean"]; R["PPCRate"] = pp["ppc_rate_mean"]
    rv = fit["reason_validation"]; ca = rv["cell_attribution_means"]
    R["AttrFirmOnlyPct"] = 100 * ca["firm_only"]; R["AttrMarketOnlyPct"] = 100 * ca["market_only"]; R["AttrBothPct"] = 100 * ca["both"]; R["AttrBasePct"] = 100 * ca["baseline_only"]
    for k, nm in (("firm", "RFirm"), ("market", "RMarket"), ("other", "ROther"), ("none", "RNone")):
        if k in rv["by_reason"]:
            R[f"{nm}N"] = rv["by_reason"][k]["n"]; R[f"{nm}AttrF"] = rv["by_reason"][k]["attr_firm"]; R[f"{nm}AttrM"] = rv["by_reason"][k]["attr_market"]
    if "mwu_attr_firm" in rv:
        R["MWUFirmP"] = rv["mwu_attr_firm"]["p"]; R["MWUFirmAUC"] = rv["mwu_attr_firm"]["auc"]; R["MWUMarketP"] = rv["mwu_attr_market"]["p"]; R["MWUMarketAUC"] = rv["mwu_attr_market"]["auc"]
        R["DiffAttrF"] = rv["diff_attr_firm"]["diff"]; R["DiffAttrFLo"] = rv["diff_attr_firm"]["lo"]; R["DiffAttrFHi"] = rv["diff_attr_firm"]["hi"]
    rec = fit["recovery"]; R["RecN"] = rec["n"]; R["RecAFCov"] = round(100 * rec["af_diff_coverage"])
    for sc, nm in (("recovery", "Rec"), ("recovery_swapped", "RecSw"), ("recovery_symmetric", "RecSym"), ("recovery_nofirm", "RecNF")):
        rr = fit[sc]; R[f"{nm}AFFTrue"] = rr["af_truth"]["firm"]; R[f"{nm}AFMTrue"] = rr["af_truth"]["market"]; R[f"{nm}PMean"] = rr["p_firm_gt_market"]["mean"]; R[f"{nm}PMin"] = rr["p_firm_gt_market"]["min"]; R[f"{nm}PMax"] = rr["p_firm_gt_market"]["max"]
        R[f"{nm}PShareAbove"] = round(100 * rr["p_firm_gt_market"]["share_above_0_95"]); R[f"{nm}AFFEst"] = rr["af_firm_median_mean"]; R[f"{nm}AFMEst"] = rr["af_market_median_mean"]; R[f"{nm}AFCov"] = round(100 * rr["af_diff_coverage"])
    R["ESSMin"] = min(m["ess"].values()); R["ESSb"] = m["ess"]["b"]; R["ESSmu"] = m["ess"]["mu"]; R["ESSAFDiff"] = m["ess_af_diff"]; R["NDraws"] = m["n_draws"]; R["SensRhatMax"] = fit["sensitivity_rhat_max"]
    for k, nm in (("drugs_all_unavail", "DrugsAllUnavail"), ("drugs_all_avail", "DrugsAllAvail"), ("firms_all_unavail", "FirmsAllUnavail")):
        kk = k if k != "firms_all_unavail" else "firms_all_unavail_share"
        R[f"PPC{nm}Obs"] = pp[f"obs_{kk}"]; R[f"PPC{nm}Mean"] = pp[f"ppc_{kk}_mean"]; R[f"PPC{nm}P"] = pp[f"ppc_{k}_p"]
    if "mwu_attr_firm_market_first" in rv: R["MWUFirmMarketFirstP"] = rv["mwu_attr_firm_market_first"]["p"]; R["MWUFirmMarketFirstAUC"] = rv["mwu_attr_firm_market_first"]["auc"]
    for p, nm in (("eps", "Eps"), ("a", "A"), ("b", "B"), ("phi", "Phi"), ("mu", "Mu")):
        R[f"Rec{nm}Cov"] = round(100 * rec["coverage"][p]); R[f"Rec{nm}Width"] = rec["mean_width"][p]; R[f"Rec{nm}Bias"] = rec["mean_bias"][p]
    for s, nm in (("strict_outcome", "Strict"), ("labeler_unit", "Lab"), ("marketer_unit", "Mkt"), ("corporate_group", "Group"), ("raw_company", "Raw"), ("allocation_outcome", "Alloc"), ("note_outcome", "Note"), ("no_discontinued", "NoDisc"), ("majority_outcome", "Maj"), ("injectables_only", "Inj"), ("shared_cells_only", "Shared"), ("prior_beta22", "Prior"), ("per_presentation_baseline", "PerPres"), ("unavailable_only", "UO"), ("no_shared_labelers", "NoShLab"), ("strict_no_shared_labelers", "StrictNoShLab"), ("strict_corporate_group", "StrictGroup")):
        ss = fit["sensitivity"][s]
        R[f"S{nm}AFF"] = ss["af_firm"]["median"]; R[f"S{nm}AFFLo"] = ss["af_firm"]["lo"]; R[f"S{nm}AFFHi"] = ss["af_firm"]["hi"]
        R[f"S{nm}AFM"] = ss["af_market"]["median"]; R[f"S{nm}AFMLo"] = ss["af_market"]["lo"]; R[f"S{nm}AFMHi"] = ss["af_market"]["hi"]
        R[f"S{nm}P"] = ss["p_firm_gt_market"]; R[f"S{nm}Phi"] = ss["phi"]["median"]; R[f"S{nm}A"] = ss["a"]["median"]; R[f"S{nm}Rhat"] = max(ss["rhat"].values())
        if "n_cells" in ss: R[f"S{nm}N"] = ss["n_cells"]
        if "F" in ss: R[f"S{nm}F"] = ss["F"]
        if "rate" in ss: R[f"S{nm}RatePct"] = 100 * ss["rate"]
        if "cells_changed" in ss: R[f"S{nm}Changed"] = ss["cells_changed"]
        if "labelers_dropped" in ss: R[f"S{nm}Dropped"] = ss["labelers_dropped"]
    for i, t in enumerate(fit["top_firm_posteriors"][:5]):
        R[f"TopZ{'ABCDE'[i]}Name"] = str(t["firm"]).title(); R[f"TopZ{'ABCDE'[i]}Z"] = t["z_post"]; R[f"TopZ{'ABCDE'[i]}N"] = t["n_drugs"]; R[f"TopZ{'ABCDE'[i]}K"] = t["n_unavailable"]
    carb = [t for t in fit["top_market_posteriors"] if "carboplatin" in str(t["drug"]).lower()]
    R["CarboplatinWPost"] = carb[0]["w_post"] if carb else "n/a"; R["CarboplatinWPostK"] = carb[0]["n_unavailable"] if carb else "n/a"; R["CarboplatinWPostN"] = carb[0]["n_firms"] if carb else "n/a"
    for i, t in enumerate(fit["top_market_posteriors"][:5]):
        R[f"TopW{'ABCDE'[i]}Name"] = str(t["drug"]); R[f"TopW{'ABCDE'[i]}W"] = t["w_post"]; R[f"TopW{'ABCDE'[i]}N"] = t["n_firms"]; R[f"TopW{'ABCDE'[i]}K"] = t["n_unavailable"]
    # regressions
    R["RegN"] = reg["n"]; R["RegNUnavail"] = reg["n_unavailable"]; R["RecallShareFirmsPct"] = 100 * reg["recall_share_firms"]
    R["RecallShareCellsPct"] = 100 * desc["unavailable_by_recall"]["True"]["n"] / (desc["unavailable_by_recall"]["True"]["n"] + desc["unavailable_by_recall"]["False"]["n"])
    R["RefFirmsNoRecall"] = reg["n_reference_firms_no_recall"]; R["RefCellsNoRecall"] = reg["n_reference_cells_no_recall"]
    R["RecallBootLo"] = reg["recall_any_firm_bootstrap"]["or_lo"]; R["RecallBootHi"] = reg["recall_any_firm_bootstrap"]["or_hi"]; R["RecallBootN"] = reg["recall_any_firm_bootstrap"]["n_boot"]
    R["RecallDifferCells"] = reg["n_cells_recall_ext_vs_other_differ"]; R["RecallDifferUnavail"] = reg["n_cells_recall_differ_unavailable"]
    R["RefFirmsNoRecallExt"] = reg["n_reference_firms_no_recall_ext"]; R["RefFirmsNoRecallOther"] = reg["n_reference_firms_no_recall_other"]
    R["RegFirms"] = reg["m2_firm"]["n_firms"]; R["RegDrugs"] = reg["m2_firm"]["n_drugs"]
    R["CellsLeftCensored"] = reg["cells_left_censored"]; R["MinEigAll"] = reg["min_eigenvalue_all"]
    nd = reg["m2_no_dextrose"]; R["MTwoNoDextroseN"] = nd["n"]; R["MTwoNoDextroseDropped"] = nd["cells_dropped"]
    for c, nm in (("log_apps", "Apps"), ("recall_any", "Recall")):
        R[f"MTwoNoDextrose{nm}OR"] = nd["coefs"][c]["or_"]; R[f"MTwoNoDextrose{nm}ORLo"] = nd["coefs"][c]["or_lo"]; R[f"MTwoNoDextrose{nm}ORHi"] = nd["coefs"][c]["or_hi"]
    for spec, nm in (("m1_structure", "MOne"), ("m2_firm", "MTwo"), ("m3_firm_count", "MThree"), ("m4_age", "MFour"), ("m5_recall_pre", "MFive"), ("m6_labelers", "MSix"), ("m7_listed", "MSeven"), ("m8_nopres", "MEight"), ("m9_recall_ext", "MNine"), ("m10_recall_other", "MTen")):
        for var, vn in (("injectable", "Inj"), ("controlled", "Ctrl"), ("log_apps", "Apps"), ("log_labelers", "Labelers"), ("log_listed", "Listed"), ("log_pres", "Pres"), ("recall_any", "Recall"), ("recall_pre_any", "RecallPre"), ("recall_ext_any", "RecallExt"), ("recall_ext_other_any", "RecallOther"), ("log_recalls1", "LogRecall"), ("log_firm_markets", "FirmSize"), ("market_age", "Age")):
            c = reg[spec]["coefs"].get(var)
            if c: R[f"{nm}{vn}OR"] = c["or_"]; R[f"{nm}{vn}Lo"] = c["or_lo"]; R[f"{nm}{vn}Hi"] = c["or_hi"]
        R[f"{nm}N"] = reg[spec]["n"]
    cal = fit["calibration"]
    R["CalibPooledGeObsPct"] = 100 * cal["pooled_share_ge_obs"]; R["CalibMaxGeObsPct"] = 100 * cal["max_share_ge_obs"]
    R["CalibWorstLoPct"] = 100 * cal["worst_ci"][0]; R["CalibWorstHiPct"] = 100 * cal["worst_ci"][1]
    wn = max(cal["nulls"], key=lambda n: n["share_ge_obs"]); R["CalibWorstN"] = len(wn["p_values"]); R["CalibWorstK"] = wn["n_ge_obs"]
    R["CalibWorstPhi"] = cal["worst_truth"]["phi"]; R["CalibWorstMu"] = cal["worst_truth"]["mu"]; R["CalibWorstEps"] = cal["worst_truth"]["eps"]
    R["CalibPooledGtNinetyFivePct"] = 100 * cal["pooled_share_gt_095"]; R["CalibNNulls"] = cal["n_nulls"]; R["CalibNRep"] = cal["n_rep"]
    R["CalibMaxPMax"] = max(n["p_max"] for n in cal["nulls"]); R["CalibMinPMean"] = min(n["p_mean"] for n in cal["nulls"]); R["CalibMaxPMean"] = max(n["p_mean"] for n in cal["nulls"])
    R["CalibAFTrueMin"] = min(n["af_true"] for n in cal["nulls"]); R["CalibAFTrueMax"] = max(n["af_true"] for n in cal["nulls"])
    R["CalibAFDrugEstMin"] = min(n["af_drug_median_mean"] for n in cal["nulls"]); R["CalibAFDrugEstMax"] = max(n["af_drug_median_mean"] for n in cal["nulls"])
    R["CalibAFDLeObsPct"] = 100 * cal["share_afd_le_obs_all"]
    R["PriorOrderBetaOneP"] = fit["prior_p_order_beta11"]; R["PriorOrderBetaTwoP"] = fit["prior_p_order_beta22"]
    ppb = fit["sensitivity"]["per_presentation_baseline"]
    R["PerPresEpsMed"] = ppb["eps_per_presentation"]["median"]; R["PerPresEpsLo"] = ppb["eps_per_presentation"]["lo"]; R["PerPresEpsHi"] = ppb["eps_per_presentation"]["hi"]
    R["PerPresMeanCellSize"] = ppb["mean_cell_size"]; R["PerPresRhatMax"] = ppb["rhat_max"]; R["PerPresNFirmsZHalf"] = ppb["n_firms_zpost_gt_0.5"]
    ov = load("outcome_variants.json")
    for oc, on in (("main", "Main"), ("unavailable_only", "UO"), ("majority", "Maj"), ("strict", "Strict"), ("allocation", "Alloc"), ("note", "Note"), ("partial_only", "Part"), ("no_discontinued", "NoDisc"), ("no_discontinued_recoded", "NoDiscRec"), ("no_discontinued_ext", "NoDiscExt"), ("injectable", "Inj"), ("tbd_as_exit", "TBD"), ("wholesaler", "Whsl"),
                   ("unit_marketer", "Mkt"), ("unit_marketer_uo", "MktUO"), ("unit_labeler", "Lab"), ("unit_labeler_uo", "LabUO")):
        v = ov[oc]
        R[f"OV{on}NCells"] = v["n_cells"]; R[f"OV{on}CoSameFirmPct"] = 100 * v["co_same_firm"]; R[f"OV{on}CoUnrelatedPct"] = 100 * v["co_unrelated"]; R[f"OV{on}CoSameDrugPct"] = 100 * v["co_same_drug"]
        R[f"OV{on}FirmSizeP"] = v["perm_firm_p_size"]; R[f"OV{on}DrugSizeP"] = v["perm_drug_p_size"]
        R[f"OV{on}CovFirmSizeP"] = v["perm_cov_firm_p_size"]; R[f"OV{on}CovDrugSizeP"] = v["perm_cov_drug_p_size"]
        R[f"OV{on}RatePct"] = 100 * v["rate"]; R[f"OV{on}FirmP"] = v["perm_firm_p"]; R[f"OV{on}DrugP"] = v["perm_drug_p"]
        R[f"OV{on}FirmRatio"] = v["disp_firm_ratio"]; R[f"OV{on}DrugRatio"] = v["disp_drug_ratio"]
        R[f"OV{on}CovFirmMed"] = v["cov_firm"]; R[f"OV{on}CovDrugMed"] = v["cov_drug"]; R[f"OV{on}CovFirmP"] = v["perm_cov_firm_p"]; R[f"OV{on}CovDrugP"] = v["perm_cov_drug_p"]
    for oc, on in (("fully_out_main", "Main"), ("fully_out_strict", "Strict"), ("fully_out_main_size", "MainSize"), ("fully_out_strict_size", "StrictSize")):
        fo = ov[oc]; R[f"FullOut{on}Obs"] = fo["obs_total"]; R[f"FullOut{on}NullMean"] = fo["null_mean_total"]; R[f"FullOut{on}P"] = fo["p_total"]
        for j, w in enumerate(("Two", "Three", "FourPlus")):
            R[f"FullOut{on}Obs{w}"] = fo["obs"][j]; R[f"FullOut{on}Null{w}Mean"] = fo["null_mean"][j]
    ex = ov["extra"]
    R["MainFirmFineP"] = ex["main_firm_p_fine"]; R["MainDrugFineP"] = ex["main_drug_p_fine"]
    R["UpdateDaysMedian"] = int(ex["update_days_median"]); R["UpdateDaysMax"] = int(ex["update_days_max"])
    R["UpdateFirmRTwoPct"] = 100 * ex["update_days_firm_r2"]; R["UpdateCorrUnavailMed"] = ex["update_days_corr_unavailable"]
    R["SpillRhoMed"] = ex["spill_rho"]; R["SpillP"] = ex["spill_p"]; R["SpillNFirms"] = ex["spill_n_firms"]
    R["OVMainPermutableFirm"] = ov["main"]["permutable_cells_firm_test"]; R["OVMainPermutableDrug"] = ov["main"]["permutable_cells_drug_test"]
    R["OVMainSingletonFirm"] = ov["main"]["singleton_strata_firm_test"]; R["OVMainSingletonDrug"] = ov["main"]["singleton_strata_drug_test"]
    R["OVNonInjNCells"] = ov["main"]["n_cells"] - ov["injectable"]["n_cells"]
    R["OVInjFirms"] = ov["injectable"]["n_firms"]; R["OVInjDrugs"] = ov["injectable"]["n_drugs"]
    R["WhslCellsChanged"] = ov["wholesaler"]["n_cells_changed"]
    R["OVMktFirms"] = ov["unit_marketer"]["n_firms"]; R["OVLabFirms"] = ov["unit_labeler"]["n_firms"]
    R["TBDRecords"] = ov["tbd_as_exit"]["n_tbd_records"]; R["TBDFirms"] = ov["tbd_as_exit"]["n_tbd_firms"]; R["TBDPairs"] = ov["tbd_as_exit"]["n_tbd_pairs"]
    R["TBDNewCells"] = ov["tbd_as_exit"]["n_new_cells"]; R["TBDCellsTurned"] = ov["tbd_as_exit"]["n_cells_turned_unavailable"]
    R["SizeRTwoDrugPct"] = 100 * ov["size_share_drug_r2"]; R["SizeRTwoFirmPct"] = 100 * ov["size_share_firm_r2"]
    for c, w in (("1", "One"), ("2", "Two"), ("3", "ThreePlus")):
        R[f"StrictRateSize{w}Pct"] = 100 * ov["strict_rate_by_size"][c]; R[f"MainRateSize{w}Pct"] = 100 * ov["main_rate_by_size"][c]; R[f"NCellsSize{w}"] = ov["n_cells_by_size"][c]
    keys7 = ("main", "unavailable_only", "majority", "strict", "partial_only", "allocation", "note")
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "all seven"}
    keys5 = ("main", "unavailable_only", "majority", "strict", "partial_only")      # distinct definitions (allocation and note
    words5 = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "all five"}  # variants change 1 and 10 cells of main)
    R["OVFirmSizeSigCount"] = words5[sum(ov[k]["perm_firm_p_size"] < 0.05 for k in keys5)]
    R["OVDrugSizeSigCount"] = words5[sum(ov[k]["perm_drug_p_size"] < 0.05 for k in keys5)]
    R["OVCovFirmSizeSigCount"] = words5[sum(ov[k]["perm_cov_firm_p_size"] < 0.05 for k in keys5)]
    R["OVCovDrugSizeSigCount"] = words5[sum(ov[k]["perm_cov_drug_p_size"] < 0.05 for k in keys5)]
    # Holm adjustment of the size-stratified firm dispersion tests over the five distinct definitions (methods referee, v4)
    ps = sorted((ov[k]["perm_firm_p_size"], k) for k in keys5); holm = {}; run = 0.0
    for i, (pv, k) in enumerate(ps):
        run = max(run, min(1.0, (len(ps) - i) * pv)); holm[k] = run
    R["OVFirmSizeHolmSigCount"] = words5[sum(v < 0.05 for v in holm.values())]; R["OVMainFirmSizeHolmP"] = holm["main"]
    R["OVFirmSizeHolmMin"] = min(holm.values()); R["OVFirmSizeHolmMinName"] = {"main": "main", "unavailable_only": "Unavailable-only", "majority": "majority", "strict": "strict", "partial_only": "partial-only"}[min(holm, key=holm.get)]
    R["OVNPermDisp"] = ov["n_perm_dispersion"]; R["OVNPermCov"] = ov["n_perm_cov"]
    fine5 = {"main": "Main", "unavailable_only": "UO", "majority": "Maj", "strict": "Strict", "partial_only": "Part"}
    for k in keys5:                                                   # finer size strata (1, 2, 3-6, 7+) for the five distinct definitions
        if "perm_firm_p_fine" in ov[k]:
            R[f"OV{fine5[k]}FirmFineP"] = ov[k]["perm_firm_p_fine"]; R[f"OV{fine5[k]}DrugFineP"] = ov[k]["perm_drug_p_fine"]
    R["OVFineAvailable"] = int(all("perm_firm_p_fine" in ov[k] for k in keys5))
    rb = load("robustness_v4.json")                                # review-round-2 checks (robustness_v4.py)
    for oc, on in (("no_schedule2", "NoCII"), ("unit_group", "Grp"), ("unlisted_existing_firms", "UnlEx")):
        v = rb[oc]
        R[f"OV{on}NCells"] = v["n_cells"]; R[f"OV{on}RatePct"] = 100 * v["rate"]; R[f"OV{on}FirmRatio"] = v["disp_firm_ratio"]; R[f"OV{on}DrugRatio"] = v["disp_drug_ratio"]
        R[f"OV{on}FirmP"] = v["perm_firm_p"]; R[f"OV{on}DrugP"] = v["perm_drug_p"]; R[f"OV{on}FirmSizeP"] = v["perm_firm_p_size"]; R[f"OV{on}DrugSizeP"] = v["perm_drug_p_size"]
        R[f"OV{on}CovFirmMed"] = v["cov_firm"]; R[f"OV{on}CovDrugMed"] = v["cov_drug"]; R[f"OV{on}CovFirmP"] = v["perm_cov_firm_p"]; R[f"OV{on}CovDrugP"] = v["perm_cov_drug_p"]
        R[f"OV{on}CovFirmSizeP"] = v["perm_cov_firm_p_size"]; R[f"OV{on}CovDrugSizeP"] = v["perm_cov_drug_p_size"]
    R["NoCIIDrugsDropped"] = rb["no_schedule2"]["n_drugs_dropped"]; R["GrpFirms"] = rb["unit_group"]["n_firms"]; R["UnlExAdded"] = rb["unlisted_existing_firms"]["n_added_cells"]
    ui = rb["unlisted_imputed"]; R["UnlImpN"] = ui["n_imputations"]
    for k, nm in (("perm_firm_p_size", "FirmSizeP"), ("perm_cov_firm_p_size", "CovFirmSizeP"), ("perm_drug_p_size", "DrugSizeP")):
        R[f"UnlImp{nm}Med"] = ui[k]["median"]; R[f"UnlImp{nm}Lo"] = ui[k]["lo"]; R[f"UnlImp{nm}Hi"] = ui[k]["hi"]
    R["UnlImpCovFirmMed"] = ui["cov_firm"]["median"]; R["UnlImpCovDrugMed"] = ui["cov_drug"]["median"]
    R["CoSameFirmFirmWeightedPct"] = 100 * rb["co_same_firm_firm_weighted"]; R["CoFirmWeightedNFirms"] = rb["co_firm_weighted_n_firms"]
    R["CoFirmWeightedNullPct"] = 100 * rb["co_firm_weighted_null_mean"]; R["CoFirmWeightedP"] = rb["co_firm_weighted_p"]
    R["MultiFirmDrugs"] = rb["multi_firm_drugs"]; R["MultiFirmDrugsSomeAvail"] = rb["multi_firm_drugs_some_available"]
    # FDA inspection classifications linked to firms (inspections.py; Section 3.5): the hand-curated establishment linkage (primary,
    # no prefix) and the automatic name-rule sensitivity (prefix B)
    ins = load("inspections.json")
    R["InspWorkbookRows"] = ins["workbook_rows_total"]; R["InspDrugRows"] = ins["drug_rows"]; R["InspDrugRowsDQA"] = ins["drug_rows_dqa"]
    R["InspEndMin"] = ins["end_date_min"]; R["InspEndMax"] = ins["end_date_max"]; R["InspFYMin"] = ins["fiscal_year_min"]; R["InspFYMax"] = ins["fiscal_year_max"]
    R["InspWindowStart"] = ins["window_start"]; R["InspWindowEnd"] = ins["window_end"]
    R["InspWindowRows"] = ins["window_rows"]; R["InspWindowLegalNames"] = ins["window_legal_names"]; R["InspWindowFEI"] = ins["window_fei"]
    R["InspWindowOAIRows"] = ins["window_oai_rows"]; R["InspWindowVAIRows"] = ins["window_vai_rows"]; R["InspWindowForeignPct"] = 100 * ins["window_share_foreign"]
    R["InspFirms"] = ins["n_firms"]; R["InspCells"] = ins["n_cells"]
    for line in open(os.path.join(HERE, "data", "raw", "download_log.txt")):               # date the inspections workbook was downloaded
        if "InspectionsDataset.xlsx" in line and line[:4].isdigit():
            R["InspDownloadDate"] = dt.datetime.strptime(line.split()[0][:10], "%Y-%m-%d").strftime("%-d %B %Y")
    status_names = (("oai", "OAI"), ("inspected_no_oai", "InspNoOAI"), ("matched_no_window_inspection", "MatchedNoWindow"), ("unmatched", "Unmatched"))
    for label, P in (("curated", ""), ("name", "B")):
        o = ins[label]
        for k, nm in (("matched", "Matched"), ("inspected", "Inspected"), ("oai", "OAI"), ("vai_oai", "VAIOAI")):
            R[f"Insp{P}Firms{nm}"] = o[f"firms_{k}"]; R[f"Insp{P}Cells{nm}"] = o[f"cells_{k}"]
        R[f"Insp{P}FirmsMatchedPct"] = 100 * o["firms_matched"] / ins["n_firms"]; R[f"Insp{P}CellsMatchedPct"] = 100 * o["cells_matched"] / ins["n_cells"]
        R[f"Insp{P}FirmsChanged"] = o["firms_changed_by_union_rule"]; R[f"Insp{P}OAIFirmsMeanCells"] = o["oai_firms_mean_cells"]; R[f"Insp{P}OtherFirmsMeanCells"] = o["other_firms_mean_cells"]
        R[f"Insp{P}OAIFirmsFEIMax"] = o["oai_firms_fei_max"]; R[f"Insp{P}MatchedNotInspected"] = o["firms_matched_not_inspected_in_window"]
        for k, nm in status_names:
            v = o["cell_rate_by_status"].get(k, dict(n=0, n_unavailable=0, rate=float("nan"), n_firms=0))
            R[f"Insp{P}Rate{nm}Pct"] = 100 * v["rate"]; R[f"Insp{P}N{nm}"] = v["n"]; R[f"Insp{P}NUnavail{nm}"] = v["n_unavailable"]; R[f"Insp{P}NFirms{nm}"] = v["n_firms"]
            R[f"Insp{P}FW{nm}Pct"] = 100 * v.get("firm_weighted_rate", float("nan"))
            R[f"Insp{P}Strict{nm}Pct"] = 100 * o["cell_strict_rate_by_status"].get(k, float("nan")); R[f"Insp{P}UO{nm}Pct"] = 100 * o["cell_uo_rate_by_status"].get(k, float("nan"))
            R[f"Insp{P}NoDisc{nm}Pct"] = 100 * o["cell_nodisc_rate_by_status"].get(k, float("nan"))
        t3 = o["oai_cells_top3"]; R[f"Insp{P}TopThreeCells"] = t3["cells"]; R[f"Insp{P}TopThreeOf"] = t3["of"]; R[f"Insp{P}TopThreeFirms"] = and_join(pretty_firm(f) for f in t3["firms"])
        tc = o.get("oai_rate_by_insp_tercile", {})
        if tc:
            R[f"Insp{P}TercLowOAIPct"] = 100 * tc["low"]["oai_share"]; R[f"Insp{P}TercHighOAIPct"] = 100 * tc["high"]["oai_share"]
            R[f"Insp{P}TercLowMedInsp"] = tc["low"]["median_insp"]; R[f"Insp{P}TercHighMedInsp"] = tc["high"]["median_insp"]
        R[f"Insp{P}UnavailOAICells"] = o["oai_unavail_cells"]; R[f"Insp{P}UnavailOAIBeforeFirstOAI"] = o["oai_unavail_cells_posted_before_first_oai"]; R[f"Insp{P}UnavailOAIBeforeWindow"] = o["oai_unavail_cells_posted_before_window"]
        R[f"Insp{P}PrePostCells"] = o["cells_oai_pre_posting"]; R[f"Insp{P}PrePostFirms"] = o["firms_oai_pre_posting"]; R[f"Insp{P}PrePostRatePct"] = 100 * o["rate_oai_pre_posting"]; R[f"Insp{P}NotPrePostRatePct"] = 100 * o["rate_not_oai_pre_posting"]
        rg = o["regressions"]; R[f"Insp{P}RegN"] = rg["n"]; R[f"Insp{P}RegFirmsOAI"] = rg["n_firms_oai"]; R[f"Insp{P}RegCellsOAI"] = rg["n_cells_oai"]
        for spec, var, nm in (("m11_oai", "oai_any", "OAI"), ("m15_oai_ninsp", "oai_any", "OAINInsp"), ("m15_oai_ninsp", "log1p_insp", "NInsp"), ("m21_oai_listed", "oai_any", "OAIListed"),
                              ("m12_oai_inspected", "oai_any", "OAIInsp"), ("m12_oai_inspected", "inspected_any", "Inspected"), ("m13_oai_recall", "oai_any", "OAIRecall"),
                              ("m13_oai_recall", "recall_any", "RecallOAI"), ("m14_vai_oai", "vai_oai_any", "VAIOAI"), ("m16_oai_share", "oai_share", "OAIShare"),
                              ("m17_oai_pre_posting", "oai_pre_posting", "PrePost"), ("m18_oai_strict", "oai_any", "OAIStrict"), ("m19_oai_uo", "oai_any", "OAIUO"),
                              ("m20_oai_nodisc", "oai_any", "OAINoDisc"), ("m11_oai", "log_firm_markets", "OAIFirmSize"), ("m22_oai_fei", "oai_any", "OAIFEI"), ("m22_oai_fei", "log1p_fei", "NFEI")):
            c = rg[spec]["coefs"][var]; R[f"Insp{P}{nm}OR"] = c["or_"]; R[f"Insp{P}{nm}Lo"] = c["or_lo"]; R[f"Insp{P}{nm}Hi"] = c["or_hi"]
        R[f"Insp{P}OAIBootLo"] = rg["bootstrap"]["or_lo"]; R[f"Insp{P}OAIBootHi"] = rg["bootstrap"]["or_hi"]; R[f"Insp{P}OAIBootN"] = rg["bootstrap"]["n_boot"]
        R[f"Insp{P}OAIPermP"] = rg["firm_perm_oai"]["p"]; R[f"Insp{P}RecallPermP"] = rg["firm_perm_recall"]["p"]; R[f"Insp{P}PermN"] = rg["firm_perm_oai"]["n_perm"]
        R[f"Insp{P}OAIPermStratP"] = rg["firm_perm_oai_strat"]["p"]
        R[f"Insp{P}FirmsNoMatch"] = o.get("firms_curated_no_match", 0); R[f"Insp{P}FirmsNoPattern"] = o.get("firms_curated_no_pattern", 0)
        R[f"Insp{P}OAIPermSetsP"] = rg["firm_perm_oai_sets"]["p"]; R[f"Insp{P}EstabSetsOAI"] = o["n_establishment_sets_oai"]; R[f"Insp{P}EstabSets"] = o["n_establishment_sets"]
        for key, nm in (("no_503b", "NoFiveOThreeB"), ("short_window", "ShortWin")):
            if key in o:
                v = o[key]; c = v["m11_oai"]["coefs"]["oai_any"]
                R[f"Insp{nm}FirmsOAI"] = v["firms_oai"]; R[f"Insp{nm}CellsOAI"] = v["cells_oai"]; R[f"Insp{nm}FirmsInspected"] = v["firms_inspected"]; R[f"Insp{nm}WindowRows"] = v["window_rows"]
                R[f"Insp{nm}RateOAIPct"] = 100 * v["rate_oai"]; R[f"Insp{nm}RateOtherPct"] = 100 * v["rate_other"]
                R[f"Insp{nm}OAIOR"] = c["or_"]; R[f"Insp{nm}OAILo"] = c["or_lo"]; R[f"Insp{nm}OAIHi"] = c["or_hi"]; R[f"Insp{nm}Excluded"] = v["excluded_establishments"]
        R[f"Insp{P}RecallOAIBothFirms"] = o["recall_oai_overlap"]["firms_both"]
        sh = o["firm_share_by_oai"]; R[f"Insp{P}ShareOAIPct"] = 100 * sh["mean_a"]; R[f"Insp{P}ShareNoOAIPct"] = 100 * sh["mean_b"]; R[f"Insp{P}ShareAUC"] = sh["auc"]; R[f"Insp{P}ShareP"] = sh["p_one_sided"]
        z = o["firm_zpost_by_oai"]; R[f"Insp{P}ZOAIMed"] = z["median_a"]; R[f"Insp{P}ZNoOAIMed"] = z["median_b"]; R[f"Insp{P}ZAUC"] = z["auc"]; R[f"Insp{P}ZP"] = z["p_one_sided"]
        for oc, on in (("within_oai_firms", "WithinOAI"), ("within_non_oai_firms", "NoOAI"), ("no_vai_oai_firms", "NoVAIOAI")):
            v = o[oc]
            if "disp_firm_ratio" not in v:
                continue
            R[f"OV{P}{on}NCells"] = v["n_cells"]; R[f"OV{P}{on}RatePct"] = 100 * v["rate"]; R[f"OV{P}{on}FirmRatio"] = v["disp_firm_ratio"]; R[f"OV{P}{on}DrugRatio"] = v["disp_drug_ratio"]
            R[f"OV{P}{on}FirmP"] = v["perm_firm_p"]; R[f"OV{P}{on}DrugP"] = v["perm_drug_p"]; R[f"OV{P}{on}FirmSizeP"] = v["perm_firm_p_size"]; R[f"OV{P}{on}DrugSizeP"] = v["perm_drug_p_size"]
            R[f"OV{P}{on}CovFirmMed"] = v["cov_firm"]; R[f"OV{P}{on}CovDrugMed"] = v["cov_drug"]; R[f"OV{P}{on}CovFirmP"] = v["perm_cov_firm_p"]; R[f"OV{P}{on}CovDrugP"] = v["perm_cov_drug_p"]
            R[f"OV{P}{on}CovFirmSizeP"] = v["perm_cov_firm_p_size"]; R[f"OV{P}{on}CovDrugSizeP"] = v["perm_cov_drug_p_size"]; R[f"OV{P}{on}Firms"] = v["n_firms"]; R[f"OV{P}{on}FirmsDropped"] = v["n_firms_dropped"]
        mk = o["marketer"]; R[f"Insp{P}MktFirms"] = mk["n_firms"]; R[f"Insp{P}MktFirmsMatched"] = mk["firms_matched"]; R[f"Insp{P}MktFirmsOAI"] = mk["firms_oai"]; R[f"Insp{P}MktCellsOAI"] = mk["cells_oai"]; R[f"Insp{P}MktCells"] = mk["n_cells"]
        c = mk["m11_oai"]["coefs"]["oai_any"]; R[f"Insp{P}MktOAIOR"] = c["or_"]; R[f"Insp{P}MktOAILo"] = c["or_lo"]; R[f"Insp{P}MktOAIHi"] = c["or_hi"]
    # three snapshots of the shortage list (panel.py; Section 3.3, Proposition 3)
    pn = load("panel.json")
    for W_, w in (("A", "w1"), ("B", "w2"), ("C", "w3")):
        v = pn["waves"][w]
        R[f"Pan{W_}Date"] = v["date"]; R[f"Pan{W_}Pres"] = v["n_presentations"]; R[f"Pan{W_}UnavailPres"] = v["n_unavailable_presentations"]
        R[f"Pan{W_}Cells"] = v["n_cells"]; R[f"Pan{W_}UnavailCells"] = v["n_unavailable_cells"]; R[f"Pan{W_}Firms"] = v["n_firms"]; R[f"Pan{W_}Drugs"] = v["n_drugs"]
        R[f"Pan{W_}CellRatePct"] = 100 * v["n_unavailable_cells"] / v["n_cells"]
        if w in ("w1", "w2"):
            for k, nm in (("disp_firm_ratio", "FirmRatio"), ("perm_firm_p", "FirmP"), ("perm_firm_p_size", "FirmSizeP"),
                          ("disp_drug_ratio", "DrugRatio"), ("perm_drug_p", "DrugP"), ("perm_drug_p_size", "DrugSizeP"),
                          ("cov_firm", "CovFirm"), ("perm_cov_firm_p", "CovFirmP"), ("perm_cov_firm_p_size", "CovFirmSizeP"),
                          ("cov_drug", "CovDrug"), ("perm_cov_drug_p", "CovDrugP"), ("perm_cov_drug_p_size", "CovDrugSizeP")):
                R[f"Pan{W_}{nm}"] = v[f"mf_{k}"]
            R[f"Pan{W_}CoSameFirmPct"] = 100 * v["mf_co_same_firm"]; R[f"Pan{W_}CoSameDrugPct"] = 100 * v["mf_co_same_drug"]; R[f"Pan{W_}CoUnrelatedPct"] = 100 * v["mf_co_unrelated"]
    for ab, nm in (("w1_w2", "AB"), ("w2_w3", "BC"), ("w1_w3", "AC")):
        o = pn[f"overlap_{ab}"]
        R[f"PanCommon{nm}"] = o["cells_common"]; R[f"PanCommonShare{nm}Pct"] = 100 * o["share_of_later"]; R[f"PanAgree{nm}Pct"] = 100 * o["status_agreement"]
        R[f"PanChange{nm}Pct"] = 100 * (1 - o["status_agreement"])
    for ab, nm in (("w1_w2", "AB"), ("w2_w3", "BC")):
        fp = pn[f"firm_persistence_{ab}"]; R[f"PanFirmRho{nm}"] = fp["spearman"]; R[f"PanFirmRho{nm}P"] = fp["p"]; R[f"PanFirmRhoN{nm}"] = fp["n_firms"]
    R["OVNoDiscExtDelisted"] = ov["no_discontinued_ext"]["n_pres_delisted"]; R["OVNoDiscExtDelistedUnavail"] = ov["no_discontinued_ext"]["n_pres_delisted_unavailable"]
    R["OVNoDiscExtDropped"] = ov["no_discontinued_ext"]["n_cells_dropped"]
    tv = pn["transition_test_variants"]
    R["PanTransOneP"] = tv["t1_only"]["p_cross"]; R["PanTransTwoP"] = tv["t2_only"]["p_cross"]; R["PanTransNoTopP"] = tv["no_top3"]["p_cross"]; R["PanTransNDP"] = tv["no_discontinued"]["p_cross"]
    R["PanTransStayP"] = tv["start_unavailable"]["p_cross"]; R["PanTransBecomeP"] = tv["start_available"]["p_cross"]
    cs = pn["change_sensitivities"]
    for k, nm in (("interval_t1", "One"), ("interval_t2", "Two"), ("entries_exits", "EE"), ("no_discontinued", "ND"), ("no_schedule2", "NoCII"), ("no_top3", "NoTop")):
        v = cs[k]; R[f"PanChg{nm}FirmP"] = v["p_firm"]; R[f"PanChg{nm}DrugP"] = v["p_drug"]; R[f"PanChg{nm}CovFirm"] = v["cov_firm"]; R[f"PanChg{nm}CovDrug"] = v["cov_drug"]
        R[f"PanChg{nm}Changes"] = v["n_changes"]; R[f"PanChg{nm}N"] = v["n_cells"]
    R["PanChgEEEntries"] = cs["entries_exits"]["n_entries"]; R["PanChgEEExits"] = cs["entries_exits"]["n_exits"]
    R["PanChgEEEntryOnsets"] = cs["entries_exits"]["n_entry_onsets"]; R["PanChgEEExitRecoveries"] = cs["entries_exits"]["n_exit_recoveries"]
    R["PanChgNoCIIDrugs"] = cs["no_schedule2"]["n_drugs_dropped"]
    R["PanTopThreeFirms"] = and_join(pretty_firm(f) for f in cs["top3_firms"]); R["PanTopThreePairPct"] = 100 * cs["top3_pair_share"]
    R["PanRatioNoTopAB"] = cs["ratio_no_top3_w1_w2"]; R["PanRatioNoTopBC"] = cs["ratio_no_top3_w2_w3"]
    for W_, w in (("A", "w1"), ("B", "w2")):
        v = pn["waves"][w]
        R[f"Pan{W_}NDFirmRatio"] = v["nd_disp_firm_ratio"]; R[f"Pan{W_}NDFirmSizeP"] = v["nd_perm_firm_p_size"]; R[f"Pan{W_}NDCovFirmSizeP"] = v["nd_perm_cov_firm_p_size"]
        R[f"Pan{W_}DroppedUnknown"] = v["n_dropped_unknown"]; R[f"Pan{W_}DiscPres"] = v["n_discontinued_pres"]
    dy0 = pn["dynamic_sim"]; R["PanDynObsOnsetOR"] = dy0["observed_onset_or"]; R["PanDynObsOnsetFirmP"] = dy0["observed_onset_firm_p"]
    for sc, nm in (("firm_persistent", "Persist"), ("firm_09", "NineTenths"), ("firm_07", "SevenTenths"), ("firm_half", "Half"), ("firm_renewed", "Renew"), ("both_persistent", "Both")):
        v = dy0[sc]; R[f"PanDyn{nm}OnsetORMed"] = v["onset_or_median"]; R[f"PanDyn{nm}OnsetORLo"] = v["onset_or_lo"]; R[f"PanDyn{nm}OnsetORHi"] = v["onset_or_hi"]
        R[f"PanDyn{nm}OnsetFirmPct"] = 100 * v["reject_onset_firm"]; R[f"PanDyn{nm}OnsetLeObsPct"] = 100 * v["share_onset_or_le_obs"]
    tt = pn["transition_test"]
    R["PanTransCross"] = tt["cross_cov"]; R["PanTransCrossNull"] = tt["cross_null_mean"]; R["PanTransCrossP"] = tt["p_cross"]
    R["PanTransLater"] = tt["later_cov"]; R["PanTransLaterNull"] = tt["later_null_mean"]; R["PanTransLaterP"] = tt["p_later"]
    R["PanStayUnavailPct"] = 100 * tt["share_unavailable_stay"]; R["PanStayAvailPct"] = 100 * tt["share_available_stay"]
    lu = pn["labeler_unit"]
    ts = pn["transition_test_size"]; R["PanTransSizeCrossP"] = ts["p_cross"]; R["PanTransSizeLaterP"] = ts["p_later"]; R["PanTransSizeCrossNull"] = ts["cross_null_mean"]
    R["PanLabTransSizeCrossP"] = lu["transition_test_size"]["p_cross"]; R["PanLabTransSizeLaterP"] = lu["transition_test_size"]["p_later"]
    R["PanLabTransCrossP"] = lu["transition_test"]["p_cross"]; R["PanLabTransLaterP"] = lu["transition_test"]["p_later"]
    R["PanLabChgFirmP"] = lu["change"]["p_firm"]; R["PanLabChgDrugP"] = lu["change"]["p_drug"]
    R["PanLabRatioAB"] = lu["ratios"]["w1_w2"]; R["PanLabRatioBC"] = lu["ratios"]["w2_w3"]; R["PanLabRatioAC"] = lu["ratios"]["w1_w3"]
    au = pn["always_unavailable"]; R["PanAlwaysN"] = au["n_cells"]; R["PanAlwaysTopFirms"] = and_join(pretty_firm(f) for f in au["top_firms"])
    R["PanAlwaysTopN"] = sum(au["top_firms"].values())
    for W_, w in (("A", "w1"), ("B", "w2"), ("C", "w3")):
        R[f"Pan{W_}MaxPosting"] = pn["waves"][w]["max_posting_date"]
    R["PanNameNearMatches"] = sum(pn[f"names_{k}_{ab}"]["near_matches"] for k in ("drug", "firm_unit") for ab in ("w1_w2", "w2_w3"))
    R["PanDrugsEnterAB"] = pn["names_drug_w1_w2"]["entering"]; R["PanDrugsLeaveAB"] = pn["names_drug_w1_w2"]["leaving"]
    R["PanDrugsEnterBC"] = pn["names_drug_w2_w3"]["entering"]; R["PanDrugsLeaveBC"] = pn["names_drug_w2_w3"]["leaving"]
    sy = pn["sync"]; R["PanSyncN"] = sy["n_cells"]; R["PanSyncOnsets"] = sy["n_onsets"]; R["PanSyncRecoveries"] = sy["n_recoveries"]; R["PanSyncUnchanged"] = sy["n_unchanged"]
    R["PanSyncEvents"] = sy["n_onsets"] + sy["n_recoveries"]
    for k, nm in (("change", "Change"), ("onset", "Onset"), ("recovery", "Recovery")):
        v = sy[k]
        R[f"PanSync{nm}CovFirm"] = v["cov_firm"]; R[f"PanSync{nm}CovFirmNull"] = v["cov_firm_null_mean"]; R[f"PanSync{nm}FirmP"] = v["p_firm"]; R[f"PanSync{nm}FirmSizeP"] = v["p_firm_size"]
        R[f"PanSync{nm}CovDrug"] = v["cov_drug"]; R[f"PanSync{nm}CovDrugNull"] = v["cov_drug_null_mean"]; R[f"PanSync{nm}DrugP"] = v["p_drug"]; R[f"PanSync{nm}DrugSizeP"] = v["p_drug_size"]
        R[f"PanSync{nm}PairsFirm"] = v["n_pairs_firm"]; R[f"PanSync{nm}PairsDrug"] = v["n_pairs_drug"]; R[f"PanSync{nm}N"] = v["n_cells"]
    dy = pn["dynamic_sim"]; R["PanDynNSim"] = dy["n_sim"]; R["PanDynNPerm"] = dy["n_perm"]; R["PanDynKeepPct"] = 100 * dy["observed_keep_share"]; R["PanDynObsRatio"] = dy["observed_ratio_pooled"]
    R["PanDynAMedian"] = dy["a_median"]; R["PanDynACal"] = dy["a_calibrated"]; R["PanDynObsCZero"] = dy["observed_same_date_cov"]; R["PanDynCZeroAtMedian"] = dy["same_date_cov_at_median"]
    for sc, nm in (("firm_persistent", "Persist"), ("firm_09", "NineTenths"), ("firm_07", "SevenTenths"), ("firm_half", "Half"), ("firm_renewed", "Renew"), ("both_persistent", "Both")):
        v = dy[sc]
        R[f"PanDyn{nm}FirmPct"] = 100 * v["reject_firm"]; R[f"PanDyn{nm}DrugPct"] = 100 * v["reject_drug"]; R[f"PanDyn{nm}Kappa"] = v["kappa"]
        R[f"PanDyn{nm}Ratio"] = v["ratio_mean"]; R[f"PanDyn{nm}RatioLo"] = v["ratio_lo"]; R[f"PanDyn{nm}RatioHi"] = v["ratio_hi"]; R[f"PanDyn{nm}GeObsPct"] = 100 * v["share_ratio_ge_obs"]
    for k, nm in (("onset", "Onset"), ("persistence", "Persist")):
        v = pn[k]
        R[f"Pan{nm}N"] = v["n_cells"]; R[f"Pan{nm}K"] = v["n_unavailable"]; R[f"Pan{nm}RatePct"] = 100 * v["rate"]; R[f"Pan{nm}NFirms"] = v["n_firms"]
        R[f"Pan{nm}OR"] = v["or_fs"]; R[f"Pan{nm}ORLo"] = v["or_fs_lo"]; R[f"Pan{nm}ORHi"] = v["or_fs_hi"]; R[f"Pan{nm}PermP"] = v["perm_p_one_sided"]
        R[f"Pan{nm}HighPct"] = 100 * v["rate_fs_high"]; R[f"Pan{nm}HighN"] = v["n_fs_high"]; R[f"Pan{nm}LowPct"] = 100 * v["rate_fs_low"]; R[f"Pan{nm}LowN"] = v["n_fs_low"]
        R[f"Pan{nm}DrugOR"] = v["or_ds"]; R[f"Pan{nm}PBootGtOne"] = v["p_boot_or_gt1"]; R[f"Pan{nm}PBootGtOnePct"] = 100 * v["p_boot_or_gt1"]
    for t, nm in (("t1", "One"), ("t2", "Two")):
        v = pn["onset_by_transition"][t]; R[f"PanOnset{nm}OR"] = v["or_fs"]; R[f"PanOnset{nm}PermP"] = v["perm_p_one_sided"]; R[f"PanOnset{nm}N"] = v["n_cells"]
        R[f"PanOnset{nm}HighPct"] = 100 * v["rate_fs_high"]; R[f"PanOnset{nm}LowPct"] = 100 * v["rate_fs_low"]
    R["PanNPermPred"] = pn["onset"]["n_perm"]; R["PanNBootPred"] = pn["onset"]["n_boot"]
    oo = pn["out_of_sample"]
    R["PanOosAUCBase"] = oo["auc_base"]; R["PanOosAUCBaseFs"] = oo["auc_base_fs"]; R["PanOosInc"] = oo["auc_increment"]; R["PanOosIncLo"] = oo["boot_inc_lo"]; R["PanOosIncHi"] = oo["boot_inc_hi"]
    R["PanOosAUCOnsetBase"] = oo["auc_base_onset"]; R["PanOosAUCOnsetBaseFs"] = oo["auc_base_fs_onset"]; R["PanOosIncOnset"] = oo["auc_increment_onset"]
    R["PanOosIncOnsetLo"] = oo["boot_inc_onset_lo"]; R["PanOosIncOnsetHi"] = oo["boot_inc_onset_hi"]; R["PanOosAUCFsAlone"] = oo["auc_fs_alone"]; R["PanOosAUCFsAloneOnset"] = oo["auc_fs_alone_onset"]
    R["PanOosNTrain"] = oo["n_train"]; R["PanOosNTest"] = oo["n_test"]; R["PanOosNOnset"] = oo["n_test_onset"]; R["PanOosNOnsetK"] = oo["n_test_onset_unavailable"]
    for ab, nm in (("w1_w2", "AB"), ("w2_w3", "BC"), ("w1_w3", "AC")):
        v = pn["persistence_states"][ab]
        R[f"PanRhoFirm{nm}"] = v["firm_ratio"]; R[f"PanRhoFirm{nm}Lo"] = v["firm_ratio_lo"]; R[f"PanRhoFirm{nm}Hi"] = v["firm_ratio_hi"]
        R[f"PanCFZero{nm}"] = v["firm_c0"]; R[f"PanCFLag{nm}"] = v["firm_c1"]; R[f"PanCFLag{nm}Lo"] = v["firm_c1_lo"]; R[f"PanCFLag{nm}Hi"] = v["firm_c1_hi"]
        R[f"PanCDZero{nm}"] = v["drug_c0"]; R[f"PanCDLag{nm}"] = v["drug_c1"]; R[f"PanCDLag{nm}Lo"] = v["drug_c1_lo"]; R[f"PanCDLag{nm}Hi"] = v["drug_c1_hi"]
        R[f"PanCDZero{nm}Lo"] = v["drug_c0_lo"]; R[f"PanCDZero{nm}Hi"] = v["drug_c0_hi"]; R[f"PanCNoneZero{nm}"] = v["none_c0"]; R[f"PanCNoneLag{nm}"] = v["none_c1"]
        R[f"PanPersistN{nm}"] = v["n_cells"]
    # leave-one-cell-out prediction and the demand-increase cells dropped (predictive_check.py; Sections 3.2 and 4.1)
    pv = load("predictive.json")
    for key, P in (("all", ""), ("firm_ge3", "Three")):
        v = pv["loo"][key]
        R[f"LOO{P}N"] = v["n_cells"]; R[f"LOO{P}NFirms"] = v["n_firms"]; R[f"LOO{P}RatePct"] = 100 * v["rate"]
        for pred, nm in (("firm_other", "Firm"), ("drug_other", "Drug"), ("both_pred", "Both"), ("size_pred", "Size"), ("size_drug_pred", "SizeDrug"), ("size_drug_firm_pred", "SizeDrugFirm")):
            R[f"LOO{P}AUC{nm}"] = v[f"auc_{pred}"]; R[f"LOO{P}Acc{nm}Pct"] = 100 * v[f"acc_{pred}"]
        R[f"LOO{P}AUCIncFirm"] = v["auc_increment_firm"]; R[f"LOO{P}PIncGtZero"] = v["p_firm_increment_gt0"]
        for nm, k in (("Firm", "firm"), ("Drug", "drug"), ("Both", "both"), ("Diff", "diff"), ("Inc", "inc")):
            R[f"LOO{P}AUC{nm}Lo"] = v[f"boot_{k}_lo"]; R[f"LOO{P}AUC{nm}Hi"] = v[f"boot_{k}_hi"]
        R[f"LOO{P}AccMajorityPct"] = 100 * v["acc_majority"]; R[f"LOO{P}PFirmGtDrug"] = v["p_firm_gt_drug"]; R[f"LOO{P}NBoot"] = v["n_boot"]
    vm = pv["loo_marketer"]["all"]; R["LOOMktAUCFirm"] = vm["auc_firm_other"]; R["LOOMktAUCDrug"] = vm["auc_drug_other"]; R["LOOMktAUCFirmLo"] = vm["boot_firm_lo"]; R["LOOMktAUCFirmHi"] = vm["boot_firm_hi"]
    dc = pv["discontinuation"]; R["DiscOnlyCells"] = dc["cells_only_through_discontinued"]; R["DiscWhollyCells"] = dc["cells_wholly_discontinued"]; R["DiscOnlyCellsOtherAvail"] = dc["only_disc_with_available_presentation"]
    nd = pv["no_demand_cells"]
    R["OVNoDemandNCells"] = nd["n_cells"]; R["OVNoDemandDropped"] = nd["n_cells_dropped"]; R["OVNoDemandDroppedUnavail"] = nd["n_unavailable_dropped"]; R["OVNoDemandRatePct"] = 100 * nd["rate"]
    R["OVNoDemandFirmRatio"] = nd["disp_firm_ratio"]; R["OVNoDemandFirmSizeP"] = nd["perm_firm_p_size"]; R["OVNoDemandCovFirmMed"] = nd["cov_firm"]; R["OVNoDemandCovFirmSizeP"] = nd["perm_cov_firm_p_size"]
    R["OVNoDemandDrugRatio"] = nd["disp_drug_ratio"]; R["OVNoDemandDrugSizeP"] = nd["perm_drug_p_size"]; R["OVNoDemandCovDrugMed"] = nd["cov_drug"]; R["OVNoDemandFirms"] = nd["n_firms"]
    # firm-level listing selection: unlisted approved suppliers added as available cells (unlisted_suppliers.py)
    us = load("unlisted_suppliers.json")
    R["PresReassignedMkt"] = us["n_pres_reassigned"]; R["UnavailPresReassignedMkt"] = us["n_unavail_pres_reassigned"]; R["UnavailCellsMkt"] = us["n_unavail_cells_marketer"]
    R["UnlAddedCells"] = us["n_added_cells"]; R["UnlAddedFirms"] = us["n_added_firms"]; R["UnlAddedFirmsNew"] = us["n_added_firms_new"]
    R["UnlDrugs"] = us["n_drugs_with_added"]; R["UnlApps"] = us["n_unlisted_applications"]; R["UnlAppsDiscn"] = us["n_discontinued"]; R["UnlAppsSameFirm"] = us["n_held_by_listed_firm"]
    for oc, on in (("main", "Unl"), ("unavailable_only", "UnlUO")):
        v = us[oc]
        R[f"OV{on}NCells"] = v["n_cells"]; R[f"OV{on}RatePct"] = 100 * v["rate"]; R[f"OV{on}FirmRatio"] = v["disp_firm_ratio"]; R[f"OV{on}DrugRatio"] = v["disp_drug_ratio"]
        R[f"OV{on}FirmP"] = v["perm_firm_p"]; R[f"OV{on}DrugP"] = v["perm_drug_p"]; R[f"OV{on}FirmSizeP"] = v["perm_firm_p_size"]; R[f"OV{on}DrugSizeP"] = v["perm_drug_p_size"]
        R[f"OV{on}CovFirmMed"] = v["cov_firm"]; R[f"OV{on}CovDrugMed"] = v["cov_drug"]; R[f"OV{on}CovFirmP"] = v["perm_cov_firm_p"]; R[f"OV{on}CovDrugP"] = v["perm_cov_drug_p"]
        R[f"OV{on}CovFirmSizeP"] = v["perm_cov_firm_p_size"]; R[f"OV{on}CovDrugSizeP"] = v["perm_cov_drug_p_size"]
        R[f"OV{on}CoSameFirmPct"] = 100 * v["co_same_firm"]; R[f"OV{on}CoUnrelatedPct"] = 100 * v["co_unrelated"]
    # theory (Section 2.5) and its empirical counterparts (theory_analysis.py)
    thr = load("theory.json")
    fp = thr["fullout_power"]
    for k, nm in (("main_fit", "MainFit"), ("strict_fit", "StrictFit"), ("mub_015", "MuBHalf"), ("swapped", "Swapped"), ("near_super_event", "Super")):
        R[f"Power{nm}Pct"] = 100 * fp[k]["power"]
    R["PowerNSim"] = fp["n_sim"]
    R["RowsFirmThree"] = thr["rows_ge3"]["firms"]; R["RowsDrugThree"] = thr["rows_ge3"]["drugs"]
    spd = load("sbc_prior_draws.json")                             # Supplementary S1: the first batch's prior draws
    b1 = spd["batches"][0]["quantities"]
    R["SBCPriorTopAFM"] = b1["af_drug"]["prior_top_tenth"]; R["SBCPriorTopAFMTail"] = b1["af_drug"]["prior_top_tail"]; R["SBCPriorTopMuB"] = b1["mu_b"]["prior_top_tenth"]
    R["SBCPriorTopAFMLater"] = ", ".join(str(bb["quantities"]["af_drug"]["prior_top_tenth"]) for bb in spd["batches"][1:])
    cors = [q["corr_rank_prior_quantile"] for bb in spd["batches"] for k, q in bb["quantities"].items() if k != "eps"]
    R["SBCPriorCorrMin"] = min(cors); R["SBCPriorCorrMax"] = max(cors)
    c1 = thr["corollary1"]                                          # Corollary 1 at the posterior (Section 3.3)
    for k, nm in (("co_same_firm", "CoSameFirm"), ("co_same_drug", "CoSameDrug"), ("co_unrelated", "CoUnrelated")):
        R[f"Cor{nm}Pct"] = 100 * c1[k]["median"]; R[f"Cor{nm}LoPct"] = 100 * c1[k]["lo"]; R[f"Cor{nm}HiPct"] = 100 * c1[k]["hi"]
    c2 = thr["corollary2"]                                          # Corollary 2 on the observed graph (Section 3.4)
    R["PortNPairsSharing"] = c2["n_pairs_sharing"]; R["PortNPairs"] = c2["n_pairs"]
    for k, nm in (("mean", "Mean"), ("sd", "SD"), ("sd_disjoint", "SDDisjoint"), ("var_ratio", "VarRatio")):
        R[f"Port{nm}Med"] = c2[k]["median"]; R[f"Port{nm}Lo"] = c2[k]["lo"]; R[f"Port{nm}Hi"] = c2[k]["hi"]
    for k, nm in (("ratio_firm", "RatioFirm"), ("ratio_drug", "RatioDrug"), ("co_same_firm", "CoSameFirm"), ("co_unrelated", "CoUnrelated")):
        pp = c1["ppc"][k]; sc = 100 if k.startswith("co") else 1
        R[f"CorPPC{nm}Lo"] = sc * pp["lo"]; R[f"CorPPC{nm}Hi"] = sc * pp["hi"]; R[f"CorPPC{nm}P"] = pp["p_upper"]
    R["CorPPCNRep"] = c1["ppc"]["n_rep"]
    R["CorPCovOrder"] = c1["p_cov_order_firm"]; R["CorPOrdersAgree"] = c1["p_orders_agree"]
    for side, nm in (("firm", "Firm"), ("drug", "Drug")):
        R[f"CorRatio{nm}Med"] = c1[f"ratio_{side}"]["median"]; R[f"CorRatio{nm}Lo"] = c1[f"ratio_{side}"]["lo"]; R[f"CorRatio{nm}Hi"] = c1[f"ratio_{side}"]["hi"]
    for oc, on in (("main", ""), ("strict", "Strict"), ("majority", "Maj"), ("unavailable_only", "UO")):
        mm = thr["mom"][oc]
        R[f"MoM{on}HankF"] = mm["firm_hankel"]; R[f"MoM{on}HankD"] = mm["drug_hankel"]
        R[f"MoM{on}FirmValidPct"] = 100 * (1 - mm["boot_share_degenerate_firm"])
        R[f"MoM{on}DegFPct"] = 100 * mm["boot_share_degenerate_firm"]; R[f"MoM{on}DegDPct"] = 100 * mm["boot_share_degenerate_drug"]
        R[f"MoM{on}HankDLeZeroPct"] = 100 * mm["boot_share_hankel_drug_le0"]
        if mm.get("boot_p_order") is not None: R[f"MoM{on}OrderP"] = mm["boot_p_order"]
        R[f"MoM{on}BothValidPct"] = 100 * mm["boot_share_both_valid"]; R[f"MoM{on}OrderAllPct"] = 100 * mm["boot_share_order_all"]
        R[f"MoM{on}MOneFirmRows"] = mm["m1_firm_rows"]; R[f"MoM{on}MOneDrugRows"] = mm["m1_drug_rows"]
        R[f"MoM{on}MOneDiffLo"] = mm["boot_m1_diff"]["lo"]; R[f"MoM{on}MOneDiffHi"] = mm["boot_m1_diff"]["hi"]
        if mm["firm"]:
            R[f"MoM{on}Phi"] = mm["firm"]["phi"]; R[f"MoM{on}PZero"] = mm["firm"]["p0"]; R[f"MoM{on}POne"] = mm["firm"]["p1"]; R[f"MoM{on}AFF"] = mm["firm"]["af_firm"]
        if mm["drug"]:
            R[f"MoM{on}Mu"] = mm["drug"]["mu"]; R[f"MoM{on}QZero"] = mm["drug"]["q0"]; R[f"MoM{on}QOne"] = mm["drug"]["q1"]; R[f"MoM{on}AFM"] = mm["drug"]["af_drug"]
        if mm.get("params"): R[f"MoM{on}Eps"] = mm["params"]["eps"]
        if mm.get("boot_share_eps_negative") is not None: R[f"MoM{on}EpsNegPct"] = 100 * mm["boot_share_eps_negative"]
        for k, nm in (("af_firm", "AFF"), ("af_drug", "AFM")):
            if k in mm["boot"]: R[f"MoM{on}{nm}Lo"] = mm["boot"][k]["lo"]; R[f"MoM{on}{nm}Hi"] = mm["boot"][k]["hi"]
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
    for oc, on in (("main", ""), ("strict", "Strict"), ("unavailable_only", "UO")):
        o = thr[f"outage_{oc}"]
        for n in range(1, 9):
            R[f"O{on}{words[n]}Med"] = o["O"][n - 1]["median"]; R[f"O{on}{words[n]}Lo"] = o["O"][n - 1]["lo"]; R[f"O{on}{words[n]}Hi"] = o["O"][n - 1]["hi"]
            R[f"OInd{on}{words[n]}Med"] = o["O_indep"][n - 1]["median"]
            R[f"ORatio{on}{words[n]}Med"] = o["O_ratio_to_indep"][n - 1]["median"]; R[f"ORatio{on}{words[n]}Lo"] = o["O_ratio_to_indep"][n - 1]["lo"]; R[f"ORatio{on}{words[n]}Hi"] = o["O_ratio_to_indep"][n - 1]["hi"]
        for k, nm in (("second_source_reduction", "Second"), ("fifth_source_reduction", "Fifth"), ("floor_reduction", "Floor"), ("q0", "QZeroPost"), ("q1", "QOnePost"),
                      ("logshare_firm", "LogShareF"), ("logshare_drug", "LogShareD"), ("logshare_base", "LogShareB")):
            R[f"{nm}{on}Med"] = o[k]["median"]; R[f"{nm}{on}Lo"] = o[k]["lo"]; R[f"{nm}{on}Hi"] = o[k]["hi"]
        R[f"PPhiAGtMuB{on}P"] = o["p_phia_gt_mub"]
        for a, an in ((0.1, "Ten"), (0.05, "Five")):
            ns = o[f"n_star_{a}"]
            R[f"NStar{an}{on}Mid"] = int(ns["median"]); R[f"NStar{an}{on}Low"] = int(ns["lo"]); R[f"NStar{an}{on}High"] = int(ns["hi"])
            R[f"NStarInd{an}{on}Mid"] = int(ns["indep"]["median"]); R[f"NStarInd{an}{on}Low"] = int(ns["indep"]["lo"]); R[f"NStarInd{an}{on}High"] = int(ns["indep"]["hi"])
        for k in range(4):
            R[f"Lift{on}{words.get(k, 'Zero') if k else 'Zero'}Med"] = o["lift_33"][k]["median"]; R[f"Lift{on}{words.get(k) if k else 'Zero'}Lo"] = o["lift_33"][k]["lo"]; R[f"Lift{on}{words.get(k) if k else 'Zero'}Hi"] = o["lift_33"][k]["hi"]
        for d in o["observed_by_n"]:
            w = words[d["n"]]
            R[f"ObsDrugs{on}{w}"] = d["drugs"]; R[f"ObsFull{on}{w}"] = d["all_unavailable"]; R[f"PPCFull{on}{w}Mean"] = d["ppc_mean"]
            R[f"PPCFull{on}{w}Low"] = int(d["ppc_lo"]); R[f"PPCFull{on}{w}High"] = int(round(d["ppc_hi"]))
        for d, w in zip(o["pairs"], ("Zero", "One", "TwoPlus")):
            R[f"PairsListed{on}{w}Mean"] = d["mean_listed"]; R[f"Pairs{on}{w}"] = d["pairs"]; R[f"PairsBoth{on}{w}"] = d["both_out"]; R[f"PairsPPC{on}{w}Mean"] = d["ppc_mean"]
            R[f"PairsPPC{on}{w}Low"] = int(d["ppc_lo"]); R[f"PairsPPC{on}{w}High"] = int(round(d["ppc_hi"]))
    # buyer-facing conditional co-unavailability (descriptives.conditional_co_unavailability)
    cu = desc["co_unavailability"]
    for k, nm in (("same_firm", "SameFirm"), ("same_drug", "SameDrug"), ("unrelated", "Unrelated")):
        R[f"CoUnavail{nm}Pct"] = 100 * cu[k]["p_second_given_first"]; R[f"CoUnavail{nm}BasePct"] = 100 * cu[k]["p_second"]
        R[f"CoUnavail{nm}Pairs"] = cu[k]["n_ordered_pairs_first_unavailable"]
    R["CoUnavailLiftPP"] = round(100 * cu["same_firm"]["p_second_given_first"]) - round(100 * cu["unrelated"]["p_second_given_first"])   # the two rounded figures of the text
    cx = desc["co_unavailability_excl_top3"]
    for k, nm in (("same_firm", "SameFirm"), ("same_drug", "SameDrug"), ("unrelated", "Unrelated")):
        R[f"CoUnavail{nm}ExclTopPct"] = 100 * cx[k]["p_second_given_first"]
    R["TopThreeFirms"] = and_join(pretty_firm(f) for f in cx["firms"]); R["TopThreeSameFirmPairSharePct"] = 100 * cx["share_same_firm_pairs_from_top3"]
    R["FirmTableMinDrugs"] = desc["firm_table_min_drugs"]; R["FirmTableN"] = len(desc["firm_table"])
    R["FirmTableMktN"] = len(desc["firm_table_marketer"]); R["NMarketers"] = desc["n_marketers"]; R["NCellsMarketer"] = desc["n_cells_marketer"]
    # rows of the firm-level tables, \input by main.tex (marketer of record, Section 3.2) and supplement.tex (reporting company)
    for fname, rows in (("firm_table.tex", desc["firm_table_marketer"]), ("firm_table_company.tex", desc["firm_table"])):
        with open(os.path.join(RES, fname), "w") as f:
            f.write("% generated by make_numbers.py -- do not edit\n")
            f.write("".join(f"{pretty_firm(r['firm'])} & {r['listed']} & {r['unavailable']} ({100 * r['unavailable'] / r['listed']:.0f}\\%) & {r['unavailable_only']} & {r['reason_firm']} & {r['reason_market']} & {r['reason_other'] + r['reason_none']}\\\\\n" for r in rows))
            f.write("\\bottomrule\n")                                  # the rule is in the file: a rule right after \input ends a row badly
    R["FirmTopFiveMarketMajorityN"] = len(desc["firm_table_top5_market_majority"]); R["FirmTopFiveMarketMajority"] = ", ".join(pretty_firm(f) for f in desc["firm_table_top5_market_majority"]) or "none"
    R["FirmTopFiveFirmMajorityN"] = len(desc["firm_table_top5_firm_majority"]); R["FirmTopFiveFirmMajority"] = ", ".join(pretty_firm(f) for f in desc["firm_table_top5_firm_majority"]) or "none"
    R["FirmTopFiveOtherMajorityN"] = len(desc["firm_table_top5_other_majority"]); R["FirmTopFiveOtherMajority"] = ", ".join(pretty_firm(f) for f in desc["firm_table_top5_other_majority"]) or "none"
    for src, nm in (("co_unavailability_unavailable_only", "UO"), ("co_unavailability_injectable", "Inj")):
        c = desc[src]
        R[f"CoUnavail{nm}SameFirmPct"] = 100 * c["same_firm"]["p_second_given_first"]; R[f"CoUnavail{nm}SameDrugPct"] = 100 * c["same_drug"]["p_second_given_first"]
        R[f"CoUnavail{nm}UnrelatedPct"] = 100 * c["unrelated"]["p_second_given_first"]; R[f"CoUnavail{nm}CellRatePct"] = 100 * c["cell_rate"]
    ln = desc["limited_notes"]; R["LimitedN"] = ln["n"]; R["LimitedAllocN"] = ln["allocation"]; R["LimitedDateN"] = ln["date"]; R["LimitedSupplyN"] = ln["limited_supply"]; R["LimitedBlankN"] = ln["blank"]
    R["StaleNotesN"] = desc["stale_notes"]["n"]
    mn = desc["marketer_notes"]; R["MarketerNoteN"] = mn["n"]; R["MarketerNoteReporters"] = mn["n_reporting_companies"]; R["MarketerNoteLabelers"] = mn["n_labelers"]
    R["MarketerNoteTopLabeler"] = mn["top_labeler"]; R["MarketerNoteTopN"] = mn["top_n"]; R["MarketerNoteTopReporters"] = ", ".join(mn["top_reporters"])
    R["MarketerNoteHiddenLabelers"] = mn["n_labelers_with_no_unavailable_own_cell"]; R["MarketerNoteListedLabelers"] = mn["n_labelers_listed_as_reporter"]
    R["MarketerNoteTopOwnN"] = mn["top_labeler_own_unavailable"]; R["MarketerNoteTopUnitCells"] = mn["top_labeler_unit_cells"]; R["MarketerNoteTopUnitUnavail"] = mn["top_labeler_unit_unavailable_cells"]
    fa = desc["ingredient_families"]; R["FamMultiN"] = fa["families_with_multiple_names"]; R["FamNamesN"] = fa["names_in_such_families"]
    R["SameFirmPairsN"] = fa["same_firm_pairs"]; R["SameFirmPairsWithinFamN"] = fa["same_firm_pairs_within_family"]; R["SameFirmPairsWithinFamFirms"] = fa["firms_with_within_family_pairs"]
    rc = desc["reason_counts"]; R["ReasonShippingN"] = rc.get("Delay in shipping of the drug", 0); R["ReasonInactiveN"] = rc.get("Shortage of an inactive ingredient component", 0); R["ReasonRegulatoryN"] = rc.get("Regulatory delay", 0)
    R["PerManufacturerNoteN"] = desc["per_manufacturer_notes"]["n"]; R["PerManufacturerNoteOtherN"] = desc["per_manufacturer_notes"]["n_other_or_blank_reason"]
    R["WholesalerAvailableN"] = desc["available_related_info"]["wholesalers"]; R["WholesalerUnavailableN"] = desc["per_manufacturer_notes"]["wholesaler_unavailable"]
    nd = desc["ndc_duplicates"]; R["NdcDupAcrossN"] = nd["across_companies_ndcs"]; R["NdcDupAcrossPres"] = nd["across_companies_presentations"]; R["NdcDupWithinN"] = nd["within_company_ndcs"]; R["NdcDupWithinPres"] = nd["within_company_presentations"]
    dc = desc["discontinued"]
    R["DiscByReason"] = dc["by_reason"]; R["DiscByNoteOnly"] = dc["by_note_only"]; R["TBDDrugs"] = dc["tbd_drugs"]
    R["DiscPres"] = dc["presentations"]; R["DiscCellsOnly"] = dc["cells_only_via_discontinued"]; R["DiscCellsAll"] = dc["cells_all_discontinued"]; R["NCellsUnavailNoDisc"] = dc["cells_unavailable_nodisc"]
    R["FirmsNoRecall"] = desc["firms_no_recall_event_5y"]
    R["CarboplatinGMPFirms"] = ", ".join(pretty_firm(f) for f in desc["carboplatin_gmp_firms"]) or "none"; R["CarboplatinGMPFirmsN"] = len(desc["carboplatin_gmp_firms"]); R["CarboplatinDemandFirms"] = desc["carboplatin_demand_firms"]
    R["CarboplatinFirmsListed"] = desc["carboplatin"]["n_firms"]; R["CarboplatinPostingDates"] = desc["carboplatin"]["n_posting_dates"]
    R["CarboplatinPostingDatesWord"] = "the same date" if desc["carboplatin"]["n_posting_dates"] == 1 else f"{desc['carboplatin']['n_posting_dates']} different dates"
    # simulation-based calibration of the sampler (analysis/sbc.py): main run (thinned), short run, Geweke kernel test
    sbc = load("sbc.json")
    QN = (("eps", "Eps"), ("a", "A"), ("b", "B"), ("phi", "Phi"), ("mu", "Mu"), ("af_firm", "AFF"), ("af_drug", "AFM"), ("phi_a", "PhiA"), ("mu_b", "MuB"))
    for run, pre in ((sbc, "SBC"), (sbc["short"], "SBCShort")):
        R[f"{pre}N"] = run["n_sbc"]; R[f"{pre}L"] = run["L"]; R[f"{pre}NIter"] = f"{run['n_iter']:,}"; R[f"{pre}Burn"] = run["burn"]; R[f"{pre}Thin"] = run["thin"]
        R[f"{pre}NBins"] = run["n_bins"]; R[f"{pre}Degenerate"] = run["n_degenerate_datasets"]; R[f"{pre}ExpectedBin"] = run["expected_per_bin"]
        R[f"{pre}ExpectedTop"] = run["expected_top_bin"]; R[f"{pre}ExpectedBottom"] = run["expected_bottom_bin"]
        R[f"{pre}MinP"] = run["min_p"]; R[f"{pre}MaxP"] = run["max_p"]; R[f"{pre}NBelowFive"] = run["n_p_below_005"]
        other = [k for k in run["tests"] if k not in ("af_drug", "mu_b")]
        R[f"{pre}MinPOther"] = min(run["tests"][k]["p"] for k in other)
        R[f"{pre}CovNinetyMinPct"] = 100 * min(run["coverage90"].values()); R[f"{pre}CovNinetyMaxPct"] = 100 * max(run["coverage90"].values())
        for k, nm in QN:
            t = run["tests"][k]
            R[f"{pre}{nm}P"] = t["p"]; R[f"{pre}{nm}MeanRank"] = t["mean_rank"]; R[f"{pre}{nm}CovNinetyPct"] = 100 * run["coverage90"][k]
            R[f"{pre}{nm}Top"] = run["top_bin"][k]; R[f"{pre}{nm}Bottom"] = run["bottom_bin"][k]
            R[f"{pre}{nm}Bins"] = " & ".join(str(c) for c in t["counts"])       # decile counts for the rank-histogram table
            R[f"{pre}{nm}BinsList"] = ", ".join(str(c) for c in t["counts"])    # the same counts for the text
        dg = run["diagnostics"]
        R[f"{pre}ESSMuBMed"] = dg["ess_mu_b"]["median"]; R[f"{pre}ESSMuBMin"] = dg["ess_mu_b"]["min"]; R[f"{pre}ESSAFMMed"] = dg["ess_af_drug"]["median"]; R[f"{pre}ESSAFMMin"] = dg["ess_af_drug"]["min"]
        R[f"{pre}ESSEpsMed"] = dg["ess_eps"]["median"]; R[f"{pre}RhatMuBMax"] = dg["rhat_mu_b"]["max"]; R[f"{pre}LagOneMuBMed"] = dg["lag1_mu_b"]["median"]; R[f"{pre}LagOneMuBMax"] = dg["lag1_mu_b"]["max"]
        R[f"{pre}NRhatAbove"] = dg["n_rhat_above_105"]; R[f"{pre}NESSBelowHundred"] = dg["n_ess_mu_b_below_100"]
        fl = run["flagged"]
        R[f"{pre}FlaggedN"] = fl["n"]
        if fl["n"]:
            R[f"{pre}FlaggedMuMean"] = fl["mu_mean"]; R[f"{pre}OtherMuMean"] = fl["other_mu_mean"]; R[f"{pre}FlaggedEpsBottomPct"] = 100 * fl["share_eps_rank_bottom"]
            R[f"{pre}FlaggedMuAboveNinetyPct"] = 100 * fl["share_mu_above_0_9"]; R[f"{pre}OtherMuAboveNinetyPct"] = 100 * fl["other_share_mu_above_0_9"]; R[f"{pre}FlaggedBMean"] = fl["b_mean"]; R[f"{pre}OtherBMean"] = fl["other_b_mean"]
    gw = sbc["geweke"]
    R["GewekeNIter"] = gw["n_iter"]; R["GewekeThin"] = gw["thin"]; R["GewekeNKept"] = gw["n_kept"]; R["GewekeNPrior"] = gw["n_prior"]
    R["GewekeMaxDOverCrit"] = gw["max_D_over_crit"]; R["GewekeNFail"] = gw["n_fail"]; R["GewekeMinESS"] = gw["min_ess"]
    for k, nm in QN:
        g = gw["results"][k]; R[f"Geweke{nm}Mean"] = g["mean"]; R[f"Geweke{nm}D"] = g["ks_D"]; R[f"Geweke{nm}Crit"] = g["ks_crit_005"]; R[f"Geweke{nm}ESS"] = g["ess"]
        R[f"Geweke{nm}AboveNinetyPct"] = 100 * g["share_above_prior_q90"]; R[f"Geweke{nm}BelowTenPct"] = 100 * g["share_below_prior_q10"]
    R["GewekeMeanMin"] = min(v["mean"] for v in gw["results"].values() if v["prior_mean"] > 0.49); R["GewekeMeanMax"] = max(v["mean"] for v in gw["results"].values() if v["prior_mean"] > 0.49)
    # binomial tails of the top-tenth counts (the quantity that is anomalous in the first batch), and the later batches alone
    from scipy.stats import binom
    ptop = sbc["expected_top_bin"] / sbc["n_sbc"]
    R["SBCMuBTopTail"] = binom.sf(sbc["top_bin"]["mu_b"] - 1, sbc["n_sbc"], ptop); R["SBCAFMTopTail"] = binom.sf(sbc["top_bin"]["af_drug"] - 1, sbc["n_sbc"], ptop)
    R["SBCMuBTopZ"] = (sbc["top_bin"]["mu_b"] - sbc["n_sbc"] * ptop) / (sbc["n_sbc"] * ptop * (1 - ptop)) ** 0.5
    # the extra batches of the main run and the pooled tests
    po = sbc["pooled"]
    R["SBCNBatches"] = po["n_batches"]; R["SBCPooledN"] = po["n"]; R["SBCPooledExpectedTop"] = po["expected_top_bin"]; R["SBCPooledExpectedBin"] = po["expected_per_bin"]
    for k, nm in QN:
        R[f"SBCPooled{nm}P"] = po["tests"][k]["p"]; R[f"SBCPooled{nm}Top"] = po["top_bin"][k]; R[f"SBCPooled{nm}Bottom"] = po["bottom_bin"][k]
        R[f"SBCPooled{nm}CovNinetyPct"] = 100 * po["coverage90"][k]
    R["SBCPooledMinPOther"] = po["min_p_other"]; R["SBCPooledMinP"] = po["min_p"]
    R["SBCBatchTopsMuB"] = ", ".join(str(x) for x in po["batch_top_mu_b"]); R["SBCBatchTopsAFM"] = ", ".join(str(x) for x in po["batch_top_af_drug"])
    fmt_p = lambda v: (f"{v:.2f}" if v >= 0.0095 else (f"{v:.3f}" if v >= 0.001 else "<0.001"))
    R["SBCBatchPsMuB"] = ", ".join(fmt_p(v) for v in po["batch_p_mu_b"]); R["SBCBatchPsAFM"] = ", ".join(fmt_p(v) for v in po["batch_p_af_drug"])
    R["SBCBatchMinPOther"] = min(po["batch_min_p_other"]); R["SBCBatchTwoMuBTop"] = po["batch_top_mu_b"][1] if len(po["batch_top_mu_b"]) > 1 else ""
    R["SBCBatchTwoAFMTop"] = po["batch_top_af_drug"][1] if len(po["batch_top_af_drug"]) > 1 else ""
    R["SBCBatchesAfterFirstMuBTop"] = sum(po["batch_top_mu_b"][1:]); R["SBCBatchesAfterFirstAFMTop"] = sum(po["batch_top_af_drug"][1:])
    R["SBCBatchesAfterFirstN"] = po["n"] - sbc["n_sbc"]; R["SBCBatchesAfterFirstExpectedTop"] = po["expected_top_bin"] * (po["n"] - sbc["n_sbc"]) / po["n"]
    nA = po["n"] - sbc["n_sbc"]
    R["SBCBatchesAfterFirstMuBTopLowerTail"] = binom.cdf(R["SBCBatchesAfterFirstMuBTop"], nA, ptop); R["SBCBatchesAfterFirstAFMTopLowerTail"] = binom.cdf(R["SBCBatchesAfterFirstAFMTop"], nA, ptop)
    R["SBCBatchesAfterFirstMuBTopUpperTail"] = binom.sf(R["SBCBatchesAfterFirstMuBTop"] - 1, nA, ptop)
    lat = {k: sum((b["ranks"][k] for b in sbc["batches"][1:]), []) for k in ("mu_b", "af_drug")}
    for k, nm in (("mu_b", "MuB"), ("af_drug", "AFM")):
        t = uniformity_test_counts(lat[k], sbc["L"]); R[f"SBCBatchesAfterFirst{nm}P"] = t
    R["SBCBatchMinPOtherQuantity"] = min(((min((b["tests"][k]["p"], k) for k in b["tests"] if k not in ("af_drug", "mu_b"))) for b in sbc["batches"]))[1]
    R["SBCBatchMinPOtherQuantity"] = {"eps": r"$\varepsilon$", "a": "$a$", "b": "$b$", "phi": r"$\phi$", "mu": r"$\mu$", "af_firm": r"$\mathrm{AF}_{\mathrm{firm}}$", "phi_a": r"$\phi a$"}[R["SBCBatchMinPOtherQuantity"]]
    R["SBCNOtherBatchTests"] = 7 * len(sbc["batches"])
    # kernel test restricted to the region of the prior where the flagged replicates lie
    rg = gw["region"]
    R["GewekeRegionMuMin"] = rg["mu_min"]; R["GewekeRegionEpsMax"] = rg["eps_max"]; R["GewekeRegionN"] = rg["n"]
    R["GewekeRegionSharePct"] = 100 * rg["share"]; R["GewekeRegionPriorSharePct"] = 100 * rg["prior_share"]
    R["GewekeRegionEpsMean"] = rg["eps_mean"]; R["GewekeRegionPriorEpsMean"] = rg["prior_eps_mean"]
    R["GewekeRegionMuBMean"] = rg["mu_b_mean"]; R["GewekeRegionPriorMuBMean"] = rg["prior_mu_b_mean"]
    R["GewekeRegionBMean"] = rg["b_mean"]; R["GewekeRegionPriorBMean"] = rg["prior_b_mean"]
    # drug-only submodel: sampler against the exact grid posterior
    rc = sbc["restricted"]
    R["SBCRestrN"] = rc["n"]; R["SBCRestrGrid"] = rc["grid"]; R["SBCRestrExpectedTop"] = rc["expected_top_bin"]
    for key, nm in (("mu_b_sampler", "SamplerMuB"), ("mu_b_exact", "ExactMuB"), ("eps_sampler", "SamplerEps"), ("eps_exact", "ExactEps")):
        R[f"SBCRestr{nm}P"] = rc["tests"][key]["p"]; R[f"SBCRestr{nm}Top"] = rc["top_bin"][key]
    R["SBCRestrMeanAbsDiffMuB"] = rc["mean_abs_diff_mu_b"]; R["SBCRestrMaxAbsDiffMuB"] = rc["max_abs_diff_mu_b"]
    R["SBCRestrMeanAbsDiffEps"] = rc["mean_abs_diff_eps"]; R["SBCRestrMaxAbsDiffEps"] = rc["max_abs_diff_eps"]
    R["SBCRestrCorrMuB"] = f"{rc['corr_mean_mu_b']:.3f}"; R["SBCRestrRankCorrMuB"] = f"{rc['rank_corr_mu_b']:.3f}"
    pr = rc["expected_top_bin"] / rc["n"]
    R["SBCRestrSamplerMuBTopTail"] = binom.sf(rc["top_bin"]["mu_b_sampler"] - 1, rc["n"], pr); R["SBCRestrExactMuBTopTail"] = binom.sf(rc["top_bin"]["mu_b_exact"] - 1, rc["n"], pr)
    # start-dependence check: flagged replicates (true mu*b in the top tenth of the cold chain) and controls refitted from the truth
    rs = sbc["restart"]
    R["SBCRestartNFlagged"] = rs["n_flagged"]; R["SBCRestartNControl"] = rs["n_control"]; R["SBCRestartRhatFlag"] = rs["rhat_flag"]
    for grp, G in (("flagged", "Flagged"), ("control", "Control")):
        g = rs[grp]
        R[f"SBCRestart{G}N"] = g["n"]; R[f"SBCRestart{G}ColdTop"] = g["n_cold_top"]; R[f"SBCRestart{G}TruthTop"] = g["n_truth_start_top"]
        R[f"SBCRestart{G}AFMTruthTop"] = g["n_truth_start_af_drug_top"]
        R[f"SBCRestart{G}RhatAbove"] = g["n_rhat_above"]; R[f"SBCRestart{G}RhatMedian"] = g["rhat_median"]; R[f"SBCRestart{G}RhatMax"] = g["rhat_max"]
        R[f"SBCRestart{G}RhatEpsAbove"] = g["n_rhat_eps_above"]; R[f"SBCRestart{G}RhatEpsMedian"] = g["rhat_eps_median"]
        R[f"SBCRestart{G}MedRankCold"] = g["median_rank_cold"]; R[f"SBCRestart{G}MedRankTruth"] = g["median_rank_truth_start"]
        R[f"SBCRestart{G}TrueMuBMean"] = g["mean_true_mu_b"]; R[f"SBCRestart{G}ColdMuBMean"] = g["mean_post_mu_b_cold"]; R[f"SBCRestart{G}TruthMuBMean"] = g["mean_post_mu_b_truth_start"]
        R[f"SBCRestart{G}TrueEpsMean"] = g["mean_true_eps"]; R[f"SBCRestart{G}ColdEpsMean"] = g["mean_post_eps_cold"]; R[f"SBCRestart{G}TruthEpsMean"] = g["mean_post_eps_truth_start"]
    # observed data: a chain started with every drug shocked against an ordinary start, same length as the SBC main run
    oh = sbc["observed_high_start"]
    R["SBCObsNIter"] = oh["n_iter"]; R["SBCObsInitMu"] = oh["start"]["mu"]; R["SBCObsInitB"] = oh["start"]["b"]; R["SBCObsInitEps"] = oh["start"]["eps"]
    for st, S in (("high", "High"), ("cold", "Cold")):
        for q, Q in (("mu_b", "MuB"), ("af_drug", "AFM"), ("eps", "Eps")):
            R[f"SBCObs{S}{Q}Med"] = oh[st][q]["median"]; R[f"SBCObs{S}{Q}Lo"] = oh[st][q]["lo"]; R[f"SBCObs{S}{Q}Hi"] = oh[st][q]["hi"]
    R["SBCObsRhatMax"] = oh["rhat_between_starts_max"]; R["SBCObsRhatMuB"] = oh["rhat_between_starts"]["mu_b"]
    # effective sample sizes and split-R-hat of the derived quantities in the main fit (theory_analysis.py)
    ed = thr["ess_derived"]
    for k, nm in (("mu_b", "MuB"), ("af_drug", "AFM"), ("af_firm", "AFF"), ("af_diff", "AFDiff"), ("b", "B"), ("mu", "Mu")):
        R[f"ESS{nm}Pooled"] = ed[k]["ess_sum"]; R[f"ESS{nm}ChainMin"] = ed[k]["ess_chain_min"]; R[f"ESS{nm}ChainMax"] = ed[k]["ess_chain_max"]; R[f"Rhat{nm}"] = ed[k]["rhat"]
    R["ESSDerivedChains"] = ed["n_chains"]; R["ESSDerivedPerChain"] = ed["draws_per_chain"]
    # round-3 methods referee: pooled targeted tails, the third refit of the first batch, integrated autocorrelation times,
    # the drug-side baseline of the moment estimator, the calibration nulls away from the non-identified corner, the
    # per-replicate SBC diagnostics and the independent collapsed-W sampler (analysis/sbc_collapsed.py)
    po_ = sbc["pooled"]; n_pooled = sum(len(b["ranks"]["mu_b"]) for b in sbc["batches"])
    # every quantity's pooled top-tenth count, its one-sided tail, and the batch counts for b (methods referee, v4 round 2)
    _bt = float(binom.sf(po_['tests']['b']['counts'][-1] - 1, n_pooled, ptop)); R["SBCPooledBTopTail"] = f"{_bt:.3f}"; R["SBCPooledBTopTailBonf"] = f"{min(1.0, 9 * _bt):.2f}"
    R["SBCBatchTopsB"] = ", ".join(str(b["top_bin"]["b"]) for b in sbc["batches"])
    R["SBCPooledBMeanRankPct"] = 100 * float(np.mean(np.concatenate([np.asarray(b["ranks"]["b"]) for b in sbc["batches"]]))) / sbc["L"]
    R["SBCPooledMuBTopTail"] = f"{binom.sf(po_['tests']['mu_b']['counts'][-1] - 1, n_pooled, ptop):.3f}"; R["SBCPooledAFMTopTail"] = f"{binom.sf(po_['tests']['af_drug']['counts'][-1] - 1, n_pooled, ptop):.3f}"
    b0 = sbc["batches"][0]
    ldp = os.path.join(RES, "sbc_long_diag.json")
    if os.path.exists(ldp):
        ld = json.load(open(ldp))
        R["SBCLongNIter"] = ld["n_iter"]; R["SBCLongThin"] = ld["thin"]; R["SBCLongMuBTop"] = ld["tests"]["mu_b"]["counts"][-1]; R["SBCLongAFMTop"] = ld["tests"]["af_drug"]["counts"][-1]
        R["SBCLongMuBP"] = ld["tests"]["mu_b"]["p"]; R["SBCLongRankCorr"] = f"{np.corrcoef(ld['ranks']['mu_b'], b0['ranks']['mu_b'])[0, 1]:.3f}"
    n_post = fit["n_chains"] * (fit["n_iter"] - fit["burn"])
    for k, nm in (("b", "B"), ("mu", "Mu"), ("mu_b", "MuB")):
        R[f"IAT{nm}"] = int(round(n_post / ed[k]["ess_sum"]))
    mom = thr["mom"]["main"]; q0 = mom["drug"]["q0"]; phia = mom["firm"]["phi"] * mom["params"]["a"]
    R["MoMEpsDrug"] = 1 - (1 - q0) / (1 - phia); R["MoMMOneFirm"] = mom["firm_moments"][0]; R["MoMMOneDrug"] = mom["drug_moments"][0]
    nulls = fit["calibration"]["nulls"]; low = [x for x in nulls if x["truth"]["mu"] <= 0.75]
    R["CalibMuLeSeventyFiveN"] = len(low); R["CalibMuLeSeventyFiveMaxPct"] = 100 * max(x["share_ge_obs"] for x in low); R["CalibMuLeSeventyFiveZeroN"] = int(sum(x["share_ge_obs"] == 0 for x in low))
    hi_ = [x for x in nulls if x["truth"]["mu"] >= 0.95 and x["eps_label"] == "median"]; R["CalibHighMuMedEpsMaxPct"] = 100 * max(x["share_ge_obs"] for x in hi_)
    if "n_rep" in fit["ppc"]:
        R["PPCNRep"] = fit["ppc"]["n_rep"]
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from analysis.fit_model import N_PPC; R["PPCNRep"] = N_PPC
    flagged_idx = {x["r"] for x in sbc["restart"]["replicates"] if x["flagged"]}
    if "per_replicate" in b0:
        e_ = np.asarray(b0["per_replicate"]["ess_mu_b"]); l1_ = np.asarray(b0["per_replicate"]["lag1_mu_b"])
        R["SBCESSMuBMin"] = int(round(e_.min())); R["SBCESSBelowHundredN"] = int((e_ < 100).sum()); R["SBCFlaggedESSBelowHundredN"] = int(sum(e_[r] < 100 for r in flagged_idx)); R["SBCLagOneMuBMax"] = f"{l1_.max():.2f}"
    else:   # before the per-replicate values were saved: batch-level summaries only
        R["SBCESSMuBMin"] = int(round(b0["diagnostics"]["ess_mu_b"]["min"])); R["SBCESSBelowHundredN"] = b0["diagnostics"]["n_ess_mu_b_below_100"]; R["SBCFlaggedESSBelowHundredN"] = "TBD"; R["SBCLagOneMuBMax"] = f"{b0['diagnostics']['lag1_mu_b']['max']:.2f}"
    cjp = os.path.join(RES, "sbc_collapsed.json")
    if os.path.exists(cjp):
        cj = json.load(open(cjp))
        R["CollNIter"] = cj["n_iter"]; R["CollThin"] = cj["thin"]; R["CollL"] = cj["L"]; R["CollMinutes"] = int(round(cj["minutes"])); R["CollObsNIter"] = cj["n_iter_obs"]; R["CollObsThin"] = cj["thin_obs"]
        for grp, G in (("flagged", "Flagged"), ("control", "Control")):
            g = cj[grp]; R[f"Coll{G}N"] = g["n"]; R[f"Coll{G}TopPrev"] = g["top_tenth_prev"]; R[f"Coll{G}Top"] = g["top_tenth_collapsed"]; R[f"Coll{G}TopAFM"] = g["top_tenth_af_drug_collapsed"]
            R[f"Coll{G}MeanAbsDiffMuB"] = g["mean_abs_diff_mu_b"]; R[f"Coll{G}MaxAbsDiffMuB"] = g["max_abs_diff_mu_b"]; R[f"Coll{G}MeanDiffMuB"] = g["mean_diff_mu_b"]
            R[f"Coll{G}Corr"] = f"{g['corr_post_mean_mu_b']:.3f}"; R[f"Coll{G}RankCorr"] = f"{g['rank_corr_mu_b']:.3f}"; R[f"Coll{G}Cover"] = g["cover90_mu_b"]
            R[f"Coll{G}ESSMed"] = int(round(g["ess_mu_b_median"])); R[f"Coll{G}ESSMin"] = int(round(g["ess_mu_b_min"])); R[f"Coll{G}Accept"] = g["accept_median"]
        ob = cj["observed"]
        for q, Q in (("mu_b", "MuB"), ("af_drug", "AFM"), ("eps", "Eps"), ("b", "B"), ("mu", "Mu")):
            R[f"CollObs{Q}Med"] = ob[q]["median"]; R[f"CollObs{Q}Lo"] = ob[q]["lo95"]; R[f"CollObs{Q}Hi"] = ob[q]["hi95"]
        R["CollObsRhatMax"] = f"{max(ob['rhat'].values()):.3f}"; R["CollObsESSMuB"] = int(round(ob["ess_pooled"]["mu_b"])); R["CollObsESSB"] = int(round(ob["ess_pooled"]["b"])); R["CollObsAccept"] = ob["accept"]
    # calibration of the ordering: the least favourable point's share on the fresh replicates only (after the selection)
    fresh = wn["p_values"][cal["n_rep"]:]
    R["CalibWorstFreshN"] = len(fresh); R["CalibWorstFreshK"] = int(sum(p >= cal["p_obs"] for p in fresh)); R["CalibWorstFreshPct"] = 100 * R["CalibWorstFreshK"] / max(len(fresh), 1)
    R["CalibWorstGridK"] = int(sum(p >= cal["p_obs"] for p in wn["p_values"][:cal["n_rep"]]))
    # paper-v5 round-4 additions to panel.py (separate random stream)
    ee = pn["entries_exits_no_iv_fluids"]; R["PanChgEENoIVFirmP"] = ee["p_firm"]; R["PanChgEENoIVDrugP"] = ee["p_drug"]; R["PanChgEENoIVDrugs"] = ee["n_drugs_dropped"]
    uo = pn["unavailable_only"]; R["PanUOTransP"] = uo["transition_test"]["p_cross"]; R["PanUOTransSizeP"] = uo["transition_test_size"]["p_cross"]
    R["PanUOChgFirmP"] = uo["change"]["p_firm"]; R["PanUORatioAB"] = uo["ratios"]["w1_w2"]; R["PanUORatioBC"] = uo["ratios"]["w2_w3"]
    R["PanTransNoTopSizeP"] = pn["no_top3_size"]["p_cross"]
    # without the firms listed for the most drugs, cross-section (outcome_variants.py)
    for k in (3, 5):
        v = ov[f"no_top{k}"]; nm = {3: "Three", 5: "Five"}[k]
        R[f"OVNoTop{nm}FirmRatio"] = v["disp_firm_ratio"]; R[f"OVNoTop{nm}FirmSizeP"] = v["perm_firm_p_size"]; R[f"OVNoTop{nm}CovFirm"] = v["cov_firm"]
        R[f"OVNoTop{nm}CovFirmSizeP"] = v["perm_cov_firm_p_size"]; R[f"OVNoTop{nm}CovFirmP"] = v["perm_cov_firm_p"]; R[f"OVNoTop{nm}Cells"] = v["n_cells"]
    R["OVNoTopFiveFirms"] = and_join(pretty_firm(f) for f in ov["no_top5"]["firms_dropped"])
    # the panel 2019-2026 from the archived FDA CSV captures (long_panel.py)
    lp = load("long_panel.json")
    comp = lp["composition"]; grid = lp["grid"]; rec = lp["recurrence"]
    fmt_date = lambda ts: dt.date(int(ts[:4]), int(ts[4:6]), int(ts[6:8])).strftime("%-d %B %Y")
    fmt_month = lambda ts: dt.date(int(ts[:4]), int(ts[4:6]), 1).strftime("%B %Y")
    md = lambda v: int(np.median(v)) if float(np.median(v)).is_integer() else float(np.median(v))
    R["LPCaptures"] = lp["info"]["n_captures"]; R["LPCapFirstMonth"] = fmt_month(comp[0]["timestamp"]); R["LPCapLastMonth"] = fmt_month(comp[-1]["timestamp"])
    R["LPCapFirst"] = fmt_date(comp[0]["timestamp"]); R["LPCapLast"] = fmt_date(comp[-1]["timestamp"])
    R["LPGridN"] = len(grid); R["LPGridFirst"] = fmt_month(grid[0]); R["LPGridLast"] = fmt_month(grid[-1])
    days = lp["intervals"]["days"]; R["LPIntervals"] = lp["intervals"]["n"]; R["LPIntervalMedianDays"] = md(days)
    R["LPIntervalMinDays"] = int(min(days)); R["LPIntervalMaxDays"] = int(max(days))
    R["LPIntervalsLong"] = sum(d > 200 for d in days)
    R["LPMigrationN"] = len(lp["intervals"]["migration"]); R["LPMigrationShare"] = int(100 * 0.15)
    R["LPDrugNameMerges"] = lp["info"]["n_names_mapped"]
    co = lp["coding"]; R["LPCategoricalPct"] = 100 * co["categorical"] / co["presentations"]
    R["LPUninformative"] = co["uninformative"]; R["LPUninformativePct"] = 100 * co["uninformative"] / co["presentations"]
    R["LPDistinctStrings"] = co["distinct_strings"]; R["LPHarmonisedRecoded"] = co["harmonised_recoded"]; R["LPStale"] = co["stale"]
    R["LPStalePct"] = 100 * co["stale_share_of_free_text_unavailable"]
    R["LPPreCatUnavailPct"] = 100 * co["pre2024_categorical_unavailable"]; R["LPPreFreeUnavailPct"] = 100 * co["pre2024_free_text_unavailable"]
    cat_start = next(c for c in comp if c["categorical_share"] > 0.8); R["LPCatEraStart"] = fmt_month(cat_start["timestamp"])
    R["LPCatSharePrePct"] = 100 * float(np.median([c["categorical_share"] for c in comp if c["timestamp"] < cat_start["timestamp"]]))
    R["LPCatSharePostPct"] = 100 * float(np.median([c["categorical_share"] for c in comp if c["timestamp"] >= cat_start["timestamp"]]))
    ma = lp["migration_agreement"]; R["LPMigAgreePct"] = 100 * ma["agree_pooled"]; R["LPMigAgreeN"] = ma["n_total"]
    R["LPMigAgreeMinPct"] = 100 * ma["agree_min"]; R["LPMigAgreeMaxPct"] = 100 * ma["agree_max"]; R["LPMigAgreeHarmPct"] = 100 * ma["agree_harmonised_pooled"]
    for c_ in ("drugs", "firms", "cells"):
        vals = [c[c_] for c in comp]; R[f"LP{c_.title()}Min"] = int(min(vals)); R[f"LP{c_.title()}Max"] = int(max(vals))
    R["LPCellRateMinPct"] = 100 * min(c["cell_rate"] for c in comp); R["LPCellRateMaxPct"] = 100 * max(c["cell_rate"] for c in comp)
    early = [c["drugs_first_posted_180d"] for c in comp if c["date"] < "2025-01-01"]; late = [c["drugs_first_posted_180d"] for c in comp if c["date"] >= "2025-01-01"]
    R["LPNewDrugsEarly"] = md(early); R["LPNewDrugsLate"] = md(late); R["LPNewDrugsLateMin"] = int(min(late)); R["LPNewDrugsLateMax"] = int(max(late))
    for W_, w in (("A", "w1"), ("B", "w2"), ("C", "w3")):
        v = lp["validation"][w]; R[f"LPVal{W_}Days"] = abs(v["days_apart"])
    R["LPValCommonMinPct"] = f"{100 * min(lp['validation'][w]['share_common_of_openfda'] for w in ('w1', 'w2', 'w3')):.1f}"
    R["LPValAgreeMinPct"] = f"{100 * min(lp['validation'][w]['status_agreement'] for w in ('w1', 'w2', 'w3')):.1f}"
    R["LPValDaysMax"] = max(R[f"LPVal{W_}Days"] for W_ in "ABC")
    rr = list(rec.values())
    R["LPFirmAboveNullN"] = sum(r["firm_ratio"] > r["firm_null_ratio"] for r in rr); R["LPDrugAboveNullN"] = sum(r["drug_ratio"] > r["drug_null_ratio"] for r in rr)
    R["LPFirmSigN"] = sum(r["firm_p_size"] < 0.05 for r in rr); R["LPDrugSigN"] = sum(r["drug_p_size"] < 0.05 for r in rr)
    R["LPFirmRatioMed"] = float(np.median([r["firm_ratio"] for r in rr])); R["LPFirmNullMed"] = float(np.median([r["firm_null_ratio"] for r in rr]))
    R["LPDrugRatioMed"] = float(np.median([r["drug_ratio"] for r in rr]))
    R["LPOverlapFirstLastPct"] = 100 * lp["overlap_first_last"]["share_of_last"]; R["LPOverlapDrugsFirstLastPct"] = 100 * lp["overlap_first_last"]["drugs_share_of_last"]
    names = dict(all="All", with_migration="WithMig", no_top3="NoTop", pre_2022="Early", **{"2022_2024": "Mid"}, post_migration="Post", labeler="Lab",
                 marketer="Mkt", group="Grp", harmonised="Harm", no_stale="NoStale", no_discontinued="ND")
    for key, nm in names.items():
        if key not in lp["tests"]: continue
        v = lp["tests"][key]
        R[f"LP{nm}TransP"] = v["transition"]["p"]; R[f"LP{nm}TransSizeP"] = v["transition_size"]["p"]
        R[f"LP{nm}StayP"] = v["stay"]["p"]; R[f"LP{nm}BecomeP"] = v["become"]["p"]
        R[f"LP{nm}ChgStartP"] = v["change"]["p_within_start"]; R[f"LP{nm}ChgDrugOnlyP"] = v["change"]["p_within_drug"]; R[f"LP{nm}ChgDrugP"] = v["change"]["p_drug"]
        R[f"LP{nm}Changes"] = v["change"]["n_changes"]; R[f"LP{nm}Intervals"] = v["transition"]["n_intervals"]; R[f"LP{nm}CellIntervals"] = v["transition"]["n_cells"]
        R[f"LP{nm}Onsets"] = v["n_onsets"]; R[f"LP{nm}Recoveries"] = v["n_recoveries"]
        R[f"LP{nm}StayUnavailPct"] = 100 * v["transition"]["stay_unavailable"]; R[f"LP{nm}StayAvailPct"] = 100 * v["transition"]["stay_available"]
        R[f"LP{nm}TransCross"] = v["transition"]["cross_cov"]; R[f"LP{nm}TransNull"] = v["transition"]["cross_null_mean"]
    t3 = [pretty_firm(f) for f in lp["top3"]["firms"]]; R["LPTopThreeFirms"] = ", ".join(t3[:-1]) + " and " + t3[-1]; R["LPTopThreePairPct"] = 100 * lp["top3"]["pair_share"]
    pi = lp["per_interval"]; R["LPIntervalsStartSigN"] = sum(v["p_within_start"] < 0.05 for k, v in pi.items() if not v["migration"])
    R["LPIntervalsNonMig"] = sum(1 for v in pi.values() if not v["migration"])
    ns = lp["null_simulation"]; R["LPSimN"] = ns["n_sim"]; R["LPSimTransPct"] = 100 * ns["transition"]; R["LPSimChgDrugPct"] = 100 * ns["change_within_drug"]
    R["LPSimChgStartPct"] = 100 * ns["change_within_start"]
    FL = lp["first_listings"]
    for key, nm in (("main", "FL"), ("lag2y", "FLTwo"), ("no_top3", "FLNoTop"), ("firm_size", "FLSize"), ("firm_size_injectable", "FLSizeInj"),
                    ("unrelated", "FLUnrel"), ("pre_2024", "FLPre"), ("families", "FLFam"), ("no_discontinued", "FLND"), ("harmonised", "FLHarm"),
                    ("marketer", "FLMkt"), ("group", "FLGrp")):
        v = FL[key]
        R[f"LP{nm}N"] = v["n_new"]; R[f"LP{nm}K"] = v["n_unavailable"]; R[f"LP{nm}Firms"] = v["n_firms"]; R[f"LP{nm}GapDays"] = int(round(v["median_gap_days"]))
        R[f"LP{nm}OR"] = v["or_fs"]; R[f"LP{nm}ORLo"] = v["or_lo"]; R[f"LP{nm}ORHi"] = v["or_hi"]; R[f"LP{nm}PermP"] = v["perm_p"]
        R[f"LP{nm}PBootGtOnePct"] = 100 * v["p_boot_gt1"]
    R["LPFLPreN"] = FL["pre_2024"]["n_new"]; R["LPFLPostN"] = FL["main"]["n_new"] - FL["pre_2024"]["n_new"]
    b_ = FL["main"]["bins"]; R["LPFLBinLowPct"] = 100 * b_[0]["rate"]; R["LPFLBinLowN"] = b_[0]["n"]
    hi_n = sum(x["n"] for x in b_[1:]); R["LPFLBinHighPct"] = 100 * sum(x["rate"] * x["n"] for x in b_[1:]) / hi_n; R["LPFLBinHighN"] = hi_n
    R["LPFLBinMinPct"] = 100 * min(x["rate"] for x in b_[1:]); R["LPFLBinMaxPct"] = 100 * max(x["rate"] for x in b_[1:])
    R["LPNPerm"] = FL["main"]["n_perm"]; R["LPNBoot"] = FL["main"]["n_boot"]
    lag = lp["first_listing_lag"]; R["LPLagRatio"] = lag["ratio"]; R["LPLagRatioLo"] = lag["ratio_lo"]; R["LPLagRatioHi"] = lag["ratio_hi"]
    bl = lag["by_lag_years"]; R["LPLagZeroRatio"] = bl["0"]["ratio"]; R["LPLagZeroRatioLo"] = bl["0"]["ratio_lo"]; R["LPLagZeroRatioHi"] = bl["0"]["ratio_hi"]
    R["LPLagOneRatio"] = bl["1"]["ratio"]; R["LPLagOneRatioLo"] = bl["1"]["ratio_lo"]; R["LPLagOneRatioHi"] = bl["1"]["ratio_hi"]
    R["LPNewCellRatePct"] = 100 * lag["new_cell_rate"]
    us = lp["update_stamps"]; R["LPStampMeanPct"] = 100 * us["mean_share"]; R["LPStampLatePct"] = 100 * us["mean_share_2025_2026"]
    fe = lp["firm_exits"]; R["LPExitAvailPct"] = 100 * fe["rate_available"]; R["LPExitUnavailPct"] = 100 * fe["rate_unavailable"]
    ev = lp["events"]; F_ = ev["firms"]
    R["LPHeleneBefore"] = fmt_date(ev["helene_before"]); R["LPHeleneAfter"] = fmt_date(ev["helene_after"])
    for f, nm in (("BAXTER", "Baxter"), ("OTSUKA ICU", "ICU"), ("FRESENIUS KABI", "FK"), ("HOSPIRA", "Hospira"), ("B BRAUN", "Braun")):
        R[f"LPHel{nm}BeforeU"], R[f"LPHel{nm}BeforeN"] = F_[f]["before"]; R[f"LPHel{nm}AfterU"], R[f"LPHel{nm}AfterN"] = F_[f]["after"]
    R["LPHelAllBeforeU"], R["LPHelAllBeforeN"] = ev["all_firms"]["before"]; R["LPHelAllAfterU"], R["LPHelAllAfterN"] = ev["all_firms"]["after"]
    R["LPAkornLast"] = fmt_month(ev["akorn_last"]); R["LPAkornCells"] = ev["akorn_cells_last"]; R["LPAkornUnavail"] = ev["akorn_unavailable_last"]
    # per-interval table for the Supplement (generated LaTeX rows)
    def pfmt(v):
        return "<0.001" if v < 0.001 else (f"{v:.3f}" if v < 0.0095 else f"{v:.2f}")
    rows_tex = []
    for tr, v in lp["per_interval"].items():
        a_, b_ = tr.split("-")
        rows_tex.append(f"{a_[:4]}-{a_[4:6]}-{a_[6:]} & {b_[:4]}-{b_[4:6]}-{b_[6:]} & {v['days']} & {100 * v['switch_share']:.0f}{'$^*$' if v['migration'] else ''} & "
                        f"{v['n_changes']} & {pfmt(v['transition_p'])} & {pfmt(v['p_within_start'])} \\\\")
    with open(os.path.join(RES, "long_panel_intervals.tex"), "w") as fh:
        fh.write("\n".join(rows_tex) + "\n\\bottomrule\n")       # the closing rule inside the file: \input then \bottomrule fails
    # round-5 checks
    R["LPRepairedFirm"] = lp["info"]["n_repaired_firm"]; R["LPDupNDC"] = lp["info"]["n_duplicate_ndc_rows"]; R["LPFirmRenames"] = lp["info"]["n_firm_renames"]
    R["LPMigFreeUnav"] = ma["free_unavailable"]; R["LPMigFreeUnavToAvail"] = ma["free_unavailable_to_available"]
    R["LPMigFreeUnavToAvailPct"] = 100 * ma["free_unavailable_to_available"] / ma["free_unavailable"]
    bf = ma["by_firm"]; R["LPMigFirmMinPct"] = 100 * min(x["agree"] for x in bf); R["LPMigFirmMaxPct"] = 100 * max(x["agree"] for x in bf); R["LPMigFirmN"] = len(bf)
    top5 = [x for x in bf if x["firm_unit"] in ("HOSPIRA", "HIKMA", "FRESENIUS KABI")]
    R["LPMigTopMinPct"] = 100 * min(x["agree"] for x in top5); R["LPMigTopMaxPct"] = 100 * max(x["agree"] for x in top5)
    rows_tex = [f"{pretty_firm(x['firm_unit'])} & {int(x['n'])} & {100 * x['agree']:.0f} & {int(x['free_unav_to_avail'])} \\\\" for x in bf]
    with open(os.path.join(RES, "long_panel_migration_firms.tex"), "w") as fh:
        fh.write("\n".join(rows_tex) + "\n\\bottomrule\n")
    sc = lp["snapshot_checks"]
    for k, nm in (("no_long730", "LongTwo"), ("no_long365", "LongOne"), ("no_lvp", "NoLVP")):
        R[f"LPSnap{nm}FirmRatio"] = sc[k]["firm_ratio"]; R[f"LPSnap{nm}FirmSizeP"] = sc[k]["firm_p_size"]; R[f"LPSnap{nm}Cells"] = sc[k]["n_cells"]
    R["LPSnapFracP"] = sc["fractional"]["p_size"]; R["LPSnapFracNDP"] = sc["fractional_nodisc"]["p_size"]
    for k, nm in (("1", "One"), ("2", "Two"), ("3", "Three")):
        R[f"LPTopSize{nm}TopPct"] = 100 * sc["top3_by_size"][k]["top3"]; R[f"LPTopSize{nm}OtherPct"] = 100 * sc["top3_by_size"][k]["other"]
    R["LPLongFlagged"] = lp["long_unavailable"]["flagged_last"]; R["LPLongFlaggedOne"] = lp["long_unavailable"]["flagged_last_1y"]
    R["LPNoLVPSigN"] = sum(v["firm_p_size"] < 0.05 for v in lp["lvp_recurrence"].values()); R["LPNoLVPDrugs"] = len(lp["lvp_drugs"])
    ru = lp["recurrence_unavailable_only"]; R["LPUOSigN"] = sum(v["firm_p_size"] < 0.05 for v in ru.values()); R["LPUODates"] = len(ru)
    rr2 = sorted(rec.items()); post = [v for k, v in rr2 if k >= cat_start["timestamp"]]
    R["LPPostSigN"] = sum(v["firm_p_size"] < 0.05 for v in post); R["LPPostDates"] = len(post)
    recent = [v for k, v in rr2 if "20250701" <= k < grid[-1]]; R["LPRecentN"] = len(recent)
    R["LPRecentPMin"] = min(v["firm_p_size"] for v in recent); R["LPRecentPMax"] = max(v["firm_p_size"] for v in recent)
    R["LPRecentRatioMin"] = min(v["firm_ratio"] for v in recent); R["LPRecentRatioMax"] = max(v["firm_ratio"] for v in recent)
    R["LPLastFirmSizeP"] = rec[grid[-1]]["firm_p_size"]
    pk = [v for k, v in rr2 if "20231201" <= k <= "20250401"]; R["LPPeakRatioMin"] = min(v["firm_ratio"] for v in pk); R["LPPeakRatioMax"] = max(v["firm_ratio"] for v in pk)
    ob = lp["onset_by_share_post"]; R["LPOnsetPostN"] = ob["n"]; R["LPOnsetPostFirms"] = ob["n_firms"]; R["LPOnsetPostRatePct"] = 100 * ob["rate"]
    for i, nm in enumerate(("Q", "R", "S", "T")):
        b_ = ob["bins"][i]; R[f"LPOnsetBin{nm}Pct"] = 100 * b_["rate"]; R[f"LPOnsetBin{nm}LoPct"] = 100 * b_["lo95"]; R[f"LPOnsetBin{nm}HiPct"] = 100 * b_["hi95"]; R[f"LPOnsetBin{nm}N"] = b_["n"]
    pp = lp["propensity_persistence"]; R["LPPropOnsetRho"] = pp["onset"]["spearman"]; R["LPPropOnsetP"] = pp["onset"]["p"]; R["LPPropOnsetFirms"] = pp["onset"]["n_firms"]
    R["LPPropRecRho"] = pp["recovery"]["spearman"]; R["LPPropRecP"] = pp["recovery"]["p"]; R["LPPropRecFirms"] = pp["recovery"]["n_firms"]
    for key, nm in (("categorical_both", "CatBoth"), ("no_long_gap", "NoGap"), ("no_lvp", "NoLVP"), ("no_long730", "NoLong"), ("no_long365", "NoLongOne")):
        v = lp["tests"][key]
        R[f"LP{nm}TransP"] = v["transition"]["p"]; R[f"LP{nm}StayP"] = v["stay"]["p"]; R[f"LP{nm}BecomeP"] = v["become"]["p"]
        R[f"LP{nm}ChgDrugOnlyP"] = v["change"]["p_within_drug"]; R[f"LP{nm}ChgStartP"] = v["change"]["p_within_start"]
    for key, nm in (("categorical", "FLCat"), ("no_long_gap", "FLNoGap"), ("no_long730", "FLNoLong")):
        v = FL[key]; R[f"LP{nm}N"] = v["n_new"]; R[f"LP{nm}OR"] = v["or_fs"]; R[f"LP{nm}ORLo"] = v["or_lo"]; R[f"LP{nm}ORHi"] = v["or_hi"]; R[f"LP{nm}PermP"] = v["perm_p"]
    for key in FL:
        pass
    R["LPFLPermDateP"] = FL["main"]["perm_p_within_date"]
    flv = [(k, v) for k, v in FL.items()]
    R["LPFLORMin"] = min(v["or_fs"] for k, v in flv); R["LPFLORMax"] = max(v["or_fs"] for k, v in flv); R["LPFLVariants"] = len(flv)
    names_fl = dict(main="Main", lag2y="Record two years earlier", no_top3="Without the three largest firms", firm_size="Firm size held fixed",
                    firm_size_injectable="Firm size and injectable form held fixed", unrelated="Without related products", pre_2024="First listings before 2024",
                    families="Ingredient families as drugs", no_discontinued="Without discontinued presentations", harmonised="Allocation-harmonised coding",
                    marketer="Marketer of record as the firm", group="Corporate group as the firm", categorical="Categorically coded new cells only",
                    no_long_gap="After intervals of at most 200 days", no_long730="Long-unavailable SKUs not listed")
    def pf(v): return "<0.001" if v < 0.001 else (f"{v:.3f}" if v < 0.0095 else f"{v:.2f}")
    rows_tex = [f"{names_fl[k]} & {v['n_new']} & {v['or_fs']:.2f} & {v['or_lo']:.2f}--{v['or_hi']:.2f} & {pf(v['perm_p'])} \\\\" for k, v in flv]
    with open(os.path.join(RES, "long_panel_first_listings.tex"), "w") as fh:
        fh.write("\n".join(rows_tex) + "\n\\bottomrule\n")
    lo = load("leave_one_out.json") if os.path.exists(os.path.join(RES, "leave_one_out.json")) else None
    if lo:
        R["LOOFirms"] = and_join(pretty_firm(f) for f in lo["firms"])
        R["LOOAllFLOR"] = lo["all"]["fl_or"]; R["LOOAllFLORLo"] = lo["all"]["fl_lo"]; R["LOOAllFLORHi"] = lo["all"]["fl_hi"]
        R["LOOFirmRatioMin"] = min(v["firm_ratio"] for v in lo["drop"].values()); R["LOOFirmRatioMax"] = max(v["firm_ratio"] for v in lo["drop"].values())
        R["LOOCoMinPct"] = 100 * min(v["co_same_firm"] for v in lo["drop"].values()); R["LOOCoMaxPct"] = 100 * max(v["co_same_firm"] for v in lo["drop"].values())
        R["LOOFLORMin"] = min(v["fl_or"] for v in lo["drop"].values()); R["LOOFLORMax"] = max(v["fl_or"] for v in lo["drop"].values())
        for i, (f, v) in enumerate(lo["drop"].items()):
            nm = ["One", "Two", "Three", "Four", "Five"][i]
            R[f"LOO{nm}Firm"] = pretty_firm(f); R[f"LOO{nm}FirmRatio"] = v["firm_ratio"]; R[f"LOO{nm}CovFirm"] = v["cov_firm"]
            R[f"LOO{nm}CoPct"] = 100 * v["co_same_firm"]; R[f"LOO{nm}UnrelPct"] = 100 * v["co_unrelated"]
            R[f"LOO{nm}FLOR"] = v["fl_or"]; R[f"LOO{nm}FLORLo"] = v["fl_lo"]; R[f"LOO{nm}FLORHi"] = v["fl_hi"]
    # range of the size-stratified firm dispersion p over the five distinct outcome definitions
    pf = {k: ov[k]["perm_firm_p_size"] for k in keys5}
    R["OVFirmSizePMin"] = min(pf.values()); R["OVFirmSizePMax"] = max(pf.values())
    names5 = {"main": "main", "unavailable_only": "Unavailable-only", "majority": "majority", "strict": "strict", "partial_only": "partial-only"}
    R["OVFirmSizePMinName"] = names5[min(pf, key=pf.get)]; R["OVFirmSizePMaxName"] = names5[max(pf, key=pf.get)]
    # software versions of the run that produced the macros, and the download dates from the download log
    R["PyVer"] = platform.python_version()
    for pkg, nm in (("numpy", "NumpyVer"), ("scipy", "ScipyVer"), ("pandas", "PandasVer"), ("statsmodels", "StatsmodelsVer"), ("matplotlib", "MplVer")):
        R[nm] = importlib.metadata.version(pkg)
    dates = sorted({ln[:10] for ln in open(os.path.join(HERE, "data", "raw", "download_log.txt")) if "https://" in ln and "->" in ln and "MDPI_template" not in ln})
    dates = [dt.date.fromisoformat(d) for d in dates]
    if len(dates) == 1 or dates[0] == dates[-1]: R["DownloadDates"] = dates[0].strftime("%-d %B %Y")
    elif dates[0].month == dates[-1].month and dates[0].year == dates[-1].year: R["DownloadDates"] = f"{dates[0].day}--{dates[-1].day} {dates[-1].strftime('%B %Y')}"
    else: R["DownloadDates"] = f"{dates[0].strftime('%-d %B %Y')} to {dates[-1].strftime('%-d %B %Y')}"
    # blind re-reading of a random sample of free-text availability strings (analysis/coding_sample.py kappa)
    kp = os.path.join(RES, "coding_sample_kappa.json")
    if os.path.exists(kp):
        kj = json.load(open(kp)); first = next((kj[c] for c in ("hand_author1", "hand_author2") if c in kj), None)
        if first:
            R["HandCodeN"] = first["n"]; R["HandCodeAgreePct"] = 100 * first["agreement"]; R["HandCodeKappa"] = first["kappa"]
            R["HandCodeDisagree"] = len(first["disagree_ids"])
        if "between" in kj:
            R["HandCodeBetweenN"] = kj["between"]["n"]; R["HandCodeBetweenKappa"] = kj["between"]["kappa"]
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(R, f, indent=1)
    m_, e_ = f"{R['MinEigAll']:.1e}".split("e"); R["MinEigAll"] = f"\\ensuremath{{{m_}\\times10^{{{int(e_)}}}}}"
    lines = ["% generated by make_numbers.py -- do not edit"]
    for k, v in R.items():
        posterior_key = bool(re.search(r"(Med|Lo|Hi|Est|True|Mean|Obs|Var|Bias|Width|Ratio|Binom|Q|AUC|OR|Diff|Cov|AFF|AFM|Rate|Z|W)$", k))
        if isinstance(v, float) and v.is_integer() and not posterior_key:
            v = int(v)
        if isinstance(v, float) and "ESS" in k:
            v = int(round(v))
        if isinstance(v, float):
            if abs(v) >= 1000: s = f"{v:,.0f}"
            elif "Pct" in k: s = f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
            elif k.endswith("P") and "Pct" not in k:
                s = f"{v:.2f}" if v >= 0.0095 else (f"{v:.3f}" if v >= 0.001 else "<0.001")
                if s == "0.05" and v < 0.05: s = f"{v:.3f}"              # do not print a p below 0.05 as 0.05
            elif k.endswith("Tail"):                                   # binomial tail probabilities, scientific below 0.001
                if v >= 0.0095: s = f"{v:.2f}"
                elif v >= 0.001: s = f"{v:.3f}"
                else:
                    m, e = f"{v:.0e}".split("e"); s = f"${m}\\times10^{{{int(e)}}}$"
            elif abs(v) >= 10: s = f"{v:.1f}"
            elif abs(v) >= 1: s = f"{v:.2f}"
            else: s = f"{v:.3f}" if abs(v) < 0.1 else f"{v:.2f}"
        elif isinstance(v, int): s = f"{v:,}" if abs(v) >= 10000 else str(v)
        elif k.endswith("Bins"): s = str(v)                      # decile counts joined by "&" for the rank-histogram table
        else: s = str(v).replace("&", r"\&").replace("%", r"\%")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and s.startswith("-"):
            s = "\\ensuremath{-}" + s[1:]                               # a typographic minus sign in text and in math
        lines.append(f"\\newcommand{{\\{k}}}{{{s}}}")
        if isinstance(v, float) and k.endswith("P") and "Pct" not in k:      # p-value with its relation sign, for "$p\XEq$"
            lines.append(f"\\newcommand{{\\{k}Eq}}{{{s if s.startswith('<') else '=' + s}}}")
    with open(os.path.join(RES, "numbers.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(MANU, exist_ok=True); shutil.copy(os.path.join(RES, "numbers.tex"), os.path.join(MANU, "numbers.tex"))
    print(f"wrote {len(R)} macros")


if __name__ == "__main__":
    main()
