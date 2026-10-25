"""
theory.py -- the formal results of Section 2.5 (Lemma 1, Theorems 1-2, Propositions 1-2, Remark 1), implemented exactly as stated.

Notation (manuscript Section 2.5). Same model as twotier.py: Z_f ~ Bern(phi), W_m ~ Bern(mu), and
    pi(z, w) = P(y_fm = 1 | Z_f = z, W_m = w) = 1 - (1 - eps) (1 - a)^z (1 - b)^w.
Row-class rates (Lemma 1):
    firm row (cells of one firm, all in different drugs), given Z_f = z:  p_z = 1 - (1 - eps)(1 - a)^z (1 - mu b)
    drug row (cells of one drug, all with different firms), given W_m = w: q_w = 1 - (1 - eps)(1 - b)^w (1 - phi a)
Row moments (eq. (7)): M^F_j = phi p_1^j + (1 - phi) p_0^j,   M^D_j = mu q_1^j + (1 - mu) q_0^j.
Theorem 1 (eq. (8)): (phi, p_0, p_1) are recovered from (M^F_1, M^F_2, M^F_3) by the two-point (Prony) inversion `prony`, and
    likewise (mu, q_0, q_1) from the drug moments; then a, b and eps follow (eq. (9)).
Proposition 1: AF_firm = 1 - p_0 / M_1 and AF_drug = 1 - q_0 / M_1, and AF_firm > AF_drug  <=>  phi a > mu b.
Theorem 2: the probability that all n listed firms of a drug are unavailable is O_n = M^D_n (eq. (13)); O_n is
    log-convex in n, O_n >= M_1^n, O_{n+1}/O_n increases to q_1, and the smallest n with O_n <= alpha lies between
    the bounds in `n_star_bounds`.
Proposition 2: for two drugs whose listed firm sets share k firms, the probability that both are completely
    unavailable (eq. (15)) is strictly increasing in k.
Corollary 1: the unavailable share S = U/k of a row with k cells has mean M_1 and variance
    M_1 (1 - M_1)/k + (1 - 1/k) D, with D = M_2 - M_1^2 the row's Hankel determinant (= the sharing covariance), so the
    expected dispersion ratio of the row is 1 + (k - 1) D / (M_1 (1 - M_1)); and D_F > D_M  <=>
    sqrt(phi (1 - phi)) a (1 - mu b) > sqrt(mu (1 - mu)) b (1 - phi a), which is a different comparison from phi a > mu b; (d) the conditional co-unavailability of two cells of one row is
    M_2/M_1 = M_1 + D/M_1.
Corollary 2: the number N of completely unavailable drugs in a portfolio has Var(N) >= its value with disjoint supplier
    sets, strictly if any two drugs share a listed firm (portfolio_outage_moments).
Proposition 3 (persistence across snapshots): with stationary latent processes Z_f(t) ~ Bern(phi), W_m(t) ~ Bern(mu)
    (independent across units and tiers; r_F(s, t) = Corr(Z_f(s), Z_f(t)), r_M likewise) and cells conditionally
    independent given the latent paths, Cov(y_fm(s), y_fm'(t)) = r_F D_F for m != m', Cov(y_fm(s), y_f'm(t)) = r_M D_M
    for f != f', and the change covariances are 2 (1 - r_F) D_F and 2 (1 - r_M) D_M (lagged_covariances; checked by
    brute_lagged_covariances).
"""
import itertools
from math import comb

import numpy as np


def pi(eps, a, b, z, w):
    return 1.0 - (1.0 - eps) * (1.0 - a) ** z * (1.0 - b) ** w


def firm_row_rates(eps, a, b, phi, mu):
    """Lemma 1: (p_0, p_1)."""
    return 1.0 - (1.0 - eps) * (1.0 - mu * b), 1.0 - (1.0 - eps) * (1.0 - a) * (1.0 - mu * b)


def drug_row_rates(eps, a, b, phi, mu):
    """Lemma 1: (q_0, q_1)."""
    return 1.0 - (1.0 - eps) * (1.0 - phi * a), 1.0 - (1.0 - eps) * (1.0 - b) * (1.0 - phi * a)


def firm_moment(j, eps, a, b, phi, mu):
    """Eq. (7), firm side: probability that j given cells of one firm (in distinct drugs) are all unavailable."""
    p0, p1 = firm_row_rates(eps, a, b, phi, mu)
    return phi * p1 ** j + (1.0 - phi) * p0 ** j


def drug_moment(j, eps, a, b, phi, mu):
    """Eq. (7), drug side, = O_j of Theorem 2."""
    q0, q1 = drug_row_rates(eps, a, b, phi, mu)
    return mu * q1 ** j + (1.0 - mu) * q0 ** j


def prony(m1, m2, m3):
    """Theorem 1 (two-point inversion). Given the first three moments of a two-point distribution on [0,1]
    (weight w on `hi`, 1-w on `lo`), return (w, lo, hi), or None if the Hankel determinant D = m2 - m1^2 is not
    positive or the roots are not real numbers in [0,1] (the moments are then not those of a non-degenerate
    two-point law). s = lo + hi and t = lo * hi solve m2 = s m1 - t and m3 = s m2 - t m1."""
    D = m2 - m1 ** 2
    if not D > 0:
        return None
    s = (m3 - m1 * m2) / D
    t = (m1 * m3 - m2 ** 2) / D
    disc = s * s - 4.0 * t
    if disc < 0:
        return None
    r = np.sqrt(disc)
    lo, hi = (s - r) / 2.0, (s + r) / 2.0
    if lo < -1e-12 or hi > 1 + 1e-12 or hi - lo <= 0:
        return None
    w = (m1 - lo) / (hi - lo)
    if not (0 < w < 1):
        return None
    return float(w), float(lo), float(hi)


def params_from_rates(phi, p0, p1, mu, q0, q1):
    """Eq. (9): (eps, a, b, phi, mu) from the two row-class decompositions.
    a = 1 - (1-p_1)/(1-p_0), b = 1 - (1-q_1)/(1-q_0), 1 - eps = (1-p_0)/(1 - mu b)."""
    a = 1.0 - (1.0 - p1) / (1.0 - p0)
    b = 1.0 - (1.0 - q1) / (1.0 - q0)
    eps = 1.0 - (1.0 - p0) / (1.0 - mu * b)
    return dict(eps=eps, a=a, b=b, phi=phi, mu=mu)


def identify(firm_moments, drug_moments):
    """Theorem 1 end to end: moments (M_1, M_2, M_3) of each side -> parameters. None if either side degenerates."""
    f = prony(*firm_moments); d = prony(*drug_moments)
    if f is None or d is None:
        return None
    return params_from_rates(f[0], f[1], f[2], d[0], d[1], d[2])


def hankel(side, eps, a, b, phi, mu):
    """D = M_2 - M_1^2 for a firm row (side='firm') or a drug row (side='drug'); eqs. (3)-(4)."""
    mom = firm_moment if side == "firm" else drug_moment
    return mom(2, eps, a, b, phi, mu) - mom(1, eps, a, b, phi, mu) ** 2


def share_variance(k, side, eps, a, b, phi, mu):
    """Corollary 1: variance of the unavailable share U/k of a row of k cells."""
    m1 = firm_moment(1, eps, a, b, phi, mu)
    return m1 * (1.0 - m1) / k + (1.0 - 1.0 / k) * hankel(side, eps, a, b, phi, mu)


def dispersion_ratio_expected(k, side, eps, a, b, phi, mu):
    """Corollary 1: Var(U/k) divided by its binomial value M_1 (1 - M_1)/k."""
    m1 = firm_moment(1, eps, a, b, phi, mu)
    return 1.0 + (k - 1) * hankel(side, eps, a, b, phi, mu) / (m1 * (1.0 - m1))


def covariance_order_firm(eps, a, b, phi, mu):
    """Corollary 1(c): True iff D_F > D_M, via sqrt(phi(1-phi)) a (1-mu b) > sqrt(mu(1-mu)) b (1-phi a)."""
    return bool(np.sqrt(phi * (1 - phi)) * a * (1 - mu * b) > np.sqrt(mu * (1 - mu)) * b * (1 - phi * a))


def co_unavailability(side, eps, a, b, phi, mu):
    """Corollary 1(d): P(a second cell of the same row is unavailable | a first one is) = M_2 / M_1 = M_1 + D / M_1;
    for cells sharing neither firm nor drug it is M_1."""
    mom = firm_moment if side == "firm" else drug_moment
    return mom(2, eps, a, b, phi, mu) / mom(1, eps, a, b, phi, mu)


def af_from_rates(p0, q0, m1):
    """Proposition 1: AF_firm = 1 - p_0/M_1, AF_drug = 1 - q_0/M_1."""
    return 1.0 - p0 / m1, 1.0 - q0 / m1


def outage(n, eps, a, b, phi, mu):
    """Theorem 2, eq. (13): probability that all n listed firms of a drug are unavailable."""
    return drug_moment(n, eps, a, b, phi, mu)


def outage_independent(n, eps, a, b, phi, mu):
    """Benchmark of Theorem 2(b): cells treated as independent with the marginal rate."""
    m1 = drug_moment(1, eps, a, b, phi, mu)
    return m1 ** n


def n_star(alpha, eps, a, b, phi, mu, n_max=10_000):
    """Smallest n >= 1 with O_n <= alpha (np.inf if none up to n_max)."""
    for n in range(1, n_max + 1):
        if outage(n, eps, a, b, phi, mu) <= alpha:
            return n
    return np.inf


def n_star_bounds(alpha, eps, a, b, phi, mu):
    """Theorem 2(d): ceil(log(alpha/mu)/log q_1) <= n*(alpha) <= ceil(log(alpha)/log q_1), for alpha < mu, q_1 < 1.
    The lower bound is floored at 1."""
    q0, q1 = drug_row_rates(eps, a, b, phi, mu)
    lo = max(1, int(np.ceil(np.log(alpha / mu) / np.log(q1) - 1e-12)))
    hi = max(1, int(np.ceil(np.log(alpha) / np.log(q1) - 1e-12)))
    return lo, hi


def joint_outage(n1, n2, k, eps, a, b, phi, mu):
    """Proposition 2, eq. (15): probability that two drugs with n1 and n2 listed firms, k of them shared, are both
    completely unavailable.  = sum_{w,w'} P(w)P(w') r_w^{n1-k} r_{w'}^{n2-k} g(w,w')^k,
    with r_w = q_w and g(w,w') = E_Z[pi(Z,w) pi(Z,w')]."""
    q = drug_row_rates(eps, a, b, phi, mu)
    tot = 0.0
    for w in (0, 1):
        for w2 in (0, 1):
            pw = (mu if w else 1 - mu) * (mu if w2 else 1 - mu)
            g = phi * pi(eps, a, b, 1, w) * pi(eps, a, b, 1, w2) + (1 - phi) * pi(eps, a, b, 0, w) * pi(eps, a, b, 0, w2)
            tot += pw * q[w] ** (n1 - k) * q[w2] ** (n2 - k) * g ** k
    return tot


def portfolio_outage_moments(n, K, eps, a, b, phi, mu):
    """Corollary 2: number N of completely unavailable drugs in a portfolio of drugs with n[i] listed firms, K[i, j] of
    them shared by drugs i and j. Returns (E[N], Var(N), Var(N) if no two drugs shared a firm). Uses eq. (13) for the
    means and Proposition 2 (eq. (15)) for the pairwise joint outages; drug states are independent across drugs, so drugs
    are dependent only through shared firms."""
    n = np.asarray(n); K = np.asarray(K); O = np.array([outage(int(x), eps, a, b, phi, mu) for x in n])
    var_disjoint = float(np.sum(O * (1 - O))); cov = 0.0
    for i in range(len(n)):
        for j in range(i + 1, len(n)):
            if K[i, j] > 0:
                cov += joint_outage(int(n[i]), int(n[j]), int(K[i, j]), eps, a, b, phi, mu) - O[i] * O[j]
    return float(O.sum()), var_disjoint + 2.0 * cov, var_disjoint


# ---------------------------------------------------------------------------------------------------------------
# Brute-force versions (unit tests only): enumerate every latent state of a small explicit graph.

def brute_share_moments(k, eps, a, b, phi, mu):
    """E[S] and E[S^2] of the unavailable share S = U/k of one firm row of k cells (firm 0 in drugs 0..k-1), by
    enumerating every latent configuration and every outcome vector."""
    e1 = e2 = 0.0
    for z in (0, 1):
        for wc in itertools.product([0, 1], repeat=k):
            pl = (phi if z else 1 - phi) * np.prod([mu if w else 1 - mu for w in wc])
            pc = [pi(eps, a, b, z, w) for w in wc]
            for yc in itertools.product([0, 1], repeat=k):
                py = np.prod([p if y else 1 - p for p, y in zip(pc, yc)])
                s = sum(yc) / k; e1 += pl * py * s; e2 += pl * py * s * s
    return e1, e2


def brute_all_unavailable(cells_sets, F, M, eps, a, b, phi, mu):
    """P(every cell in each listed set is unavailable) for an explicit graph: cells_sets is a list of (f, m) cells.
    Enumerates all 2^(F+M) latent configurations."""
    tot = 0.0
    for zc in itertools.product([0, 1], repeat=F):
        for wc in itertools.product([0, 1], repeat=M):
            pr = np.prod([phi if z else 1 - phi for z in zc]) * np.prod([mu if w else 1 - mu for w in wc])
            for f, m in cells_sets:
                pr *= pi(eps, a, b, zc[f], wc[m])
            tot += pr
    return tot


# ---------------------------------------------------------------------------------------------------------------
# Method-of-moments estimator (Section 2.6): unbiased row moments from data.

def row_moments(y, idx, min_size=3):
    """Unbiased estimates of M_1, M_2, M_3 for one side: for each row (firm or drug) with k >= min_size cells and
    u unavailable, C(u, j)/C(k, j) is unbiased for M_j (Lemma 1: exchangeable, conditionally i.i.d. within a row);
    rows are averaged with equal weight. Returns (moments, number of rows)."""
    y = np.asarray(y); idx = np.asarray(idx)
    k = np.bincount(idx); u = np.bincount(idx, weights=y).astype(int)
    keep = k >= min_size
    if keep.sum() == 0:
        return None, 0
    ks, us = k[keep], u[keep]
    m = [float(np.mean([comb(int(uu), j) / comb(int(kk), j) for uu, kk in zip(us, ks)])) for j in (1, 2, 3)]
    return tuple(m), int(keep.sum())


def mom_estimate(y, fi, mi, min_size=3):
    """Theorem 1 applied to the estimated row moments. Each side is inverted separately; the attributable
    fractions use the side's own M_1 (Proposition 1), so AF_firm needs only the firm side and AF_drug only the drug
    side. Returns a dict; entries are None where a side's moments are degenerate."""
    mf, nf = row_moments(y, fi, min_size); md, nd = row_moments(y, mi, min_size)
    out = dict(n_firm_rows=nf, n_drug_rows=nd, firm_moments=mf, drug_moments=md,
               firm_hankel=None if mf is None else mf[1] - mf[0] ** 2,
               drug_hankel=None if md is None else md[1] - md[0] ** 2)
    f = prony(*mf) if mf else None; d = prony(*md) if md else None
    out["firm"] = None if f is None else dict(phi=f[0], p0=f[1], p1=f[2], af_firm=1 - f[1] / mf[0])
    out["drug"] = None if d is None else dict(mu=d[0], q0=d[1], q1=d[2], af_drug=1 - d[1] / md[0])
    if f is not None and d is not None:
        out["params"] = params_from_rates(f[0], f[1], f[2], d[0], d[1], d[2])
        out["phi_a_gt_mu_b"] = bool(d[1] > f[1])          # Proposition 1: ordering <=> q_0 > p_0
    return out


def lagged_covariances(eps, a, b, phi, mu, r_F, r_M):
    """Proposition 3: same-firm (different drugs) and same-drug (different firms) covariances of cells observed at two
    times whose latent indicators have correlations r_F and r_M, and the covariances of the changes y(t) - y(s)."""
    DF, DM = hankel("firm", eps, a, b, phi, mu), hankel("drug", eps, a, b, phi, mu)
    return dict(cov_firm_lag=r_F * DF, cov_drug_lag=r_M * DM, cov_firm_change=2 * (1 - r_F) * DF,
                cov_drug_change=2 * (1 - r_M) * DM, D_F=DF, D_M=DM)


def carryover_covariances(eps, a, b, phi, mu, r_F, kappa):
    """Proposition 3 with cell-level carry-over: between s and t each cell independently keeps its status at s with
    probability kappa and is otherwise redrawn from the model given the latent states at t. For two cells of one firm in
    different drugs: the cross-date covariance (kappa + (1 - kappa) r_F) D_F, the same-date covariance at t
    (1 - 2 kappa (1 - kappa)(1 - r_F)) D_F, and the ratio of the cross-date covariance to the mean of the two same-date
    covariances, (kappa + (1 - kappa) r_F) / (1 - kappa (1 - kappa)(1 - r_F)); and the covariance of the two cells'
    changes, 2 (1 - kappa)^2 (1 - r_F) D_F."""
    DF = hankel("firm", eps, a, b, phi, mu)
    cross = (kappa + (1 - kappa) * r_F) * DF
    same_t = (1 - 2 * kappa * (1 - kappa) * (1 - r_F)) * DF
    return dict(cov_firm_lag=cross, cov_firm_same_t=same_t, ratio=cross / ((DF + same_t) / 2), D_F=DF,
                cov_firm_change=2 * (1 - kappa) ** 2 * (1 - r_F) * DF)


def brute_carryover_covariances(eps, a, b, phi, mu, r_F, r_M, kappa):
    """Exact enumeration for carryover_covariances: one firm and two drugs observed at s and t, every latent path, every
    pair of carry-over indicators and every outcome. Returns Cov(y_1(s), y_2(t)) and Cov(y_1(t), y_2(t))."""
    zp, wp = _pair_path_probs(phi, r_F), _pair_path_probs(mu, r_M)
    E = dict(m1s=0.0, m2t=0.0, m1t=0.0, x=0.0, tt=0.0, ss=0.0, m2s=0.0, x21=0.0)
    for (z0, z1), w1, w2, c1, c2 in itertools.product(zp, wp, wp, (0, 1), (0, 1)):
        pr = zp[(z0, z1)] * wp[w1] * wp[w2] * (kappa if c1 else 1 - kappa) * (kappa if c2 else 1 - kappa)
        if pr <= 0:
            continue
        P1s, P1t = pi(eps, a, b, z0, w1[0]), pi(eps, a, b, z1, w1[1])
        P2s, P2t = pi(eps, a, b, z0, w2[0]), pi(eps, a, b, z1, w2[1])
        # given latents and carry-over indicators the two cells are independent; a kept cell at t equals itself at s
        m1t = P1s if c1 else P1t; m2t = P2s if c2 else P2t
        E["m1s"] += pr * P1s; E["m1t"] += pr * m1t; E["m2t"] += pr * m2t
        E["x"] += pr * P1s * m2t          # y_1(s) y_2(t): y_2(t) is y_2(s) or a fresh draw, independent of y_1(s) either way
        E["tt"] += pr * m1t * m2t
        E["ss"] += pr * P1s * P2s; E["m2s"] += pr * P2s
        E["x21"] += pr * P2s * m1t                     # y_2(s) y_1(t)
    c = lambda a_, b_, ma, mb: a_ - ma * mb
    lag12 = c(E["x"], 0, E["m1s"], E["m2t"]); lag21 = c(E["x21"], 0, E["m2s"], E["m1t"])
    same_s = c(E["ss"], 0, E["m1s"], E["m2s"]); same_t = c(E["tt"], 0, E["m1t"], E["m2t"])
    return dict(cov_firm_lag=lag12, cov_firm_same_t=same_t, cov_firm_change=same_t + same_s - lag12 - lag21)


def _pair_path_probs(p, r):
    """Joint distribution of a stationary binary indicator at two times with P(=1) = p and correlation r."""
    p11 = p * p + r * p * (1 - p)
    return {(1, 1): p11, (1, 0): p - p11, (0, 1): p - p11, (0, 0): 1 - 2 * p + p11}


def brute_lagged_covariances(eps, a, b, phi, mu, r_F, r_M):
    """Exact enumeration for Proposition 3 on two firms x two drugs observed at times s and t: every latent path
    (each firm and each drug a binary indicator at the two times) and every outcome of the four cells at both times.
    Returns Cov(y_11(s), y_12(t)) (same firm), Cov(y_11(s), y_21(t)) (same drug), Cov(y_11(s), y_22(t)) (neither)
    and the change covariances Cov(y_11(t) - y_11(s), y_12(t) - y_12(s)) and Cov(y_11(t) - y_11(s), y_21(t) - y_21(s))."""
    zp, wp = _pair_path_probs(phi, r_F), _pair_path_probs(mu, r_M)
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]          # (firm, drug)
    E = {}                                             # moments E[y_c(u) y_c'(v)] and E[y_c(u)]
    mean = np.zeros((4, 2)); prod = np.zeros((4, 2, 4, 2))
    for z0, z1, w0, w1 in itertools.product(zp, zp, wp, wp):
        pr = zp[z0] * zp[z1] * wp[w0] * wp[w1]
        if pr <= 0:
            continue
        Z, W = (z0, z1), (w0, w1)
        # conditional probabilities of each cell at each time (index 0 = s, 1 = t)
        P = np.array([[pi(eps, a, b, Z[f][u], W[m][u]) for u in (0, 1)] for f, m in cells])
        mean += pr * P
        for i in range(4):
            for u in range(2):
                for j in range(4):
                    for v in range(2):
                        prod[i, u, j, v] += pr * (P[i, u] if (i, u) == (j, v) else P[i, u] * P[j, v])
    cov = lambda i, u, j, v: prod[i, u, j, v] - mean[i, u] * mean[j, v]
    ch = lambda i, j: cov(i, 1, j, 1) + cov(i, 0, j, 0) - cov(i, 1, j, 0) - cov(i, 0, j, 1)
    return dict(cov_firm_lag=cov(0, 0, 1, 1), cov_drug_lag=cov(0, 0, 2, 1), cov_none_lag=cov(0, 0, 3, 1),
                cov_firm_change=ch(0, 1), cov_drug_change=ch(0, 2))

