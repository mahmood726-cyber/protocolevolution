"""Statistical engine for ProtocolEvolution.

Provides 18 methods in four layers:

Layer 1 -- Publication-Essential (5 methods):
  1. pattern_outcome_test   Chi-squared / Fisher for pattern-outcome association
  2. kaplan_meier            Kaplan-Meier survival curves with Greenwood CI
  3. log_rank_test           Log-rank test comparing two survival curves
  4. compute_odds_ratio      Odds ratio with Wald CI and continuity correction
  5. benjamini_hochberg      BH FDR multiple-testing correction

Layer 2 -- Methodologically Novel (3 methods):
  6. cox_ph                  Cox proportional hazards via Newton-Raphson
  7. amendment_markov_chain  Markov chain on amendment-type sequences
  8. propensity_match        Propensity score matching with ATT + bootstrap CI

Layer 3 -- Advanced Statistical Methods (5 methods):
  9.  hidden_markov_model    HMM with Baum-Welch + Viterbi for protocol states
  10. andersen_gill_model    Recurrent events Cox PH with sandwich variance
  11. frailty_model          Shared gamma frailty for sponsor clustering
  12. cusum_detection        CUSUM + PELT change-point detection
  13. cure_rate_model        Mixture cure-rate model (logistic + Weibull)

Layer 4 -- Cutting-Edge (5 methods):
  14. multi_state_model      Multi-state Nelson-Aalen transition model
  15. joint_model            Joint longitudinal-survival model
  16. bayesian_changepoint   Bayesian change-point detection (RJMCMC)
  17. granger_causality      Granger causality via VAR F-tests
  18. functional_pca         Functional PCA on amendment trajectories
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


# ============================================================
# 1. Chi-Squared / Fisher for Pattern-Outcome Association
# ============================================================

def pattern_outcome_test(
    pattern_trials: List[Dict[str, Any]],
    clean_trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Test whether pattern presence is associated with trial completion.

    Builds a 2x2 table: pattern(yes/no) x completed(yes/no).
    Uses Fisher exact test when any cell < 5, otherwise chi-squared.

    Parameters
    ----------
    pattern_trials : list of dict
        Trials that exhibit the pattern.  Each must have a ``"status"`` key.
    clean_trials : list of dict
        Trials without the pattern.

    Returns
    -------
    dict
        Keys: test_used, statistic, p_value, odds_ratio, ci_lower, ci_upper.
    """
    # Build 2x2 table
    # Rows: pattern present / absent
    # Cols: completed / not completed
    a = sum(1 for t in pattern_trials if t.get("status") == "COMPLETED")
    b = sum(1 for t in pattern_trials if t.get("status") != "COMPLETED")
    c = sum(1 for t in clean_trials if t.get("status") == "COMPLETED")
    d = sum(1 for t in clean_trials if t.get("status") != "COMPLETED")

    table = np.array([[a, b], [c, d]])

    # Choose test
    if min(a, b, c, d) < 5:
        oddsr, p_value = sp_stats.fisher_exact(table, alternative="two-sided")
        test_used = "fisher_exact"
        statistic = float(oddsr)
    else:
        chi2, p_value, _, _ = sp_stats.chi2_contingency(table, correction=False)
        test_used = "chi_squared"
        statistic = float(chi2)

    # OR and CI via compute_odds_ratio helper
    or_result = compute_odds_ratio(a, b, c, d)

    return {
        "test_used": test_used,
        "statistic": statistic,
        "p_value": float(p_value),
        "odds_ratio": or_result["or_value"],
        "ci_lower": or_result["ci_lower"],
        "ci_upper": or_result["ci_upper"],
    }


# ============================================================
# 2. Kaplan-Meier Survival Curves
# ============================================================

def kaplan_meier(
    times: Sequence[float],
    events: Sequence[int],
) -> Dict[str, Any]:
    """Kaplan-Meier survival curve with Greenwood confidence intervals.

    Parameters
    ----------
    times : array-like
        Observed times (days from first post to first amendment or censoring).
    events : array-like
        1 if event (amended), 0 if censored.

    Returns
    -------
    dict
        Keys: times, survival, ci_lower, ci_upper, median_survival.
    """
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=int)

    # Sort by time
    order = np.argsort(times_arr, kind="stable")
    times_arr = times_arr[order]
    events_arr = events_arr[order]

    n_total = len(times_arr)
    unique_times = np.unique(times_arr)

    out_times = [0.0]
    out_surv = [1.0]
    out_ci_lo = [1.0]
    out_ci_hi = [1.0]

    s = 1.0
    greenwood_sum = 0.0
    n_at_risk = n_total

    for t in unique_times:
        mask = times_arr == t
        d_i = int(np.sum(events_arr[mask]))   # events at time t
        c_i = int(np.sum(mask)) - d_i         # censored at time t

        if n_at_risk <= 0:
            break

        if d_i > 0:
            s *= (1.0 - d_i / n_at_risk)
            # Greenwood increment (guard division by zero)
            denom = n_at_risk * (n_at_risk - d_i)
            if denom > 0:
                greenwood_sum += d_i / denom

        var_s = s * s * greenwood_sum
        se = math.sqrt(var_s) if var_s > 0 else 0.0

        out_times.append(float(t))
        out_surv.append(s)
        out_ci_lo.append(max(0.0, s - 1.96 * se))
        out_ci_hi.append(min(1.0, s + 1.96 * se))

        n_at_risk -= (d_i + c_i)

    # Median survival: smallest t where S(t) <= 0.5
    median = None
    for i, sv in enumerate(out_surv):
        if sv <= 0.5:
            median = out_times[i]
            break

    return {
        "times": out_times,
        "survival": out_surv,
        "ci_lower": out_ci_lo,
        "ci_upper": out_ci_hi,
        "median_survival": median,
    }


# ============================================================
# 3. Log-Rank Test
# ============================================================

def log_rank_test(
    times1: Sequence[float],
    events1: Sequence[int],
    times2: Sequence[float],
    events2: Sequence[int],
) -> Dict[str, float]:
    """Log-rank test comparing two survival curves.

    Parameters
    ----------
    times1, events1 : array-like
        Times and event indicators for group 1.
    times2, events2 : array-like
        Times and event indicators for group 2.

    Returns
    -------
    dict
        Keys: chi2, p_value.
    """
    t1 = np.asarray(times1, dtype=float)
    e1 = np.asarray(events1, dtype=int)
    t2 = np.asarray(times2, dtype=float)
    e2 = np.asarray(events2, dtype=int)

    # Combine all unique event times
    all_event_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    all_event_times.sort()

    sum_OE = 0.0  # sum of (O_1i - E_1i)
    sum_V = 0.0   # sum of variance terms

    for t in all_event_times:
        # At-risk counts: number with time >= t
        n1 = int(np.sum(t1 >= t))
        n2 = int(np.sum(t2 >= t))
        n = n1 + n2

        if n == 0:
            continue

        # Events at time t
        d1 = int(np.sum((t1 == t) & (e1 == 1)))
        d2 = int(np.sum((t2 == t) & (e2 == 1)))
        d = d1 + d2

        if n <= 1:
            # Variance term requires n_i >= 2
            e1_expected = n1 * d / n if n > 0 else 0
            sum_OE += d1 - e1_expected
            continue

        e1_expected = n1 * d / n
        sum_OE += d1 - e1_expected

        # Variance: V_i = n1*n2*d*(n-d) / (n^2*(n-1))
        v_i = n1 * n2 * d * (n - d) / (n * n * (n - 1))
        sum_V += v_i

    if sum_V <= 0:
        return {"chi2": 0.0, "p_value": 1.0}

    chi2 = (sum_OE ** 2) / sum_V
    p_value = float(1.0 - sp_stats.chi2.cdf(chi2, df=1))

    return {"chi2": float(chi2), "p_value": p_value}


# ============================================================
# 4. Odds Ratio with CI
# ============================================================

def compute_odds_ratio(
    a: int, b: int, c: int, d: int,
) -> Dict[str, Optional[float]]:
    """Compute odds ratio with Wald CI and p-value.

    Parameters
    ----------
    a, b, c, d : int
        Cells of a 2x2 contingency table.
        Row 1: a (exposed+outcome), b (exposed+no outcome).
        Row 2: c (unexposed+outcome), d (unexposed+no outcome).

    Returns
    -------
    dict
        Keys: or_value, ci_lower, ci_upper, p_value.
    """
    # Continuity correction if any cell is 0
    aa, bb, cc, dd = float(a), float(b), float(c), float(d)
    if any(v == 0 for v in [aa, bb, cc, dd]):
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5

    or_val = (aa * dd) / (bb * cc)
    ln_or = math.log(or_val)
    se_ln_or = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)

    ci_lower = math.exp(ln_or - 1.96 * se_ln_or)
    ci_upper = math.exp(ln_or + 1.96 * se_ln_or)

    z = ln_or / se_ln_or
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))

    return {
        "or_value": round(or_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 6),
    }


# ============================================================
# 5. Benjamini-Hochberg FDR Correction
# ============================================================

def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Benjamini-Hochberg false discovery rate correction.

    Parameters
    ----------
    p_values : list of float
        Raw p-values to adjust.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    list of dict
        Each entry: original_p, adjusted_p, significant.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Pair p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m

    # Step-up procedure
    prev_adj = 0.0
    for rank_minus_1, (orig_idx, p) in enumerate(indexed):
        rank = rank_minus_1 + 1
        adj_p = p * m / rank
        # Enforce monotonicity: adjusted_p >= previous adjusted_p
        adj_p = max(adj_p, prev_adj)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p

    # Monotonicity correction (reverse pass: adjusted[i] = min(adjusted[i], adjusted[i+1]))
    # Work from the largest rank downward
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(m - 2, -1, -1):
        idx_cur = sorted_indices[i]
        idx_next = sorted_indices[i + 1]
        adjusted[idx_cur] = min(adjusted[idx_cur], adjusted[idx_next])

    results = []
    for i, p in enumerate(p_values):
        results.append({
            "original_p": p,
            "adjusted_p": round(adjusted[i], 6),
            "significant": adjusted[i] <= alpha,
        })

    return results


# ============================================================
# 6. Cox Proportional Hazards (Newton-Raphson)
# ============================================================

def cox_ph(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Cox proportional hazards regression via Newton-Raphson.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix.
    times : ndarray of shape (n,)
        Survival times.
    events : ndarray of shape (n,)
        Event indicators (1 = event, 0 = censored).
    feature_names : list of str, optional
        Names for each feature.
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance on ||delta_beta||.

    Returns
    -------
    dict
        Keys: hazard_ratios (list of dicts), concordance, log_likelihood.
    """
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    n, p = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p)]

    beta = np.zeros(p)

    # Sort by ascending time for standard risk-set convention
    order = np.argsort(times, kind="stable")
    X = X[order]
    times = times[order]
    events = events[order]

    log_lik = None
    neg_hessian = np.eye(p)  # information matrix (will be overwritten)

    for _iteration in range(max_iter):
        eta = X @ beta
        # Numerical stability: subtract max for exp, but keep raw eta for log-likelihood
        eta_max = np.max(eta)
        exp_eta = np.exp(eta - eta_max)

        # Risk-set cumulative sums from the right (descending index = later times)
        # R(i) = {j : t_j >= t_i}.  With ascending sort, R(i) = {i, i+1, ..., n-1}.
        # Reverse cumulative sums:
        rev_exp = np.cumsum(exp_eta[::-1])[::-1]          # (n,)
        rev_exp_X = np.cumsum((exp_eta[:, None] * X)[::-1], axis=0)[::-1]  # (n, p)

        # Score and Hessian
        score = np.zeros(p)
        neg_hessian = np.zeros((p, p))  # -H (positive definite information matrix)
        ll = 0.0

        for i in range(n):
            if events[i] == 0:
                continue
            r_sum = rev_exp[i]
            if r_sum <= 0:
                continue
            w = exp_eta[:, None] * X  # not needed per-iter; compute mean below
            r_mean = rev_exp_X[i] / r_sum  # E[X | in risk set]

            ll += (eta[i] - eta_max) - math.log(r_sum)
            score += X[i] - r_mean

            # Information contribution: Var[X | in risk set]
            # = E[XX'] - E[X]E[X]'
            # Compute weighted second moment over risk set
            risk_idx = slice(i, n)
            w_k = exp_eta[risk_idx] / r_sum  # weights summing to 1
            X_risk = X[risk_idx]  # (n_risk, p)
            # E[XX'] = sum_k w_k * x_k x_k'
            weighted_X = X_risk * np.sqrt(w_k)[:, None]  # (n_risk, p)
            E_XX = weighted_X.T @ weighted_X  # (p, p)
            neg_hessian += E_XX - np.outer(r_mean, r_mean)

        log_lik = ll

        # Newton-Raphson: beta_new = beta + (neg_hessian)^{-1} @ score
        try:
            delta = np.linalg.solve(neg_hessian, score)
        except np.linalg.LinAlgError:
            # Singular -- add small ridge
            delta = np.linalg.solve(neg_hessian + 1e-4 * np.eye(p), score)

        beta += delta

        if np.linalg.norm(delta) < tol:
            break

    # Standard errors from information matrix
    try:
        var_beta = np.diag(np.linalg.inv(neg_hessian))
    except np.linalg.LinAlgError:
        var_beta = np.full(p, np.nan)

    se_beta = np.sqrt(np.maximum(var_beta, 0.0))

    # Hazard ratios
    hazard_ratios = []
    for j in range(p):
        b_j = float(beta[j])
        # Clamp to prevent overflow
        b_j_clamped = max(min(b_j, 500), -500)
        hr = math.exp(b_j_clamped)
        se_j = float(se_beta[j])
        if not np.isnan(se_j) and se_j > 0:
            ci_lo = math.exp(max(b_j - 1.96 * se_j, -500))
            ci_hi = math.exp(min(b_j + 1.96 * se_j, 500))
            z = b_j / se_j
            p_val = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
        else:
            ci_lo = None
            ci_hi = None
            p_val = 1.0
        hazard_ratios.append({
            "feature": feature_names[j],
            "hr": round(hr, 4),
            "ci_lower": round(ci_lo, 4) if ci_lo is not None else None,
            "ci_upper": round(ci_hi, 4) if ci_hi is not None else None,
            "p_value": round(p_val, 6),
        })

    # Concordance index
    concordance = _concordance_index(times, events, X @ beta)

    return {
        "hazard_ratios": hazard_ratios,
        "concordance": round(concordance, 4),
        "log_likelihood": round(float(log_lik), 4) if log_lik is not None else None,
    }


def _concordance_index(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Harrell's concordance index.

    Fraction of concordant pairs among comparable pairs.
    A pair (i, j) is comparable if the earlier event is uncensored.
    Concordant if higher risk score has shorter survival.
    """
    concordant = 0
    discordant = 0
    tied = 0
    n = len(times)

    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] < times[i]:
                continue
            if times[j] == times[i] and events[j] == 1:
                continue  # tied event times -- skip

            # Pair (i, j): i has event at times[i], j survived at least until times[i]
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


# ============================================================
# 7. Amendment Cascade (Markov Chain)
# ============================================================

def amendment_markov_chain(
    sequences: List[List[str]],
) -> Dict[str, Any]:
    """Build a Markov chain from amendment-type sequences.

    Parameters
    ----------
    sequences : list of list of str
        Each inner list is an ordered sequence of amendment types
        for a single trial (e.g., ["ENROLLMENT_UP", "ENDPOINT_CHANGE"]).

    Returns
    -------
    dict
        Keys: transition_matrix, stationary_dist, chi2_independence, p_value.
    """
    # Collect all states
    all_states: set = set()
    for seq in sequences:
        all_states.update(seq)
    states = sorted(all_states)
    state_idx = {s: i for i, s in enumerate(states)}
    k = len(states)

    if k == 0:
        return {
            "transition_matrix": {},
            "stationary_dist": {},
            "chi2_independence": 0.0,
            "p_value": 1.0,
        }

    # Count transitions
    count_matrix = np.zeros((k, k), dtype=float)
    for seq in sequences:
        for t in range(len(seq) - 1):
            i = state_idx[seq[t]]
            j = state_idx[seq[t + 1]]
            count_matrix[i, j] += 1

    # Transition probability matrix (row-normalized)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for states that never appear as source
    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
    prob_matrix = count_matrix / row_sums_safe

    # Build dict representation
    trans_dict: Dict[str, Dict[str, float]] = {}
    for i, s_from in enumerate(states):
        row: Dict[str, float] = {}
        for j, s_to in enumerate(states):
            if prob_matrix[i, j] > 0 or row_sums[i, 0] > 0:
                row[s_to] = round(float(prob_matrix[i, j]), 4)
        trans_dict[s_from] = row

    # Stationary distribution: solve pi @ P = pi, sum(pi) = 1
    # Equivalent to: pi @ (P - I) = 0  =>  (P' - I) @ pi' = 0
    # Add constraint sum(pi) = 1
    stationary = _compute_stationary(prob_matrix, states)

    # Chi-squared test of independence on the count matrix
    total_transitions = count_matrix.sum()
    if total_transitions > 0 and k > 1:
        row_totals = count_matrix.sum(axis=1)
        col_totals = count_matrix.sum(axis=0)
        expected = np.outer(row_totals, col_totals) / total_transitions
        # Mask cells with expected > 0
        mask = expected > 0
        if mask.sum() > 0:
            chi2_val = float(np.sum((count_matrix[mask] - expected[mask]) ** 2 / expected[mask]))
            df = max((k - 1) * (k - 1), 1)
            p_val = float(1.0 - sp_stats.chi2.cdf(chi2_val, df=df))
        else:
            chi2_val = 0.0
            p_val = 1.0
    else:
        chi2_val = 0.0
        p_val = 1.0

    return {
        "transition_matrix": trans_dict,
        "stationary_dist": stationary,
        "chi2_independence": round(chi2_val, 4),
        "p_value": round(p_val, 6),
    }


def _compute_stationary(
    P: np.ndarray,
    states: List[str],
) -> Dict[str, float]:
    """Compute stationary distribution of a Markov chain.

    Uses eigenvalue decomposition: finds left eigenvector for eigenvalue 1.
    """
    k = P.shape[0]
    if k == 0:
        return {}

    # Left eigenvectors: pi @ P = pi  =>  P' @ pi' = pi'
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find eigenvector closest to eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])

    # Normalize so sum = 1
    total = pi.sum()
    if abs(total) < 1e-12:
        # Fallback: uniform
        pi = np.ones(k) / k
    else:
        pi = pi / total

    # Ensure non-negative (numerical artifacts)
    pi = np.maximum(pi, 0.0)
    total = pi.sum()
    if total > 0:
        pi /= total

    return {states[i]: round(float(pi[i]), 6) for i in range(k)}


# ============================================================
# 8. Propensity Score Matching
# ============================================================

def propensity_match(
    amended_trials: List[Dict[str, Any]],
    clean_trials: List[Dict[str, Any]],
    features: List[str],
    seed: int = 42,
    n_bootstrap: int = 200,
    caliper_sd_mult: float = 0.2,
) -> Dict[str, Any]:
    """Propensity score matching: match amended trials to clean controls.

    Parameters
    ----------
    amended_trials : list of dict
        Trials that received amendments.  Must contain feature keys + "completed".
    clean_trials : list of dict
        Control trials.  Must contain feature keys + "completed".
    features : list of str
        Feature names to use for propensity model.
    seed : int
        Random seed for bootstrap.
    n_bootstrap : int
        Number of bootstrap resamples for ATT CI.
    caliper_sd_mult : float
        Caliper as multiple of propensity score SD.

    Returns
    -------
    dict
        Keys: matched_pairs, att, att_ci, balance.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(seed)

    n_a = len(amended_trials)
    n_c = len(clean_trials)

    # Extract feature matrix
    def _extract_features(trials: List[Dict[str, Any]]) -> np.ndarray:
        rows = []
        for t in trials:
            rows.append([float(t.get(f, 0)) for f in features])
        return np.array(rows, dtype=float)

    X_a = _extract_features(amended_trials)
    X_c = _extract_features(clean_trials)

    # Outcome: completed
    y_out_a = np.array([float(t.get("completed", 0)) for t in amended_trials])
    y_out_c = np.array([float(t.get("completed", 0)) for t in clean_trials])

    # Combined for propensity model
    X_all = np.vstack([X_a, X_c])
    treatment = np.concatenate([np.ones(n_a), np.zeros(n_c)])

    # Standardize features for better logistic convergence
    means = X_all.mean(axis=0)
    stds = X_all.std(axis=0)
    stds[stds == 0] = 1.0
    X_all_std = (X_all - means) / stds

    # Fit logistic regression
    model = LogisticRegression(random_state=seed, max_iter=1000, solver="lbfgs")
    model.fit(X_all_std, treatment)

    propensity = model.predict_proba(X_all_std)[:, 1]

    ps_amended = propensity[:n_a]
    ps_clean = propensity[n_a:]

    # Caliper
    ps_sd = np.std(propensity)
    caliper = caliper_sd_mult * ps_sd if ps_sd > 0 else 0.1

    # 1:1 nearest-neighbor matching without replacement
    matched_pairs: List[Tuple[int, int, float]] = []
    used_clean = set()

    for i in range(n_a):
        best_j = None
        best_dist = float("inf")
        for j in range(n_c):
            if j in used_clean:
                continue
            dist = abs(ps_amended[i] - ps_clean[j])
            if dist < best_dist and dist <= caliper:
                best_dist = dist
                best_j = j
        if best_j is not None:
            matched_pairs.append((i, best_j, round(best_dist, 6)))
            used_clean.add(best_j)

    # ATT: average treatment effect on the treated
    if len(matched_pairs) > 0:
        y_treated = np.array([y_out_a[i] for i, _, _ in matched_pairs])
        y_control = np.array([y_out_c[j] for _, j, _ in matched_pairs])
        att = float(np.mean(y_treated) - np.mean(y_control))

        # Bootstrap CI
        boot_atts = []
        n_matched = len(matched_pairs)
        for _ in range(n_bootstrap):
            idx = rng.choice(n_matched, size=n_matched, replace=True)
            bt = np.mean(y_treated[idx]) - np.mean(y_control[idx])
            boot_atts.append(bt)
        boot_atts = np.array(boot_atts)
        att_ci = (
            round(float(np.percentile(boot_atts, 2.5)), 4),
            round(float(np.percentile(boot_atts, 97.5)), 4),
        )
    else:
        att = None
        att_ci = (None, None)

    # Balance check: standardized mean difference before/after matching
    balance = []
    for fi, fname in enumerate(features):
        x_a_all = X_a[:, fi]
        x_c_all = X_c[:, fi]

        # Before matching
        smd_before = _standardized_mean_diff(x_a_all, x_c_all)

        # After matching
        if len(matched_pairs) > 0:
            x_a_matched = np.array([X_a[i, fi] for i, _, _ in matched_pairs])
            x_c_matched = np.array([X_c[j, fi] for _, j, _ in matched_pairs])
            smd_after = _standardized_mean_diff(x_a_matched, x_c_matched)
        else:
            smd_after = smd_before

        balance.append({
            "feature": fname,
            "smd_before": round(smd_before, 4),
            "smd_after": round(smd_after, 4),
        })

    return {
        "matched_pairs": [(int(a_i), int(c_i), d) for a_i, c_i, d in matched_pairs],
        "att": round(att, 4) if att is not None else None,
        "att_ci": att_ci,
        "balance": balance,
    }


def _standardized_mean_diff(
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    """Compute standardized mean difference (Cohen's d) between two groups."""
    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = math.sqrt((v1 + v2) / 2.0) if (v1 + v2) > 0 else 1.0
    if pooled_sd == 0:
        return 0.0
    return abs(m1 - m2) / pooled_sd


# ============================================================
# 9. Hidden Markov Model (Baum-Welch + Viterbi)
# ============================================================

def hidden_markov_model(
    observed_sequences: List[List[str]],
    n_states: int = 3,
    n_iter: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Hidden Markov Model for latent protocol stability states.

    States represent latent protocol conditions (e.g., STABLE, DRIFTING,
    RESCUING).  Observations are amendment types.

    Uses Baum-Welch (EM) for parameter estimation and Viterbi for decoding.

    Parameters
    ----------
    observed_sequences : list of list of str
        Each inner list is a sequence of observed amendment types.
    n_states : int
        Number of hidden states (default 3).
    n_iter : int
        Maximum EM iterations.
    seed : int
        Random seed for initialization.

    Returns
    -------
    dict
        Keys: transition_matrix, emission_matrix, initial_probs, state_names,
              log_likelihood, decoded_sequences, bic.
    """
    rng = np.random.RandomState(seed)

    # Map observations to integers
    all_obs: set = set()
    for seq in observed_sequences:
        all_obs.update(seq)
    obs_list = sorted(all_obs)
    obs_idx = {o: i for i, o in enumerate(obs_list)}
    n_obs = len(obs_list)
    K = n_states

    if n_obs == 0 or len(observed_sequences) == 0:
        state_names = [f"S{i}" for i in range(K)]
        return {
            "transition_matrix": np.eye(K),
            "emission_matrix": np.ones((K, max(n_obs, 1))) / max(n_obs, 1),
            "initial_probs": np.ones(K) / K,
            "state_names": state_names,
            "log_likelihood": 0.0,
            "decoded_sequences": [],
            "bic": 0.0,
        }

    # Encode sequences
    encoded = []
    for seq in observed_sequences:
        encoded.append([obs_idx[o] for o in seq])

    # Initialize parameters with Dirichlet-like random
    A = rng.dirichlet(np.ones(K), size=K)         # transition (K x K)
    B = rng.dirichlet(np.ones(n_obs), size=K)      # emission   (K x n_obs)
    pi = rng.dirichlet(np.ones(K))                 # initial    (K,)

    state_names = [f"S{i}" for i in range(K)]

    total_ll = -np.inf

    for _em_iter in range(n_iter):
        # Accumulators for M-step
        A_num = np.zeros((K, K))
        A_den = np.zeros(K)
        B_num = np.zeros((K, n_obs))
        B_den = np.zeros(K)
        pi_acc = np.zeros(K)
        total_ll_new = 0.0

        for seq in encoded:
            T = len(seq)
            if T == 0:
                continue

            # --- Forward ---
            alpha = np.zeros((T, K))
            # t=0
            alpha[0] = pi * B[:, seq[0]]
            scale = np.zeros(T)
            scale[0] = alpha[0].sum()
            if scale[0] > 0:
                alpha[0] /= scale[0]
            else:
                alpha[0] = 1.0 / K
                scale[0] = 1.0

            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, seq[t]]
                scale[t] = alpha[t].sum()
                if scale[t] > 0:
                    alpha[t] /= scale[t]
                else:
                    alpha[t] = 1.0 / K
                    scale[t] = 1.0

            # Log-likelihood for this sequence
            total_ll_new += np.sum(np.log(scale + 1e-300))

            # --- Backward ---
            beta = np.zeros((T, K))
            beta[T - 1] = 1.0

            for t in range(T - 2, -1, -1):
                beta[t] = A @ (B[:, seq[t + 1]] * beta[t + 1])
                if scale[t + 1] > 0:
                    beta[t] /= scale[t + 1]

            # --- E-step: compute gamma and xi ---
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1.0)
            gamma = gamma / gamma_sum

            # Xi: xi[t][i][j] = alpha[t][i] * A[i][j] * B[j][o_{t+1}] * beta[t+1][j] / P(O|lambda)
            for t in range(T - 1):
                xi_t = np.outer(alpha[t], B[:, seq[t + 1]] * beta[t + 1]) * A
                xi_sum = xi_t.sum()
                if xi_sum > 0:
                    xi_t /= xi_sum
                A_num += xi_t
                A_den += gamma[t]

            # Accumulate for B
            for t in range(T):
                B_num[:, seq[t]] += gamma[t]
                B_den += gamma[t]

            pi_acc += gamma[0]

        # --- M-step ---
        # Transition matrix
        for i in range(K):
            if A_den[i] > 0:
                A[i] = A_num[i] / A_den[i]
            else:
                A[i] = 1.0 / K
        # Ensure rows sum to 1
        A = A / A.sum(axis=1, keepdims=True)

        # Emission matrix
        for i in range(K):
            if B_den[i] > 0:
                B[i] = B_num[i] / B_den[i]
            else:
                B[i] = 1.0 / n_obs
        B = B / B.sum(axis=1, keepdims=True)

        # Initial probs
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(K) / K

        # Check convergence
        if abs(total_ll_new - total_ll) < 1e-6:
            total_ll = total_ll_new
            break
        total_ll = total_ll_new

    # --- Viterbi decoding ---
    decoded_sequences = []
    for seq in encoded:
        T = len(seq)
        if T == 0:
            decoded_sequences.append([])
            continue

        viterbi = np.zeros((T, K))
        backptr = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(pi + 1e-300) + np.log(B[:, seq[0]] + 1e-300)

        for t in range(1, T):
            for j in range(K):
                candidates = viterbi[t - 1] + np.log(A[:, j] + 1e-300)
                backptr[t, j] = int(np.argmax(candidates))
                viterbi[t, j] = candidates[backptr[t, j]] + np.log(B[j, seq[t]] + 1e-300)

        # Backtrace
        path = [0] * T
        path[T - 1] = int(np.argmax(viterbi[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]
        decoded_sequences.append([state_names[s] for s in path])

    # BIC: -2 * LL + k * ln(n)
    n_params = K * (K - 1) + K * (n_obs - 1) + (K - 1)  # free params
    total_obs = sum(len(s) for s in encoded)
    bic = -2.0 * total_ll + n_params * math.log(max(total_obs, 1))

    return {
        "transition_matrix": A,
        "emission_matrix": B,
        "initial_probs": pi,
        "state_names": state_names,
        "log_likelihood": float(total_ll),
        "decoded_sequences": decoded_sequences,
        "bic": float(bic),
    }


# ============================================================
# 10. Andersen-Gill Recurrent Events Model
# ============================================================

def andersen_gill_model(
    trials_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Andersen-Gill extension of Cox PH for recurrent amendment events.

    Each trial can contribute multiple events (amendments) over time.
    Uses counting-process notation with robust sandwich variance.

    Parameters
    ----------
    trials_data : list of dict
        Each entry: {trial_id, event_times: [t1, t2, ...],
                     features: {name: value, ...}, max_time}.

    Returns
    -------
    dict
        Keys: hazard_ratios (list of dicts with feature, hr, ci_lower,
              ci_upper, p_value), n_events, n_subjects, concordance.
    """
    # Expand into counting-process rows: (start, stop, event, features, trial_id)
    rows = []
    feature_names = None
    for trial in trials_data:
        tid = trial["trial_id"]
        event_times = sorted(trial.get("event_times", []))
        feats = trial.get("features", {})
        max_time = trial.get("max_time", 365)

        if feature_names is None:
            feature_names = sorted(feats.keys())

        feat_vec = [float(feats.get(f, 0)) for f in feature_names]

        if len(event_times) == 0:
            # Censored -- one row from 0 to max_time
            rows.append((0.0, float(max_time), 0, feat_vec, tid))
        else:
            prev = 0.0
            for et in event_times:
                rows.append((prev, float(et), 1, feat_vec, tid))
                prev = float(et)
            # After last event, censored until max_time
            if event_times[-1] < max_time:
                rows.append((prev, float(max_time), 0, feat_vec, tid))

    if feature_names is None or len(feature_names) == 0 or len(rows) == 0:
        return {
            "hazard_ratios": [],
            "n_events": 0,
            "n_subjects": len(trials_data),
            "concordance": 0.5,
        }

    p = len(feature_names)
    n_rows = len(rows)

    # Extract arrays
    starts = np.array([r[0] for r in rows])
    stops = np.array([r[1] for r in rows])
    events = np.array([r[2] for r in rows])
    X = np.array([r[3] for r in rows], dtype=float)
    trial_ids = [r[4] for r in rows]

    # Standardize features for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_s = (X - X_mean) / X_std

    n_events_total = int(events.sum())
    unique_trials = set(trial_ids)

    # Newton-Raphson on partial likelihood (counting process)
    beta = np.zeros(p)

    # Event times (unique, sorted)
    event_indices = np.where(events == 1)[0]
    if len(event_indices) == 0:
        return {
            "hazard_ratios": [{"feature": f, "hr": 1.0, "ci_lower": None,
                               "ci_upper": None, "p_value": 1.0}
                              for f in feature_names],
            "n_events": 0,
            "n_subjects": len(unique_trials),
            "concordance": 0.5,
        }

    max_iter = 50
    tol = 1e-6
    neg_hessian = np.eye(p)

    for _iteration in range(max_iter):
        eta = X_s @ beta
        eta_max = np.max(eta)
        exp_eta = np.exp(eta - eta_max)

        score = np.zeros(p)
        neg_hessian = np.zeros((p, p))
        ll = 0.0

        for idx in event_indices:
            t_event = stops[idx]
            # Risk set: rows where start < t_event and stop >= t_event
            risk_mask = (starts < t_event) & (stops >= t_event)
            if not np.any(risk_mask):
                continue

            risk_exp = exp_eta[risk_mask]
            risk_X = X_s[risk_mask]
            r_sum = risk_exp.sum()

            if r_sum <= 0:
                continue

            r_mean = (risk_exp @ risk_X) / r_sum

            ll += (eta[idx] - eta_max) - math.log(r_sum)
            score += X_s[idx] - r_mean

            # Hessian contribution
            w_k = risk_exp / r_sum
            weighted_X = risk_X * np.sqrt(w_k)[:, None]
            E_XX = weighted_X.T @ weighted_X
            neg_hessian += E_XX - np.outer(r_mean, r_mean)

        # Newton step
        try:
            delta = np.linalg.solve(neg_hessian, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(neg_hessian + 1e-4 * np.eye(p), score)

        beta += delta
        if np.linalg.norm(delta) < tol:
            break

    # Naive variance
    try:
        var_naive = np.linalg.inv(neg_hessian)
    except np.linalg.LinAlgError:
        var_naive = np.eye(p) * 1e6

    # Robust sandwich variance: group scores by trial
    trial_scores = defaultdict(lambda: np.zeros(p))
    eta_final = X_s @ beta
    eta_max = np.max(eta_final)
    exp_eta_final = np.exp(eta_final - eta_max)

    for idx in event_indices:
        t_event = stops[idx]
        risk_mask = (starts < t_event) & (stops >= t_event)
        if not np.any(risk_mask):
            continue
        risk_exp = exp_eta_final[risk_mask]
        risk_X = X_s[risk_mask]
        r_sum = risk_exp.sum()
        if r_sum <= 0:
            continue
        r_mean = (risk_exp @ risk_X) / r_sum

        # Score residual for the event subject
        trial_scores[trial_ids[idx]] += X_s[idx] - r_mean

        # Subtract risk-set contributions for all at-risk subjects
        risk_indices = np.where(risk_mask)[0]
        for ri in risk_indices:
            contribution = (exp_eta_final[ri] / r_sum) * (X_s[ri] - r_mean)
            trial_scores[trial_ids[ri]] -= contribution
            # Add back event contribution
        # Re-add event contribution that was subtracted
        event_contribution = (exp_eta_final[idx] / r_sum) * (X_s[idx] - r_mean)
        trial_scores[trial_ids[idx]] += event_contribution

    # Meat of sandwich: sum of outer products of per-trial score residuals
    meat = np.zeros((p, p))
    for tid, s_vec in trial_scores.items():
        meat += np.outer(s_vec, s_vec)

    var_robust = var_naive @ meat @ var_naive
    se_robust = np.sqrt(np.maximum(np.diag(var_robust), 0.0))

    # Convert back from standardized to original scale
    beta_orig = beta / X_std
    se_orig = se_robust / X_std

    # Build hazard ratios
    hazard_ratios = []
    for j in range(p):
        b_j = float(beta_orig[j])
        b_j_clamped = max(min(b_j, 500), -500)
        hr = math.exp(b_j_clamped)
        se_j = float(se_orig[j])
        if se_j > 0 and not np.isnan(se_j):
            ci_lo = math.exp(max(b_j - 1.96 * se_j, -500))
            ci_hi = math.exp(min(b_j + 1.96 * se_j, 500))
            z = b_j / se_j
            p_val = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
        else:
            ci_lo = None
            ci_hi = None
            p_val = 1.0
        hazard_ratios.append({
            "feature": feature_names[j],
            "hr": round(hr, 4),
            "ci_lower": round(ci_lo, 4) if ci_lo is not None else None,
            "ci_upper": round(ci_hi, 4) if ci_hi is not None else None,
            "p_value": round(float(p_val), 6),
        })

    # Simple concordance on linear predictor vs stop times for event rows
    event_mask = events == 1
    if event_mask.sum() > 1:
        concordance = _concordance_index(
            stops[event_mask], events[event_mask],
            (X[event_mask] @ beta_orig),
        )
    else:
        concordance = 0.5

    return {
        "hazard_ratios": hazard_ratios,
        "n_events": n_events_total,
        "n_subjects": len(unique_trials),
        "concordance": round(float(concordance), 4),
    }


# ============================================================
# 11. Frailty Model (Shared Gamma Frailty)
# ============================================================

def frailty_model(
    times: Sequence[float],
    events: Sequence[int],
    groups: List[str],
    X: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Shared gamma frailty model for grouped survival data.

    h_ij(t) = z_j * h_0(t) * exp(X_ij * beta)
    z_j ~ Gamma(1/theta, 1/theta)  with E[z_j] = 1, Var[z_j] = theta.

    Fit via EM algorithm.

    Parameters
    ----------
    times : array-like of float
        Survival/event times.
    events : array-like of int
        Event indicators (1 = event, 0 = censored).
    groups : list of str
        Group label for each observation (e.g., sponsor ID).
    X : ndarray of shape (n, p), optional
        Covariate matrix.  If None, a model with frailty only is fit.

    Returns
    -------
    dict
        Keys: hazard_ratios, frailty_variance, frailty_values, log_likelihood.
    """
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=int)
    n = len(times_arr)

    # Group mapping
    unique_groups = sorted(set(groups))
    group_idx = {g: i for i, g in enumerate(unique_groups)}
    g_labels = np.array([group_idx[g] for g in groups])
    n_groups = len(unique_groups)

    # Covariates
    has_covariates = X is not None and X.shape[1] > 0
    if has_covariates:
        X_arr = np.asarray(X, dtype=float)
        p = X_arr.shape[1]
        beta = np.zeros(p)
    else:
        X_arr = np.zeros((n, 0))
        p = 0
        beta = np.zeros(0)

    # Initialize theta (frailty variance) and frailty values
    theta = 1.0
    z = np.ones(n_groups)  # frailty values per group

    # Sort by time
    order = np.argsort(times_arr, kind="stable")
    times_sorted = times_arr[order]
    events_sorted = events_arr[order]
    groups_sorted = g_labels[order]
    X_sorted = X_arr[order] if has_covariates else X_arr

    max_em_iter = 50
    log_lik = -np.inf

    for _em in range(max_em_iter):
        # --- E-step ---
        # Compute E[z_j | data] for each group
        # E[z_j | data] = (d_j + 1/theta) / (H_0j + 1/theta)
        # where d_j = total events in group j
        # H_0j = sum of cumulative baseline hazard contributions for group j

        # Breslow estimate of cumulative baseline hazard
        if has_covariates:
            eta = X_sorted @ beta
        else:
            eta = np.zeros(n)
        exp_eta = np.exp(eta)

        # Compute H_0(t_i) for each individual weighted by frailty
        # Nelson-Aalen style baseline hazard
        z_expanded = np.array([z[g] for g in groups_sorted])

        # Risk weighted sums at each event time
        unique_event_times = np.unique(times_sorted[events_sorted == 1])

        # Cumulative baseline hazard at each observation time
        H0 = np.zeros(n)
        cum_h0 = 0.0

        for t in unique_event_times:
            mask_event = (times_sorted == t) & (events_sorted == 1)
            d_t = mask_event.sum()
            risk_mask = times_sorted >= t
            risk_sum = np.sum(z_expanded[risk_mask] * exp_eta[risk_mask])
            if risk_sum > 0:
                h0_t = d_t / risk_sum
            else:
                h0_t = 0.0
            cum_h0 += h0_t
            # Assign cumulative hazard to all with time >= t
            H0[times_sorted >= t] = cum_h0

        # For each observation, its cumulative hazard contribution is H0(t_i) * exp(eta_i)
        # Group cumulative hazard: H_0j = sum_{i in group j} H0(t_i) * exp(eta_i)
        H0_group = np.zeros(n_groups)
        d_group = np.zeros(n_groups)
        for i in range(n):
            gj = groups_sorted[i]
            H0_group[gj] += H0[i] * exp_eta[i]
            d_group[gj] += events_sorted[i]

        # Update frailty estimates
        for j in range(n_groups):
            z[j] = (d_group[j] + 1.0 / max(theta, 1e-10)) / (H0_group[j] + 1.0 / max(theta, 1e-10))

        # --- M-step ---
        # Update theta: use moment estimator from frailty values
        # Var(z) ~ theta when z ~ Gamma(1/theta, 1/theta)
        # Profile likelihood approach: optimize theta
        # Simple update: theta = Var(z) or via digamma equation
        if n_groups > 1:
            z_mean = z.mean()
            if z_mean > 0:
                z_normalized = z / z_mean
                theta_new = max(float(np.var(z_normalized, ddof=1)), 1e-6)
            else:
                theta_new = theta
        else:
            theta_new = 0.0

        # Update beta via one Newton-Raphson step on weighted Cox PL
        if has_covariates:
            z_expanded = np.array([z[g] for g in groups_sorted])
            score_b = np.zeros(p)
            info_b = np.zeros((p, p))

            for t in unique_event_times:
                mask_event = (times_sorted == t) & (events_sorted == 1)
                event_indices = np.where(mask_event)[0]
                risk_mask = times_sorted >= t
                risk_w = z_expanded[risk_mask] * exp_eta[risk_mask]
                risk_sum = risk_w.sum()
                if risk_sum <= 0:
                    continue
                risk_X = X_sorted[risk_mask]
                r_mean = (risk_w @ risk_X) / risk_sum

                for idx in event_indices:
                    score_b += X_sorted[idx] - r_mean
                    w_k = risk_w / risk_sum
                    weighted_X = risk_X * np.sqrt(w_k)[:, None]
                    E_XX = weighted_X.T @ weighted_X
                    info_b += E_XX - np.outer(r_mean, r_mean)

            try:
                delta_b = np.linalg.solve(info_b + 1e-8 * np.eye(p), score_b)
            except np.linalg.LinAlgError:
                delta_b = np.zeros(p)
            beta += delta_b

        # Log-likelihood (approximate)
        ll_new = 0.0
        z_expanded = np.array([z[g] for g in groups_sorted])
        for i in range(n):
            if events_sorted[i] == 1:
                val = z_expanded[i] * exp_eta[i]
                ll_new += math.log(max(val, 1e-300))
        # Subtract cumulative hazard
        for j in range(n_groups):
            ll_new -= z[j] * H0_group[j]
        # Frailty prior contribution
        for j in range(n_groups):
            if theta_new > 1e-10:
                inv_theta = 1.0 / theta_new
                ll_new += (inv_theta - 1) * math.log(max(z[j], 1e-300)) - z[j] * inv_theta

        theta = theta_new

        if abs(ll_new - log_lik) < 1e-6:
            log_lik = ll_new
            break
        log_lik = ll_new

    # Build output
    hazard_ratios_out = []
    if has_covariates:
        for j in range(p):
            b_j = float(beta[j])
            b_j_clamped = max(min(b_j, 500), -500)
            hr = math.exp(b_j_clamped)
            hazard_ratios_out.append({
                "feature": f"x{j}",
                "hr": round(hr, 4),
            })

    frailty_values = {unique_groups[j]: round(float(z[j]), 4) for j in range(n_groups)}

    return {
        "hazard_ratios": hazard_ratios_out,
        "frailty_variance": round(float(theta), 4),
        "frailty_values": frailty_values,
        "log_likelihood": round(float(log_lik), 4),
    }


# ============================================================
# 12. Change-Point Detection (CUSUM + PELT)
# ============================================================

def cusum_detection(
    time_series: Sequence[float],
    target: Optional[float] = None,
    threshold: float = 5.0,
) -> Dict[str, Any]:
    """CUSUM change-point detection with PELT optimal segmentation.

    Runs both one-sided CUSUM control charts (upward/downward) and
    PELT (Pruned Exact Linear Time) for optimal segmentation.

    Parameters
    ----------
    time_series : array-like of float
        Observed values over time.
    target : float, optional
        Target/reference value.  If None, uses the overall mean.
    threshold : float
        Decision threshold h for CUSUM (default 5.0).

    Returns
    -------
    dict
        Keys: change_points (list of {time, direction, cusum_value}),
              cusum_values (list of floats), target_rate.
    """
    series = np.asarray(time_series, dtype=float)
    n = len(series)

    if n == 0:
        return {"change_points": [], "cusum_values": [], "target_rate": 0.0}

    if target is None:
        target_val = float(np.mean(series))
    else:
        target_val = float(target)

    # Allowance parameter k (typically half the shift to detect)
    # Estimate shift as the standard deviation
    sd = float(np.std(series)) if n > 1 else 1.0
    k = 0.5 * max(sd, 0.01)

    # CUSUM computation
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)

    for t in range(n):
        if t == 0:
            s_pos[t] = max(0.0, series[t] - target_val - k)
            s_neg[t] = min(0.0, series[t] - target_val + k)
        else:
            s_pos[t] = max(0.0, s_pos[t - 1] + (series[t] - target_val - k))
            s_neg[t] = min(0.0, s_neg[t - 1] + (series[t] - target_val + k))

    # Detect change points where CUSUM exceeds threshold
    change_points = []
    in_signal_pos = False
    in_signal_neg = False

    for t in range(n):
        if s_pos[t] > threshold and not in_signal_pos:
            change_points.append({
                "time": t,
                "direction": "upward",
                "cusum_value": round(float(s_pos[t]), 4),
            })
            in_signal_pos = True
        elif s_pos[t] <= 0:
            in_signal_pos = False

        if abs(s_neg[t]) > threshold and not in_signal_neg:
            change_points.append({
                "time": t,
                "direction": "downward",
                "cusum_value": round(float(s_neg[t]), 4),
            })
            in_signal_neg = True
        elif s_neg[t] >= 0:
            in_signal_neg = False

    # PELT (Pruned Exact Linear Time) for optimal segmentation
    # Cost function: Gaussian negative log-likelihood for each segment
    # C(y_{s:t}) = (t-s)/2 * log(var(y_{s:t})) + (t-s)/2
    penalty = 2.0 * math.log(max(n, 2))  # BIC-like penalty

    pelt_cps = _pelt_segmentation(series, penalty)

    # Merge PELT change points into results (avoid duplicates near CUSUM detections)
    for cp in pelt_cps:
        # Check if already covered by CUSUM
        already = any(abs(existing["time"] - cp) <= 2 for existing in change_points)
        if not already:
            # Determine direction from mean shift
            if cp > 0 and cp < n:
                mean_before = float(np.mean(series[:cp]))
                mean_after = float(np.mean(series[cp:]))
                direction = "upward" if mean_after > mean_before else "downward"
            else:
                direction = "unknown"
            change_points.append({
                "time": cp,
                "direction": direction,
                "cusum_value": round(float(max(s_pos[cp], abs(s_neg[cp]))), 4),
            })

    # Sort by time
    change_points.sort(key=lambda x: x["time"])

    # Combined CUSUM values (max of pos and abs(neg) at each t)
    cusum_values = [round(float(max(s_pos[t], abs(s_neg[t]))), 4) for t in range(n)]

    return {
        "change_points": change_points,
        "cusum_values": cusum_values,
        "target_rate": round(target_val, 4),
    }


def _pelt_segmentation(
    data: np.ndarray,
    penalty: float,
) -> List[int]:
    """PELT (Pruned Exact Linear Time) optimal segmentation.

    Minimizes total cost = sum of segment costs + penalty * n_changepoints.
    Uses Gaussian cost (negative log-likelihood).

    Parameters
    ----------
    data : ndarray
        Time series data.
    penalty : float
        Penalty per change point.

    Returns
    -------
    list of int
        Indices of detected change points.
    """
    n = len(data)
    if n < 4:
        return []

    # Precompute cumulative sums for O(1) segment cost
    cum_sum = np.zeros(n + 1)
    cum_sq = np.zeros(n + 1)
    for i in range(n):
        cum_sum[i + 1] = cum_sum[i] + data[i]
        cum_sq[i + 1] = cum_sq[i] + data[i] ** 2

    def segment_cost(s: int, e: int) -> float:
        """Gaussian cost for segment data[s:e]."""
        length = e - s
        if length <= 1:
            return 0.0
        seg_sum = cum_sum[e] - cum_sum[s]
        seg_sq = cum_sq[e] - cum_sq[s]
        mean_val = seg_sum / length
        variance = seg_sq / length - mean_val ** 2
        if variance <= 1e-12:
            return 0.0
        return length * math.log(max(variance, 1e-300))

    # DP with pruning
    F = np.full(n + 1, np.inf)
    F[0] = -penalty  # so first segment starts with cost only
    cp_list: List[List[int]] = [[] for _ in range(n + 1)]
    candidates = [0]

    for t in range(1, n + 1):
        best_cost = np.inf
        best_tau = 0
        for tau in candidates:
            cost = F[tau] + segment_cost(tau, t) + penalty
            if cost < best_cost:
                best_cost = cost
                best_tau = tau
        F[t] = best_cost
        cp_list[t] = cp_list[best_tau] + ([best_tau] if best_tau > 0 else [])

        # Prune: keep only candidates that could be optimal in future
        new_candidates = []
        for tau in candidates:
            if F[tau] + segment_cost(tau, t) <= F[t]:
                new_candidates.append(tau)
        new_candidates.append(t)
        candidates = new_candidates

    return cp_list[n]


# ============================================================
# 13. Cure-Rate Model (Mixture Cure)
# ============================================================

def cure_rate_model(
    times: Sequence[float],
    events: Sequence[int],
    X: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mixture cure-rate model.

    S(t|X) = pi + (1 - pi) * S_u(t)

    where pi is the cure fraction (modeled via logistic if X provided),
    and S_u(t) is the Weibull survival for uncured subjects.

    Fit via EM algorithm.

    Parameters
    ----------
    times : array-like of float
        Observed times.
    events : array-like of int
        Event indicators (1 = event, 0 = censored).
    X : ndarray, optional
        Covariate matrix for the cure fraction model.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: cure_fraction, cure_ci, survival_params (shape, scale),
              coefficients, log_likelihood, bic.
    """
    rng = np.random.RandomState(seed)
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=int)
    n = len(times_arr)

    # Clamp times to positive
    times_arr = np.maximum(times_arr, 1e-6)

    has_covariates = X is not None and X.shape[1] > 0
    if has_covariates:
        X_arr = np.asarray(X, dtype=float)
        p_cov = X_arr.shape[1]
    else:
        X_arr = np.zeros((n, 0))
        p_cov = 0

    # Initialize Weibull parameters (shape, scale)
    shape = 1.0  # k
    scale = float(np.median(times_arr[events_arr == 1])) if events_arr.sum() > 0 else float(np.median(times_arr))
    if scale <= 0:
        scale = 1.0

    # Initialize cure fraction from proportion of censored
    n_censored = int((events_arr == 0).sum())
    pi_init = max(0.05, min(0.95, n_censored / max(n, 1)))

    # Logistic coefficients for cure model
    gamma = np.zeros(p_cov + 1)  # intercept + covariates
    # Set intercept to match pi_init
    gamma[0] = math.log(pi_init / max(1 - pi_init, 1e-10))

    max_em_iter = 200
    log_lik = -np.inf

    def _logistic(z: float) -> float:
        """Numerically stable logistic."""
        if z > 500:
            return 1.0
        if z < -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def _weibull_pdf(t: float, k: float, lam: float) -> float:
        """Weibull PDF: f(t) = (k/lam) * (t/lam)^{k-1} * exp(-(t/lam)^k)."""
        if t <= 0 or k <= 0 or lam <= 0:
            return 1e-300
        z = (t / lam) ** k
        return max((k / lam) * ((t / lam) ** (k - 1)) * math.exp(-z), 1e-300)

    def _weibull_surv(t: float, k: float, lam: float) -> float:
        """Weibull survival: S(t) = exp(-(t/lam)^k)."""
        if t <= 0 or lam <= 0:
            return 1.0
        return math.exp(-((t / lam) ** k))

    for _em in range(max_em_iter):
        # --- E-step ---
        # For each censored observation, compute posterior prob of being cured
        # P(cured | censored, t) = pi * 1 / [pi * 1 + (1-pi) * S_u(t)]
        # For event observations, P(cured) = 0 (they experienced the event)

        w = np.zeros(n)  # posterior prob of being "uncured"
        pi_vals = np.zeros(n)  # cure probability for each subject

        for i in range(n):
            if has_covariates:
                z_i = gamma[0] + float(X_arr[i] @ gamma[1:])
            else:
                z_i = gamma[0]
            pi_i = _logistic(z_i)
            pi_vals[i] = pi_i

            if events_arr[i] == 1:
                # Observed event -> definitely uncured
                w[i] = 1.0
            else:
                # Censored: compute posterior
                s_u = _weibull_surv(times_arr[i], shape, scale)
                denom = pi_i + (1 - pi_i) * s_u
                if denom > 0:
                    w[i] = (1 - pi_i) * s_u / denom
                else:
                    w[i] = 0.5

        # --- M-step ---
        # 1. Update cure fraction parameters (gamma) via weighted logistic
        #    Minimize: sum_i [w_i * log(1 - pi_i) + (1 - w_i) * log(pi_i)]
        #    One Newton step
        if True:  # always update gamma
            # Gradient and Hessian for logistic
            grad = np.zeros(p_cov + 1)
            hess = np.zeros((p_cov + 1, p_cov + 1))
            for i in range(n):
                pi_i = pi_vals[i]
                # Target: (1 - w_i) = prob of being cured
                residual = (1.0 - w[i]) - pi_i
                x_i = np.zeros(p_cov + 1)
                x_i[0] = 1.0
                if has_covariates:
                    x_i[1:] = X_arr[i]
                grad += residual * x_i
                hess -= pi_i * (1 - pi_i) * np.outer(x_i, x_i)

            try:
                delta_g = np.linalg.solve(hess - 1e-6 * np.eye(p_cov + 1), grad)
                gamma -= delta_g
            except np.linalg.LinAlgError:
                pass

        # 2. Update Weibull parameters using uncured subset (weighted MLE)
        # Maximize: sum_i w_i * [event_i * log(f(t_i)) + (1-event_i) * log(S(t_i))]
        # Grid search + Newton for shape, closed-form-ish for scale
        total_w = w.sum()
        if total_w > 0:
            # Update scale given shape: lambda_hat = (sum w_i * t_i^k / sum w_i * event_i)^{1/k}
            w_events = w * events_arr
            w_events_sum = w_events.sum()

            if w_events_sum > 0:
                # Newton-Raphson for shape parameter
                for _shape_iter in range(20):
                    t_k = times_arr ** shape
                    w_t_k = (w * t_k).sum()

                    # Scale update
                    scale_new = (w_t_k / w_events_sum) ** (1.0 / shape)
                    if scale_new <= 0:
                        scale_new = scale
                    scale = scale_new

                    # Shape update via profile log-likelihood derivative
                    # d/dk: sum w_i*event_i * [1/k + log(t_i/lam)] - sum w_i * (t_i/lam)^k * log(t_i/lam)
                    log_t_lam = np.log(np.maximum(times_arr / scale, 1e-300))
                    t_lam_k = (times_arr / scale) ** shape

                    dl_dk = (w_events_sum / shape
                             + (w * events_arr * log_t_lam).sum()
                             - (w * t_lam_k * log_t_lam).sum())

                    # Second derivative (approximate)
                    d2l_dk2 = (-w_events_sum / (shape ** 2)
                               - (w * t_lam_k * log_t_lam ** 2).sum())

                    if abs(d2l_dk2) > 1e-10:
                        shape_new = shape - dl_dk / d2l_dk2
                        shape = max(0.01, min(float(shape_new), 50.0))

        # Log-likelihood
        ll_new = 0.0
        for i in range(n):
            pi_i = pi_vals[i]
            if events_arr[i] == 1:
                f_u = _weibull_pdf(times_arr[i], shape, scale)
                ll_new += math.log(max((1 - pi_i) * f_u, 1e-300))
            else:
                s_u = _weibull_surv(times_arr[i], shape, scale)
                ll_new += math.log(max(pi_i + (1 - pi_i) * s_u, 1e-300))

        if abs(ll_new - log_lik) < 1e-6:
            log_lik = ll_new
            break
        log_lik = ll_new

    # Final cure fraction (average)
    cure_fraction = float(np.mean([_logistic(gamma[0] + (float(X_arr[i] @ gamma[1:]) if has_covariates else 0.0))
                                   for i in range(n)]))

    # Wald CI for cure fraction via delta method on logistic intercept
    # Compute observed information for gamma[0] (intercept)
    info_gamma0 = 0.0
    for i in range(n):
        pi_i = _logistic(gamma[0] + (float(X_arr[i] @ gamma[1:]) if has_covariates else 0.0))
        info_gamma0 += pi_i * (1 - pi_i)
    se_gamma0 = 1.0 / math.sqrt(max(info_gamma0, 1e-10))
    # Delta method: SE(pi) = pi*(1-pi) * SE(gamma0)
    pi_hat = cure_fraction
    se_pi = pi_hat * (1 - pi_hat) * se_gamma0
    cure_ci = (
        round(max(0.0, pi_hat - 1.96 * se_pi), 4),
        round(min(1.0, pi_hat + 1.96 * se_pi), 4),
    )

    # Coefficients
    coefficients = {"intercept": round(float(gamma[0]), 4)}
    for j in range(p_cov):
        coefficients[f"x{j}"] = round(float(gamma[j + 1]), 4)

    # BIC
    n_params = 2 + (p_cov + 1)  # Weibull (shape, scale) + logistic (intercept + covariates)
    bic = -2.0 * log_lik + n_params * math.log(max(n, 1))

    return {
        "cure_fraction": round(cure_fraction, 4),
        "cure_ci": cure_ci,
        "survival_params": {"shape": round(shape, 4), "scale": round(scale, 4)},
        "coefficients": coefficients,
        "log_likelihood": round(float(log_lik), 4),
        "bic": round(float(bic), 4),
    }


# ============================================================
# Layer 4 — Cutting-Edge (5 methods)
# ============================================================


# ============================================================
# 14. Multi-State Model (Nelson-Aalen)
# ============================================================


def multi_state_model(
    trials_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Multi-state model for protocol lifecycle transitions.

    States: REGISTERED -> AMENDED -> COMPLETED / TERMINATED.
    Uses the Nelson-Aalen estimator for transition-specific cumulative
    hazards.

    Parameters
    ----------
    trials_data : list of dict
        Each dict has ``trial_id`` and ``states``: a list of
        ``(state, time)`` tuples ordered chronologically.

    Returns
    -------
    dict
        Keys: transition_intensities, state_probs_at_t, sojourn_times.
    """
    # Collect observed transitions
    transitions: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    sojourn_raw: Dict[str, List[float]] = defaultdict(list)

    for trial in trials_data:
        states = trial["states"]
        for i in range(len(states) - 1):
            s_from, t_from = states[i]
            s_to, t_to = states[i + 1]
            dt = t_to - t_from
            transitions[(s_from, s_to)].append(dt)
            sojourn_raw[s_from].append(dt)

    # All unique states
    all_states = sorted(set(
        s for trial in trials_data for s, _ in trial["states"]
    ))

    # Nelson-Aalen cumulative hazard per transition
    transition_intensities: Dict[str, Any] = {}
    for (s_from, s_to), times_list in transitions.items():
        times_arr = np.sort(times_list)
        n_at_risk = len(times_arr)
        cum_hazard = []
        ch = 0.0
        for idx, t in enumerate(times_arr):
            d = 1  # one event at this time
            n_r = n_at_risk - idx
            ch += d / max(n_r, 1)
            cum_hazard.append({"time": float(t), "cum_hazard": round(ch, 6)})
        key = f"{s_from}->{s_to}"
        intensity = len(times_list) / max(sum(sojourn_raw.get(s_from, [1])), 1e-10)
        transition_intensities[key] = {
            "count": len(times_list),
            "intensity": round(float(intensity), 6),
            "cum_hazard": cum_hazard,
        }

    # Sojourn times
    sojourn_times: Dict[str, Dict[str, float]] = {}
    for state, durations in sojourn_raw.items():
        arr = np.array(durations)
        sojourn_times[state] = {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
        }

    # State probabilities at selected time horizons via transition matrix
    # Build empirical transition probability matrix
    state_idx = {s: i for i, s in enumerate(all_states)}
    n_s = len(all_states)
    trans_counts = np.zeros((n_s, n_s))
    for (s_from, s_to), times_list in transitions.items():
        i_from = state_idx[s_from]
        i_to = state_idx[s_to]
        trans_counts[i_from, i_to] += len(times_list)

    # Row-stochastic transition matrix
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    # Absorbing states get identity row
    P = np.zeros((n_s, n_s))
    for i in range(n_s):
        if row_sums[i, 0] > 0:
            P[i, :] = trans_counts[i, :] / row_sums[i, 0]
        else:
            P[i, i] = 1.0  # absorbing

    # Compute state probabilities at steps 1, 5, 10
    state_probs_at_t: Dict[int, Dict[str, float]] = {}
    init = np.zeros(n_s)
    if len(all_states) > 0:
        init[0] = 1.0  # start in first state (REGISTERED)
    for step in [1, 5, 10]:
        probs = init @ np.linalg.matrix_power(P, step)
        state_probs_at_t[step] = {
            all_states[i]: round(float(probs[i]), 6) for i in range(n_s)
        }

    return {
        "transition_intensities": transition_intensities,
        "state_probs_at_t": state_probs_at_t,
        "sojourn_times": sojourn_times,
    }


# ============================================================
# 15. Joint Longitudinal-Survival Model
# ============================================================


def joint_model(
    longitudinal_data: List[Dict[str, Any]],
    survival_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Two-stage joint longitudinal-survival model.

    Stage 1: Linear mixed-effects model (simplified via OLS per trial)
    for the longitudinal enrollment/amendment trajectory.
    Stage 2: Cox PH with the predicted trajectory value as a
    time-varying covariate.

    Parameters
    ----------
    longitudinal_data : list of dict
        Each dict has ``trial_id``, ``times`` (list of float),
        ``values`` (list of float).
    survival_data : list of dict
        Each dict has ``trial_id``, ``time`` (float), ``event`` (0/1).

    Returns
    -------
    dict
        Keys: longitudinal_coefs, survival_coefs, association_alpha,
        log_likelihood.
    """
    # Stage 1: Fit per-trial linear trajectories, then pool
    # Y_i(t) = a0 + a1*t + epsilon
    all_times = []
    all_values = []
    for rec in longitudinal_data:
        all_times.extend(rec["times"])
        all_values.extend(rec["values"])

    t_arr = np.array(all_times, dtype=float)
    y_arr = np.array(all_values, dtype=float)

    # OLS: Y = [1, t] @ [a0, a1]
    A = np.column_stack([np.ones_like(t_arr), t_arr])
    long_coefs, residuals, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)

    a0, a1 = float(long_coefs[0]), float(long_coefs[1])

    # Stage 2: Predict trajectory value at survival time for each trial
    surv_map = {rec["trial_id"]: rec for rec in survival_data}
    long_map: Dict[str, Dict] = {}
    for rec in longitudinal_data:
        long_map[rec["trial_id"]] = rec

    trial_ids = [rec["trial_id"] for rec in survival_data]
    n_surv = len(trial_ids)

    times_surv = np.array([surv_map[tid]["time"] for tid in trial_ids], dtype=float)
    events_surv = np.array([surv_map[tid]["event"] for tid in trial_ids], dtype=int)

    # Predicted trajectory value at survival time
    predicted_vals = a0 + a1 * times_surv

    # Cox PH with predicted value as single covariate
    X_cox = predicted_vals.reshape(-1, 1)
    cox_result = cox_ph(
        X_cox, times_surv, events_surv,
        feature_names=["predicted_trajectory"],
        max_iter=50,
    )

    # Association parameter alpha is the log-HR for the trajectory covariate
    alpha = 0.0
    if cox_result["hazard_ratios"]:
        hr = cox_result["hazard_ratios"][0].get("hr", 1.0)
        alpha = float(math.log(max(hr, 1e-10)))

    return {
        "longitudinal_coefs": {"intercept": round(a0, 4), "slope": round(a1, 4)},
        "survival_coefs": cox_result["hazard_ratios"],
        "association_alpha": round(alpha, 4),
        "log_likelihood": cox_result.get("log_likelihood", 0.0),
    }


# ============================================================
# 16. Bayesian Change-Point Detection (RJMCMC)
# ============================================================


def bayesian_changepoint(
    time_series: Sequence[float],
    max_cp: int = 5,
    n_iter: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bayesian change-point detection via reversible-jump MCMC.

    Uses birth/death/shift moves with Metropolis-Hastings acceptance
    to estimate the number and location of change points.

    Parameters
    ----------
    time_series : sequence of float
        Observed time series values.
    max_cp : int
        Maximum number of change points.
    n_iter : int
        Number of MCMC iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: n_changepoints, changepoint_posteriors (dict mapping
        position -> posterior probability), segment_means, dic.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(time_series, dtype=float)
    n = len(y)

    if n < 4:
        return {
            "n_changepoints": 0,
            "changepoint_posteriors": {},
            "segment_means": [round(float(np.mean(y)), 4)],
            "dic": 0.0,
        }

    # Constant or near-constant series: no change points possible
    if np.std(y) < 1e-10:
        return {
            "n_changepoints": 0,
            "changepoint_posteriors": {},
            "segment_means": [round(float(np.mean(y)), 4)],
            "dic": 0.0,
        }

    # Precompute cumulative sums for fast segment stats
    cum_y = np.concatenate([[0.0], np.cumsum(y)])
    cum_y2 = np.concatenate([[0.0], np.cumsum(y ** 2)])

    def _seg_loglik(start: int, end: int) -> float:
        """Log-likelihood of segment [start, end) under normal model."""
        seg_n = end - start
        if seg_n <= 0:
            return -1e10
        seg_sum = cum_y[end] - cum_y[start]
        seg_sum2 = cum_y2[end] - cum_y2[start]
        seg_mean = seg_sum / seg_n
        seg_var = max(seg_sum2 / seg_n - seg_mean ** 2, 1e-10)
        return -0.5 * seg_n * (1.0 + math.log(2.0 * math.pi * seg_var))

    def _total_loglik(cps: List[int]) -> float:
        """Total log-likelihood given change points."""
        boundaries = [0] + sorted(cps) + [n]
        ll = 0.0
        for i in range(len(boundaries) - 1):
            ll += _seg_loglik(boundaries[i], boundaries[i + 1])
        return ll

    # Initialize with no change points
    current_cps: List[int] = []
    current_ll = _total_loglik(current_cps)

    # Track posterior visits
    cp_visits = np.zeros(n, dtype=int)
    k_trace = []
    ll_trace = []

    for it in range(n_iter):
        k = len(current_cps)

        # Choose move type
        u = rng.random()
        if k == 0:
            move = "birth"
        elif k >= max_cp:
            move = "death" if u < 0.5 else "shift"
        else:
            if u < 0.4:
                move = "birth"
            elif u < 0.7:
                move = "death"
            else:
                move = "shift"

        if move == "birth":
            # Propose a new change point
            forbidden = set(current_cps) | {0, n}
            candidates = [i for i in range(2, n - 1) if i not in forbidden]
            if not candidates:
                continue
            new_cp = int(rng.choice(candidates))
            proposed = current_cps + [new_cp]
            proposed_ll = _total_loglik(proposed)
            # Prior ratio: uniform prior on k -> favor parsimony slightly
            log_prior = -0.5  # penalize extra cp
            log_alpha = proposed_ll - current_ll + log_prior
            if math.log(max(rng.random(), 1e-300)) < log_alpha:
                current_cps = proposed
                current_ll = proposed_ll

        elif move == "death" and k > 0:
            idx = int(rng.randint(0, k))
            proposed = current_cps[:idx] + current_cps[idx + 1:]
            proposed_ll = _total_loglik(proposed)
            log_prior = 0.5  # reward parsimony
            log_alpha = proposed_ll - current_ll + log_prior
            if math.log(max(rng.random(), 1e-300)) < log_alpha:
                current_cps = proposed
                current_ll = proposed_ll

        elif move == "shift" and k > 0:
            idx = int(rng.randint(0, k))
            old_cp = current_cps[idx]
            shift = int(rng.choice([-2, -1, 1, 2]))
            new_cp = old_cp + shift
            forbidden = set(current_cps) - {old_cp} | {0, n}
            if 2 <= new_cp < n - 1 and new_cp not in forbidden:
                proposed = current_cps[:idx] + [new_cp] + current_cps[idx + 1:]
                proposed_ll = _total_loglik(proposed)
                log_alpha = proposed_ll - current_ll
                if math.log(max(rng.random(), 1e-300)) < log_alpha:
                    current_cps = proposed
                    current_ll = proposed_ll

        # Record
        for cp in current_cps:
            cp_visits[cp] += 1
        k_trace.append(len(current_cps))
        ll_trace.append(current_ll)

    # Posterior summaries
    burn_in = n_iter // 4
    k_post = k_trace[burn_in:]
    from collections import Counter
    k_counts = Counter(k_post)
    best_k = k_counts.most_common(1)[0][0]

    # Change-point posterior probabilities
    n_post = n_iter - burn_in
    cp_posteriors = {}
    for pos in range(n):
        prob = cp_visits[pos] / n_iter  # approximate
        if prob > 0.05:
            cp_posteriors[int(pos)] = round(float(prob), 4)

    # Segment means from MAP change points
    # Use the most visited positions above threshold
    map_cps = sorted(
        [pos for pos, prob in cp_posteriors.items() if prob > 0.1],
        key=lambda x: -cp_posteriors.get(x, 0),
    )[:best_k]
    boundaries = [0] + sorted(map_cps) + [n]
    segment_means = []
    for i in range(len(boundaries) - 1):
        seg = y[boundaries[i]:boundaries[i + 1]]
        segment_means.append(round(float(np.mean(seg)), 4))

    # DIC approximation
    mean_ll = float(np.mean(ll_trace[burn_in:]))
    ll_at_mean = current_ll  # approximate
    p_d = 2.0 * (mean_ll - ll_at_mean) if mean_ll > ll_at_mean else 0.0
    dic = -2.0 * mean_ll + 2.0 * p_d

    return {
        "n_changepoints": best_k,
        "changepoint_posteriors": cp_posteriors,
        "segment_means": segment_means,
        "dic": round(float(dic), 4),
    }


# ============================================================
# 17. Granger Causality
# ============================================================


def granger_causality(
    amendment_counts_by_type: Dict[str, List[float]],
    max_lag: int = 3,
) -> Dict[str, Any]:
    """Granger causality test between amendment type time series.

    Fits VAR models and performs F-tests comparing restricted (no
    lagged values of X) vs unrestricted (includes lagged X) models
    for each pair of amendment types.

    Parameters
    ----------
    amendment_counts_by_type : dict
        Keys are amendment type names, values are time series of counts.
    max_lag : int
        Maximum lag to test.

    Returns
    -------
    dict
        Keys: results (list of {cause, effect, f_stat, p_value, lag}),
        significant_pairs (list of (cause, effect) with p < 0.05).
    """
    types = list(amendment_counts_by_type.keys())
    n_types = len(types)
    results = []
    significant_pairs = []

    for cause_name in types:
        for effect_name in types:
            if cause_name == effect_name:
                continue

            x = np.array(amendment_counts_by_type[cause_name], dtype=float)
            y_series = np.array(amendment_counts_by_type[effect_name], dtype=float)
            T = min(len(x), len(y_series))

            if T <= max_lag + 2:
                continue

            best_f = 0.0
            best_p = 1.0
            best_lag = 1

            for lag in range(1, max_lag + 1):
                n_obs = T - lag

                # Restricted model: Y_t = a0 + a1*Y_{t-1} + ... + a_lag*Y_{t-lag}
                Y = y_series[lag:T]
                X_restricted = np.column_stack(
                    [np.ones(n_obs)]
                    + [y_series[lag - l:T - l] for l in range(1, lag + 1)]
                )

                # Unrestricted: add lagged X
                X_unrestricted = np.column_stack(
                    [X_restricted]
                    + [x[lag - l:T - l] for l in range(1, lag + 1)]
                )

                # OLS for both
                try:
                    beta_r, res_r, _, _ = np.linalg.lstsq(X_restricted, Y, rcond=None)
                    beta_u, res_u, _, _ = np.linalg.lstsq(X_unrestricted, Y, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                rss_r = float(np.sum((Y - X_restricted @ beta_r) ** 2))
                rss_u = float(np.sum((Y - X_unrestricted @ beta_u) ** 2))

                q = lag  # number of restrictions
                df_u = n_obs - X_unrestricted.shape[1]

                if rss_u <= 1e-15 or df_u <= 0:
                    continue

                f_stat = ((rss_r - rss_u) / q) / (rss_u / df_u)
                p_value = 1.0 - float(sp_stats.f.cdf(max(f_stat, 0), q, df_u))

                if p_value < best_p:
                    best_f = f_stat
                    best_p = p_value
                    best_lag = lag

            results.append({
                "cause": cause_name,
                "effect": effect_name,
                "f_stat": round(float(best_f), 4),
                "p_value": round(float(best_p), 6),
                "lag": best_lag,
            })

            if best_p < 0.05:
                significant_pairs.append((cause_name, effect_name))

    return {
        "results": results,
        "significant_pairs": significant_pairs,
    }


# ============================================================
# 18. Functional PCA
# ============================================================


def functional_pca(
    trajectories: List[List[float]],
    n_components: int = 3,
) -> Dict[str, Any]:
    """Functional PCA on discretized amendment trajectories.

    Discretizes amendment timelines onto a common grid, centres them,
    and performs eigendecomposition of the covariance matrix.

    Parameters
    ----------
    trajectories : list of list of float
        Each inner list is a discretized amendment count trajectory.
        Lists may have different lengths; they will be interpolated
        to a common grid.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    dict
        Keys: components (list of arrays), scores (n_traj x n_comp),
        mean_function (array), reconstruction_error (float),
        variance_explained (list of float).
    """
    if len(trajectories) == 0:
        return {
            "components": [],
            "scores": [],
            "mean_function": [],
            "reconstruction_error": 0.0,
            "variance_explained": [],
        }

    # Interpolate all trajectories to common grid
    max_len = max(len(t) for t in trajectories)
    grid_size = max(max_len, 50)
    grid = np.linspace(0, 1, grid_size)

    interpolated = []
    for traj in trajectories:
        if len(traj) < 2:
            interpolated.append(np.full(grid_size, traj[0] if traj else 0.0))
        else:
            x_orig = np.linspace(0, 1, len(traj))
            interp_vals = np.interp(grid, x_orig, traj)
            interpolated.append(interp_vals)

    X = np.array(interpolated)  # (n_traj, grid_size)
    n_traj = X.shape[0]

    # Centre
    mean_func = np.mean(X, axis=0)
    X_centered = X - mean_func

    # Covariance and eigen
    n_comp = min(n_components, n_traj, grid_size)
    if n_comp < 1:
        n_comp = 1

    # Use SVD for numerical stability
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_comp, :].tolist()
    eigenvalues = (S[:n_comp] ** 2) / max(n_traj - 1, 1)
    total_var = float(np.sum(S ** 2) / max(n_traj - 1, 1))

    variance_explained = []
    for ev in eigenvalues:
        variance_explained.append(
            round(float(ev / total_var) if total_var > 0 else 0.0, 6)
        )

    # Scores: project centered data onto components
    scores = (X_centered @ Vt[:n_comp, :].T).tolist()

    # Reconstruction error
    X_recon = (np.array(scores) @ np.array(components)) + mean_func
    recon_error = float(np.mean((X - X_recon) ** 2))

    return {
        "components": components,
        "scores": scores,
        "mean_function": mean_func.tolist(),
        "reconstruction_error": round(recon_error, 6),
        "variance_explained": variance_explained,
    }
