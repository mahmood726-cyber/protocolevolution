"""Statistical engine for ProtocolEvolution.

Provides 8 methods in two layers:

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
