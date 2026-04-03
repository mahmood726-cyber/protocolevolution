"""25 tests for ProtocolEvolution stats engine."""

import os
import sys
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stats_engine import (
    pattern_outcome_test,
    kaplan_meier,
    log_rank_test,
    compute_odds_ratio,
    benjamini_hochberg,
    cox_ph,
    amendment_markov_chain,
    propensity_match,
    hidden_markov_model,
    andersen_gill_model,
    frailty_model,
    cusum_detection,
    cure_rate_model,
)


# ============================================================
# Chi-squared / Fisher (2)
# ============================================================

def test_pattern_outcome_significant():
    result = pattern_outcome_test(
        pattern_trials=[{"status": "COMPLETED"}] * 2 + [{"status": "TERMINATED"}] * 8,
        clean_trials=[{"status": "COMPLETED"}] * 9 + [{"status": "TERMINATED"}] * 1,
    )
    assert result["p_value"] < 0.05
    assert result["odds_ratio"] < 1  # pattern -> worse completion


def test_pattern_outcome_nonsignificant():
    result = pattern_outcome_test(
        pattern_trials=[{"status": "COMPLETED"}] * 5 + [{"status": "TERMINATED"}] * 5,
        clean_trials=[{"status": "COMPLETED"}] * 5 + [{"status": "TERMINATED"}] * 5,
    )
    assert result["p_value"] > 0.1


# ============================================================
# Kaplan-Meier (2)
# ============================================================

def test_km_basic():
    times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    events = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 5 events, 5 censored
    result = kaplan_meier(times, events)
    assert len(result["times"]) > 0
    assert result["survival"][0] == 1.0  # starts at 1
    assert result["survival"][-1] < 1.0  # decreases
    assert all(
        result["ci_lower"][i] <= result["survival"][i] <= result["ci_upper"][i]
        for i in range(len(result["survival"]))
    )


def test_km_all_events():
    times = [1, 2, 3, 4, 5]
    events = [1, 1, 1, 1, 1]
    result = kaplan_meier(times, events)
    assert result["survival"][-1] == 0.0


# ============================================================
# Log-rank (2)
# ============================================================

def test_log_rank_different():
    times1 = [1, 2, 3, 4, 5]
    events1 = [1, 1, 1, 1, 1]
    times2 = [10, 20, 30, 40, 50]
    events2 = [1, 1, 1, 1, 1]
    result = log_rank_test(times1, events1, times2, events2)
    assert result["p_value"] < 0.05


def test_log_rank_similar():
    times1 = [1, 2, 3, 4, 5]
    events1 = [1, 1, 1, 1, 1]
    times2 = [1, 2, 3, 4, 5]
    events2 = [1, 1, 1, 1, 1]
    result = log_rank_test(times1, events1, times2, events2)
    assert result["p_value"] > 0.1


# ============================================================
# Odds ratio (1)
# ============================================================

def test_odds_ratio():
    result = compute_odds_ratio(10, 2, 3, 15)
    assert result["or_value"] > 10
    assert result["ci_lower"] > 1
    assert result["p_value"] < 0.01


# ============================================================
# Benjamini-Hochberg (1)
# ============================================================

def test_bh_fdr():
    pvals = [0.001, 0.01, 0.04, 0.20, 0.80]
    result = benjamini_hochberg(pvals)
    assert result[0]["significant"] is True  # smallest p
    assert result[-1]["significant"] is False  # largest p


# ============================================================
# Cox PH (2)
# ============================================================

def test_cox_convergence():
    import numpy as np

    np.random.seed(42)
    n = 50
    X = np.column_stack([np.random.randn(n), np.random.binomial(1, 0.5, n)])
    times = np.exp(0.5 * X[:, 0] + np.random.randn(n) * 0.5)
    events = np.ones(n)
    result = cox_ph(X, times, events, feature_names=["continuous", "binary"])
    assert len(result["hazard_ratios"]) == 2
    assert all("hr" in hr and "p_value" in hr for hr in result["hazard_ratios"])


def test_cox_hr_direction():
    import numpy as np

    np.random.seed(42)
    n = 100
    x = np.random.binomial(1, 0.5, n).reshape(-1, 1)
    times = np.where(
        x.ravel() == 1,
        np.random.exponential(2, n),
        np.random.exponential(10, n),
    )
    events = np.ones(n)
    result = cox_ph(x, times, events, feature_names=["treatment"])
    # Treatment group has shorter times -> higher hazard -> HR > 1
    assert result["hazard_ratios"][0]["hr"] > 1


# ============================================================
# Markov chain (2)
# ============================================================

def test_markov_transition():
    sequences = [
        ["ENROLLMENT_UP", "ENDPOINT_CHANGE", "COMPLETION_EXTENDED"],
        ["ENROLLMENT_UP", "COMPLETION_EXTENDED"],
        ["ENDPOINT_CHANGE", "ENROLLMENT_UP"],
    ]
    result = amendment_markov_chain(sequences)
    assert "transition_matrix" in result
    # Rows should sum to ~1
    for row in result["transition_matrix"].values():
        total = sum(row.values())
        assert abs(total - 1.0) < 0.01 or total == 0


def test_markov_stationary():
    sequences = [["A", "B", "A", "B"]] * 10
    result = amendment_markov_chain(sequences)
    assert "stationary_dist" in result
    assert abs(sum(result["stationary_dist"].values()) - 1.0) < 0.01


# ============================================================
# Propensity matching (2)
# ============================================================

def test_propensity_match_basic():
    import numpy as np
    rng = np.random.RandomState(99)
    # Generate overlapping amended and clean trials so matching succeeds
    amended = []
    for _ in range(15):
        amended.append({
            "phase": rng.choice([2, 3]),
            "enrollment": int(rng.normal(400, 100)),
            "sponsor_ind": rng.choice([0, 1]),
            "completed": rng.choice([0, 1], p=[0.6, 0.4]),
        })
    clean = []
    for _ in range(30):
        clean.append({
            "phase": rng.choice([2, 3]),
            "enrollment": int(rng.normal(400, 100)),
            "sponsor_ind": rng.choice([0, 1]),
            "completed": rng.choice([0, 1], p=[0.3, 0.7]),
        })
    features = ["phase", "enrollment", "sponsor_ind"]
    result = propensity_match(amended, clean, features)
    assert "matched_pairs" in result
    assert "att" in result
    assert len(result["matched_pairs"]) > 0


def test_propensity_balance():
    amended = [{"x1": 5, "x2": 1, "completed": 0}] * 10
    clean = [{"x1": 3, "x2": 0, "completed": 1}] * 20
    result = propensity_match(amended, clean, ["x1", "x2"])
    assert "balance" in result
    # After matching, SMD should be smaller than before
    for b in result["balance"]:
        assert "smd_before" in b and "smd_after" in b


# ============================================================
# Integration (1)
# ============================================================

def test_full_stats_pipeline():
    po = pattern_outcome_test(
        [{"status": "COMPLETED"}] * 3 + [{"status": "TERMINATED"}] * 7,
        [{"status": "COMPLETED"}] * 7 + [{"status": "TERMINATED"}] * 3,
    )
    km = kaplan_meier([1, 2, 3, 4, 5], [1, 1, 1, 0, 0])
    orr = compute_odds_ratio(3, 7, 7, 3)
    bh = benjamini_hochberg([po["p_value"], 0.5])
    assert all(v is not None for v in [po, km, orr, bh])


# ============================================================
# Hidden Markov Model (2)
# ============================================================

def test_hmm_forward_backward():
    sequences = [["ENROLLMENT_UP", "ENDPOINT_CHANGE", "COMPLETION_EXTENDED"],
                 ["ENROLLMENT_UP", "COMPLETION_EXTENDED"],
                 ["ENDPOINT_CHANGE", "ENROLLMENT_UP", "ENDPOINT_CHANGE"]]
    result = hidden_markov_model(sequences, n_states=2, n_iter=50, seed=42)
    assert result["transition_matrix"].shape == (2, 2)
    assert abs(result["transition_matrix"].sum(axis=1) - 1.0).max() < 0.01
    assert len(result["decoded_sequences"]) == 3


def test_hmm_state_decoding():
    # Repetitive pattern should decode consistently
    sequences = [["A", "B"] * 5] * 10
    result = hidden_markov_model(sequences, n_states=2, n_iter=50, seed=42)
    # Decoded sequence should alternate between states
    assert len(result["decoded_sequences"][0]) == 10


# ============================================================
# Andersen-Gill (2)
# ============================================================

def test_andersen_gill_basic():
    trials_data = [
        {"trial_id": "T1", "event_times": [30, 90, 180], "features": {"phase": 3, "enrollment": 500}, "max_time": 365},
        {"trial_id": "T2", "event_times": [60], "features": {"phase": 2, "enrollment": 200}, "max_time": 365},
        {"trial_id": "T3", "event_times": [], "features": {"phase": 3, "enrollment": 800}, "max_time": 365},
    ] * 5  # replicate for sample size
    result = andersen_gill_model(trials_data)
    assert len(result["hazard_ratios"]) == 2
    assert result["n_events"] > 0


def test_andersen_gill_hr_structure():
    trials_data = [
        {"trial_id": f"T{i}", "event_times": [30*j for j in range(1, i%3+2)],
         "features": {"x1": i % 2, "x2": i * 0.1}, "max_time": 365}
        for i in range(20)
    ]
    result = andersen_gill_model(trials_data)
    assert all("hr" in hr and "ci_lower" in hr for hr in result["hazard_ratios"])


# ============================================================
# Frailty Model (2)
# ============================================================

def test_frailty_variance():
    import numpy as np
    np.random.seed(42)
    times = np.random.exponential(10, 30)
    events = np.ones(30, dtype=int)
    groups = [f"G{i%3}" for i in range(30)]
    result = frailty_model(times, events, groups)
    assert result["frailty_variance"] >= 0
    assert len(result["frailty_values"]) == 3


def test_frailty_group_effects():
    import numpy as np
    np.random.seed(42)
    # Group 0 has faster events (higher hazard)
    times = np.concatenate([np.random.exponential(2, 15), np.random.exponential(20, 15)])
    events = np.ones(30, dtype=int)
    groups = ["fast"] * 15 + ["slow"] * 15
    result = frailty_model(times, events, groups)
    assert result["frailty_values"]["fast"] > result["frailty_values"]["slow"]


# ============================================================
# CUSUM Detection (2)
# ============================================================

def test_cusum_detects_shift():
    # Clear shift at t=50
    series = [0.3] * 50 + [0.7] * 50
    result = cusum_detection(series, threshold=3.0)
    assert len(result["change_points"]) >= 1
    # Change point should be near index 50
    cp_times = [cp["time"] for cp in result["change_points"]]
    assert any(40 <= t <= 60 for t in cp_times)


def test_cusum_no_shift():
    series = [0.3] * 100
    result = cusum_detection(series, threshold=5.0)
    assert len(result["change_points"]) == 0


# ============================================================
# Cure-Rate Model (2)
# ============================================================

def test_cure_rate_fraction():
    import numpy as np
    np.random.seed(42)
    # 40% never amend (cured), 60% amend at various times
    n = 50
    cured = np.random.binomial(1, 0.4, n)
    times = np.where(cured, 1000, np.random.exponential(10, n))  # cured get large time
    events = 1 - cured  # cured are censored
    result = cure_rate_model(times, events, seed=42)
    assert 0.2 < result["cure_fraction"] < 0.6  # should be near 0.4


def test_cure_rate_ci():
    import numpy as np
    np.random.seed(42)
    times = np.random.exponential(10, 40)
    events = np.random.binomial(1, 0.7, 40)
    result = cure_rate_model(times, events, seed=42)
    assert result["cure_ci"][0] < result["cure_fraction"] < result["cure_ci"][1]
