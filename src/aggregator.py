"""Roll up amendment signals by sponsor, condition, and phase."""

from collections import defaultdict
from typing import Any, Dict, List

from .amendment_harvester import extract_amendment_signals
from .pattern_detector import detect_patterns


def aggregate_amendments(
    trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate amendment prevalence and patterns across trials.

    Parameters
    ----------
    trials : list of dict
        List of trial records.

    Returns
    -------
    dict
        Keys: totalTrials, prevalence, bySponsor, byCondition, byPhase.
    """
    total = len(trials)
    if total == 0:
        return {
            "totalTrials": 0,
            "prevalence": {},
            "bySponsor": {},
            "byCondition": {},
            "byPhase": {},
        }

    # Per-trial signals and patterns
    all_signals = []
    all_patterns = []
    for trial in trials:
        signals = extract_amendment_signals(trial)
        pats = detect_patterns(trial)
        all_signals.append(signals)
        all_patterns.append(pats)

    # Prevalence counts
    enrollment_changed = sum(
        1
        for s in all_signals
        if s["enrollmentDeltaPct"] is not None and abs(s["enrollmentDeltaPct"]) > 20
    )
    endpoint_changed = sum(1 for s in all_signals if s["endpointChanged"])
    completion_extended = sum(
        1
        for s in all_signals
        if s["completionExtensionDays"] is not None
        and s["completionExtensionDays"] > 180
    )
    eligibility_changed = sum(
        1 for s in all_signals if s["eligibilityChangeCount"] >= 3
    )
    any_pattern = sum(1 for pats in all_patterns if len(pats) > 0)

    prevalence = {
        "enrollmentChanged": enrollment_changed,
        "endpointChanged": endpoint_changed,
        "completionExtended": completion_extended,
        "eligibilityChanged": eligibility_changed,
        "anyPattern": any_pattern,
        "enrollmentChangedPct": round(enrollment_changed / total * 100, 1),
        "endpointChangedPct": round(endpoint_changed / total * 100, 1),
        "completionExtendedPct": round(completion_extended / total * 100, 1),
        "eligibilityChangedPct": round(eligibility_changed / total * 100, 1),
        "anyPatternPct": round(any_pattern / total * 100, 1),
    }

    # By sponsor
    by_sponsor: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"trials": 0, "amended": 0, "patterns": defaultdict(int)}
    )
    for i, trial in enumerate(trials):
        sponsor = all_signals[i]["sponsor"] or "Unknown"
        by_sponsor[sponsor]["trials"] += 1
        if len(all_patterns[i]) > 0:
            by_sponsor[sponsor]["amended"] += 1
        for pat in all_patterns[i]:
            by_sponsor[sponsor]["patterns"][pat["pattern"]] += 1

    # Convert defaultdicts for JSON serialization
    by_sponsor_clean = {}
    for sp, info in by_sponsor.items():
        by_sponsor_clean[sp] = {
            "trials": info["trials"],
            "amended": info["amended"],
            "patterns": dict(info["patterns"]),
        }

    # By condition
    by_condition: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"trials": 0, "amended": 0, "patterns": defaultdict(int)}
    )
    for i, trial in enumerate(trials):
        condition = all_signals[i]["condition"] or "Unknown"
        by_condition[condition]["trials"] += 1
        if len(all_patterns[i]) > 0:
            by_condition[condition]["amended"] += 1
        for pat in all_patterns[i]:
            by_condition[condition]["patterns"][pat["pattern"]] += 1

    by_condition_clean = {}
    for cond, info in by_condition.items():
        by_condition_clean[cond] = {
            "trials": info["trials"],
            "amended": info["amended"],
            "patterns": dict(info["patterns"]),
        }

    # By phase
    by_phase: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"trials": 0, "amended": 0}
    )
    for i, trial in enumerate(trials):
        phase = all_signals[i]["phase"] or "Unknown"
        by_phase[phase]["trials"] += 1
        if len(all_patterns[i]) > 0:
            by_phase[phase]["amended"] += 1

    by_phase_clean = {ph: dict(info) for ph, info in by_phase.items()}

    return {
        "totalTrials": total,
        "prevalence": prevalence,
        "bySponsor": by_sponsor_clean,
        "byCondition": by_condition_clean,
        "byPhase": by_phase_clean,
    }
