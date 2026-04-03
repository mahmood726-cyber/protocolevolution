"""Link amendment patterns to trial outcomes.

Compares completion rates for amended vs clean trials,
and tracks pattern-specific fates (COMPLETED vs TERMINATED etc.).
"""

from typing import Any, Dict, List

from .amendment_harvester import extract_amendment_signals
from .pattern_detector import detect_patterns


def _is_amended(trial: Dict[str, Any]) -> bool:
    """Determine if a trial shows amendment signals."""
    signals = extract_amendment_signals(trial)

    delta_pct = signals.get("enrollmentDeltaPct")
    if delta_pct is not None and abs(delta_pct) > 20:
        return True

    ext_days = signals.get("completionExtensionDays")
    if ext_days is not None and ext_days > 180:
        return True

    if signals.get("endpointChanged"):
        return True

    if signals.get("eligibilityChangeCount", 0) >= 3:
        return True

    return False


def link_amendments_to_outcomes(
    trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare outcomes between amended and clean trials.

    Parameters
    ----------
    trials : list of dict
        List of trial records.

    Returns
    -------
    dict
        Keys: amendedTrials, cleanTrials, amendedCompletionRate,
        cleanCompletionRate, completionRateDelta, patternFates.
    """
    amended_trials: List[Dict[str, Any]] = []
    clean_trials: List[Dict[str, Any]] = []

    for trial in trials:
        if _is_amended(trial):
            amended_trials.append(trial)
        else:
            clean_trials.append(trial)

    # Completion rates
    amended_completed = sum(
        1 for t in amended_trials if t.get("status") == "COMPLETED"
    )
    clean_completed = sum(
        1 for t in clean_trials if t.get("status") == "COMPLETED"
    )

    amended_rate = (
        round(amended_completed / len(amended_trials) * 100, 1)
        if amended_trials
        else None
    )
    clean_rate = (
        round(clean_completed / len(clean_trials) * 100, 1)
        if clean_trials
        else None
    )

    delta = None
    if amended_rate is not None and clean_rate is not None:
        delta = round(amended_rate - clean_rate, 1)

    # Pattern-specific fates
    pattern_fates: Dict[str, Dict[str, int]] = {}
    for trial in trials:
        detected = detect_patterns(trial)
        status = trial.get("status", "UNKNOWN")
        for pat in detected:
            name = pat["pattern"]
            if name not in pattern_fates:
                pattern_fates[name] = {}
            pattern_fates[name][status] = pattern_fates[name].get(status, 0) + 1

    return {
        "amendedTrials": len(amended_trials),
        "cleanTrials": len(clean_trials),
        "amendedCompletionRate": amended_rate,
        "cleanCompletionRate": clean_rate,
        "completionRateDelta": delta,
        "patternFates": pattern_fates,
    }
