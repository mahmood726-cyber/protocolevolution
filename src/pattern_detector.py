"""Detect 5 composite concerning patterns from trial amendment signals.

Patterns:
  RESCUE           - sample size UP + endpoint changed + timeline extended
  STEALTH_SWITCH   - endpoint changed WITHOUT enrollment adjustment
  SCOPE_CREEP      - >=3 eligibility criteria loosened
  GOALPOST_MOVE    - completion extended >=2 times without enrollment increase
  FUNNEL_NARROWING - eligibility tightened + sample size decreased
"""

from typing import Any, Dict, List

from .amendment_harvester import extract_amendment_signals
from .diff_engine import compute_trial_diffs
from .change_classifier import classify_changes, ChangeType, ChangeCategory


def detect_patterns(trial: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect composite concerning patterns in a trial.

    Parameters
    ----------
    trial : dict
        Trial record with CT.gov-style fields.

    Returns
    -------
    list of dict
        Detected patterns, each with: pattern, severity, evidence list.
    """
    signals = extract_amendment_signals(trial)
    diffs = compute_trial_diffs(trial)
    changes = classify_changes(diffs)

    patterns: List[Dict[str, Any]] = []

    # Build lookup sets for quick checks
    change_types = {c["type"] for c in changes}
    change_categories = {c["category"] for c in changes}

    has_enrollment_increase = ChangeType.SAMPLE_SIZE_INCREASE.value in change_types
    has_enrollment_decrease = ChangeType.SAMPLE_SIZE_DECREASE.value in change_types
    has_endpoint_change = ChangeCategory.ENDPOINTS.value in change_categories
    has_completion_extended = ChangeType.COMPLETION_EXTENDED.value in change_types

    # Count loosened eligibility
    loosening_types = {
        ChangeType.AGE_RANGE_WIDENED.value,
        ChangeType.INCLUSION_LOOSENED.value,
        ChangeType.EXCLUSION_REMOVED.value,
    }
    loosened_count = sum(1 for c in changes if c["type"] in loosening_types)

    # Count tightened eligibility
    tightening_types = {
        ChangeType.AGE_RANGE_NARROWED.value,
        ChangeType.INCLUSION_TIGHTENED.value,
        ChangeType.EXCLUSION_ADDED.value,
    }
    tightened_count = sum(1 for c in changes if c["type"] in tightening_types)

    completion_extensions = signals.get("completionExtensions", 0)

    # --- RESCUE ---
    if has_enrollment_increase and has_endpoint_change and has_completion_extended:
        patterns.append({
            "pattern": "RESCUE",
            "severity": "HIGH",
            "evidence": [
                f"Enrollment increased: {signals['enrollmentEstimated']} -> {signals['enrollmentActual']}",
                "Primary endpoint changed",
                f"Completion extended by {signals['completionExtensionDays']} days",
            ],
        })

    # --- STEALTH_SWITCH ---
    if has_endpoint_change and not has_enrollment_increase and not has_enrollment_decrease:
        patterns.append({
            "pattern": "STEALTH_SWITCH",
            "severity": "HIGH",
            "evidence": [
                "Primary endpoint changed without enrollment adjustment",
                f"Enrollment remained: {signals['enrollmentEstimated']} -> {signals['enrollmentActual']}",
            ],
        })

    # --- SCOPE_CREEP ---
    if loosened_count >= 3:
        loosened_details = [
            c["details"] for c in changes if c["type"] in loosening_types
        ]
        patterns.append({
            "pattern": "SCOPE_CREEP",
            "severity": "MODERATE",
            "evidence": [
                f"{loosened_count} eligibility criteria loosened",
                *loosened_details,
            ],
        })

    # --- GOALPOST_MOVE ---
    if completion_extensions >= 2 and not has_enrollment_increase:
        patterns.append({
            "pattern": "GOALPOST_MOVE",
            "severity": "MODERATE",
            "evidence": [
                f"Completion extended {completion_extensions} times",
                "No enrollment increase to justify extensions",
                f"Enrollment: {signals['enrollmentEstimated']} -> {signals['enrollmentActual']}",
            ],
        })

    # --- FUNNEL_NARROWING ---
    if tightened_count > 0 and has_enrollment_decrease:
        tightened_details = [
            c["details"] for c in changes if c["type"] in tightening_types
        ]
        patterns.append({
            "pattern": "FUNNEL_NARROWING",
            "severity": "MODERATE",
            "evidence": [
                f"Eligibility tightened ({tightened_count} criteria)",
                f"Sample size decreased: {signals['enrollmentEstimated']} -> {signals['enrollmentActual']}",
                *tightened_details,
            ],
        })

    return patterns
