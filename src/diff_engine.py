"""Compare registered vs current trial state to identify significant diffs.

Produces a list of diff records, each flagged with significance thresholds.
"""

from typing import Any, Dict, List

from .amendment_harvester import extract_amendment_signals


def compute_trial_diffs(trial: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute diffs between registered and current trial state.

    Parameters
    ----------
    trial : dict
        Trial record with CT.gov-style fields.

    Returns
    -------
    list of dict
        Each diff has: field, oldValue, newValue, significant (bool), details.
    """
    signals = extract_amendment_signals(trial)
    diffs: List[Dict[str, Any]] = []

    # --- Enrollment diff ---
    est = signals["enrollmentEstimated"]
    act = signals["enrollmentActual"]
    delta_pct = signals["enrollmentDeltaPct"]
    if est is not None and act is not None and est != act:
        significant = delta_pct is not None and abs(delta_pct) > 20
        diffs.append({
            "field": "enrollment",
            "oldValue": est,
            "newValue": act,
            "significant": significant,
            "details": f"Delta {delta_pct}% ({signals['enrollmentDelta']:+d} participants)",
        })

    # --- Completion date diff ---
    ext_days = signals["completionExtensionDays"]
    if ext_days is not None and ext_days != 0:
        significant = abs(ext_days) > 180
        orig = trial.get("completionDateOriginal", "")
        final = trial.get("completionDateFinal", "")
        diffs.append({
            "field": "completionDate",
            "oldValue": orig,
            "newValue": final,
            "significant": significant,
            "details": f"Extended by {ext_days} days",
        })

    # --- Update activity diff ---
    gap = signals["updateGapDays"]
    if gap is not None:
        # >3 years span = significant sustained updating
        significant = gap > 3 * 365
        if significant:
            diffs.append({
                "field": "updateActivity",
                "oldValue": trial.get("studyFirstPostDate", ""),
                "newValue": trial.get("lastUpdatePostDate", ""),
                "significant": True,
                "details": f"Update span {gap} days ({gap / 365:.1f} years)",
            })

    # --- Eligibility diff ---
    elig_count = signals["eligibilityChangeCount"]
    if elig_count > 0:
        significant = elig_count >= 3
        diffs.append({
            "field": "eligibility",
            "oldValue": "original criteria",
            "newValue": f"{elig_count} changes detected",
            "significant": significant,
            "details": "; ".join(
                f"[{c.get('type', 'unknown')}] {c.get('detail', '')}"
                for c in signals["eligibilityChanges"]
            ),
        })

    # --- Primary endpoint mismatch ---
    if signals["endpointChanged"]:
        protocol_primary = trial.get("protocolPrimary", [])
        results_primary = trial.get("resultsPrimary", [])
        diffs.append({
            "field": "primaryEndpoint",
            "oldValue": ", ".join(protocol_primary),
            "newValue": ", ".join(results_primary),
            "significant": True,
            "details": "Registered primary endpoint differs from reported primary",
        })

    return diffs
