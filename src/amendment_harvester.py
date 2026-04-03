"""Extract amendment proxy signals from CT.gov trial records.

CT.gov API v2 has NO version history endpoint, so we detect amendments
via proxy signals: enrollment deltas, completion extensions, update gaps,
outcome mismatches, and eligibility changes.
"""

from datetime import datetime, date
from typing import Any, Dict, Optional


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string in YYYY-MM-DD, YYYY-MM, or YYYY format."""
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def extract_amendment_signals(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Extract amendment proxy signals from a single trial record.

    Parameters
    ----------
    trial : dict
        Trial record with CT.gov-style fields.

    Returns
    -------
    dict
        Extracted signals including enrollment delta, update gap,
        completion extension, eligibility changes, etc.
    """
    nct_id = trial.get("nctId", "")

    # --- Enrollment signals ---
    est = trial.get("enrollmentEstimated")
    act = trial.get("enrollmentActual")
    if est is not None and act is not None and est > 0:
        enrollment_delta = act - est
        enrollment_delta_pct = round((enrollment_delta / est) * 100, 1)
        enrollment_ratio = round(act / est, 3)
    else:
        enrollment_delta = None
        enrollment_delta_pct = None
        enrollment_ratio = None

    # --- Update gap ---
    first_post = _parse_date(trial.get("studyFirstPostDate"))
    last_update = _parse_date(trial.get("lastUpdatePostDate"))
    if first_post and last_update:
        update_gap_days = (last_update - first_post).days
    else:
        update_gap_days = None

    # --- Completion extension ---
    comp_orig = _parse_date(trial.get("completionDateOriginal"))
    comp_final = _parse_date(trial.get("completionDateFinal"))
    if comp_orig and comp_final:
        completion_extension_days = (comp_final - comp_orig).days
    else:
        completion_extension_days = None

    completion_extensions = trial.get("completionDateExtensions", 0)

    # --- Eligibility changes ---
    elig_changes = trial.get("eligibilityChanges", [])
    eligibility_change_count = len(elig_changes)

    # --- Outcome mismatch ---
    protocol_primary = trial.get("protocolPrimary", [])
    results_primary = trial.get("resultsPrimary", [])
    endpoint_changed = (
        len(protocol_primary) > 0
        and len(results_primary) > 0
        and set(protocol_primary) != set(results_primary)
    )

    return {
        "nctId": nct_id,
        "enrollmentEstimated": est,
        "enrollmentActual": act,
        "enrollmentDelta": enrollment_delta,
        "enrollmentDeltaPct": enrollment_delta_pct,
        "enrollmentRatio": enrollment_ratio,
        "updateGapDays": update_gap_days,
        "completionExtensionDays": completion_extension_days,
        "completionExtensions": completion_extensions,
        "eligibilityChangeCount": eligibility_change_count,
        "eligibilityChanges": elig_changes,
        "endpointChanged": endpoint_changed,
        "status": trial.get("status", ""),
        "sponsor": trial.get("sponsor", ""),
        "sponsorClass": trial.get("sponsorClass", ""),
        "condition": trial.get("condition", ""),
        "phase": trial.get("phase", ""),
    }
