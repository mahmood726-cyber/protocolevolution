"""Classify diffs into a taxonomy of 4 categories and 12+ change types."""

from enum import Enum
from typing import Any, Dict, List


class ChangeCategory(str, Enum):
    ENROLLMENT = "ENROLLMENT"
    ENDPOINTS = "ENDPOINTS"
    ELIGIBILITY = "ELIGIBILITY"
    TIMELINE = "TIMELINE"


class ChangeType(str, Enum):
    # Enrollment
    SAMPLE_SIZE_INCREASE = "SAMPLE_SIZE_INCREASE"
    SAMPLE_SIZE_DECREASE = "SAMPLE_SIZE_DECREASE"
    # Endpoints
    PRIMARY_ADDED = "PRIMARY_ADDED"
    PRIMARY_REMOVED = "PRIMARY_REMOVED"
    PRIMARY_MODIFIED = "PRIMARY_MODIFIED"
    # Eligibility
    AGE_RANGE_WIDENED = "AGE_RANGE_WIDENED"
    AGE_RANGE_NARROWED = "AGE_RANGE_NARROWED"
    INCLUSION_LOOSENED = "INCLUSION_LOOSENED"
    INCLUSION_TIGHTENED = "INCLUSION_TIGHTENED"
    EXCLUSION_ADDED = "EXCLUSION_ADDED"
    EXCLUSION_REMOVED = "EXCLUSION_REMOVED"
    # Timeline
    COMPLETION_EXTENDED = "COMPLETION_EXTENDED"
    COMPLETION_SHORTENED = "COMPLETION_SHORTENED"


# Map diff field + direction to (category, type)
_FIELD_CLASSIFIERS = {
    "enrollment": lambda d: _classify_enrollment(d),
    "completionDate": lambda d: _classify_completion(d),
    "eligibility": lambda d: _classify_eligibility(d),
    "primaryEndpoint": lambda d: _classify_endpoint(d),
    "updateActivity": lambda d: [],  # informational only
}


def _classify_enrollment(diff: Dict[str, Any]) -> List[Dict[str, Any]]:
    old = diff.get("oldValue", 0)
    new = diff.get("newValue", 0)
    if new > old:
        return [{
            "category": ChangeCategory.ENROLLMENT.value,
            "type": ChangeType.SAMPLE_SIZE_INCREASE.value,
            "field": "enrollment",
            "details": diff.get("details", ""),
        }]
    elif new < old:
        return [{
            "category": ChangeCategory.ENROLLMENT.value,
            "type": ChangeType.SAMPLE_SIZE_DECREASE.value,
            "field": "enrollment",
            "details": diff.get("details", ""),
        }]
    return []


def _classify_completion(diff: Dict[str, Any]) -> List[Dict[str, Any]]:
    details = diff.get("details", "")
    # Parse days from details string
    days = 0
    if "Extended by" in details or "by" in details:
        import re
        m = re.search(r"(-?\d+)\s*days", details)
        if m:
            days = int(m.group(1))

    if days > 0:
        return [{
            "category": ChangeCategory.TIMELINE.value,
            "type": ChangeType.COMPLETION_EXTENDED.value,
            "field": "completionDate",
            "details": details,
        }]
    elif days < 0:
        return [{
            "category": ChangeCategory.TIMELINE.value,
            "type": ChangeType.COMPLETION_SHORTENED.value,
            "field": "completionDate",
            "details": details,
        }]
    return []


def _classify_eligibility(diff: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Classify eligibility changes from the details string or raw changes."""
    details = diff.get("details", "")
    results = []

    # Map known eligibility change type keywords
    type_map = {
        "age_range_widened": (ChangeCategory.ELIGIBILITY.value, ChangeType.AGE_RANGE_WIDENED.value),
        "age_range_narrowed": (ChangeCategory.ELIGIBILITY.value, ChangeType.AGE_RANGE_NARROWED.value),
        "inclusion_loosened": (ChangeCategory.ELIGIBILITY.value, ChangeType.INCLUSION_LOOSENED.value),
        "inclusion_tightened": (ChangeCategory.ELIGIBILITY.value, ChangeType.INCLUSION_TIGHTENED.value),
        "exclusion_added": (ChangeCategory.ELIGIBILITY.value, ChangeType.EXCLUSION_ADDED.value),
        "exclusion_removed": (ChangeCategory.ELIGIBILITY.value, ChangeType.EXCLUSION_REMOVED.value),
    }

    # Parse individual changes from the semicolon-separated details
    parts = [p.strip() for p in details.split(";") if p.strip()]
    for part in parts:
        matched = False
        for key, (cat, ctype) in type_map.items():
            if key.replace("_", " ") in part.lower() or key in part.lower():
                results.append({
                    "category": cat,
                    "type": ctype,
                    "field": "eligibility",
                    "details": part,
                })
                matched = True
                break
        if not matched:
            # Generic eligibility change
            results.append({
                "category": ChangeCategory.ELIGIBILITY.value,
                "type": "ELIGIBILITY_CHANGE",
                "field": "eligibility",
                "details": part,
            })

    # If no parts parsed but there is an eligibility diff, add a generic one
    if not results and diff.get("field") == "eligibility":
        results.append({
            "category": ChangeCategory.ELIGIBILITY.value,
            "type": "ELIGIBILITY_CHANGE",
            "field": "eligibility",
            "details": details,
        })

    return results


def _classify_endpoint(diff: Dict[str, Any]) -> List[Dict[str, Any]]:
    old_val = diff.get("oldValue", "")
    new_val = diff.get("newValue", "")

    results = []

    # Determine what happened: added, removed, or modified
    old_set = set(o.strip() for o in old_val.split(",") if o.strip()) if old_val else set()
    new_set = set(n.strip() for n in new_val.split(",") if n.strip()) if new_val else set()

    added = new_set - old_set
    removed = old_set - new_set

    if added:
        results.append({
            "category": ChangeCategory.ENDPOINTS.value,
            "type": ChangeType.PRIMARY_ADDED.value,
            "field": "primaryEndpoint",
            "details": f"Added: {', '.join(added)}",
        })
    if removed:
        results.append({
            "category": ChangeCategory.ENDPOINTS.value,
            "type": ChangeType.PRIMARY_REMOVED.value,
            "field": "primaryEndpoint",
            "details": f"Removed: {', '.join(removed)}",
        })
    if not added and not removed and old_set != new_set:
        results.append({
            "category": ChangeCategory.ENDPOINTS.value,
            "type": ChangeType.PRIMARY_MODIFIED.value,
            "field": "primaryEndpoint",
            "details": diff.get("details", ""),
        })

    return results


def classify_changes(diffs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify a list of diffs into the change taxonomy.

    Parameters
    ----------
    diffs : list of dict
        Output from compute_trial_diffs.

    Returns
    -------
    list of dict
        Classified changes, each with category, type, field, details.
    """
    classified: List[Dict[str, Any]] = []
    for diff in diffs:
        field = diff.get("field", "")
        classifier = _FIELD_CLASSIFIERS.get(field)
        if classifier:
            classified.extend(classifier(diff))
    return classified
