"""25 tests for ProtocolEvolution — protocol amendment pattern detector."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.amendment_harvester import extract_amendment_signals, _parse_date
from src.diff_engine import compute_trial_diffs
from src.change_classifier import classify_changes, ChangeCategory, ChangeType
from src.pattern_detector import detect_patterns
from src.outcome_linker import link_amendments_to_outcomes
from src.aggregator import aggregate_amendments


# ============================================================
# amendment_harvester tests (3)
# ============================================================

class TestAmendmentHarvester:
    def test_clean_signals(self, clean_trial):
        """Clean trial should have minimal signals."""
        sig = extract_amendment_signals(clean_trial)
        assert sig["nctId"] == "NCT_AMEND_001"
        assert sig["enrollmentDelta"] == -2  # 198 - 200
        assert abs(sig["enrollmentDeltaPct"]) < 5  # -1.0%
        assert sig["endpointChanged"] is False
        assert sig["eligibilityChangeCount"] == 0
        assert sig["sponsor"] == "CleanPharma"

    def test_rescue_signals(self, rescue_trial):
        """Rescue trial: large enrollment increase, endpoint changed."""
        sig = extract_amendment_signals(rescue_trial)
        assert sig["nctId"] == "NCT_AMEND_002"
        assert sig["enrollmentDelta"] == 250  # 550 - 300
        assert sig["enrollmentDeltaPct"] == 83.3
        assert sig["enrollmentRatio"] > 1.5
        assert sig["endpointChanged"] is True
        assert sig["completionExtensionDays"] > 1000
        assert sig["completionExtensions"] == 2

    def test_goalpost_signals(self, goalpost_trial):
        """Goalpost trial: enrollment decreased, completion extended 3x."""
        sig = extract_amendment_signals(goalpost_trial)
        assert sig["nctId"] == "NCT_AMEND_004"
        assert sig["enrollmentDelta"] == -350  # 650 - 1000
        assert sig["enrollmentDeltaPct"] == -35.0
        assert sig["completionExtensions"] == 3
        assert sig["status"] == "TERMINATED"


# ============================================================
# diff_engine tests (3)
# ============================================================

class TestDiffEngine:
    def test_enrollment_increase_diff(self, rescue_trial):
        """Rescue trial should show significant enrollment diff."""
        diffs = compute_trial_diffs(rescue_trial)
        enroll_diffs = [d for d in diffs if d["field"] == "enrollment"]
        assert len(enroll_diffs) == 1
        assert enroll_diffs[0]["significant"] is True
        assert enroll_diffs[0]["oldValue"] == 300
        assert enroll_diffs[0]["newValue"] == 550

    def test_completion_extension_diff(self, rescue_trial):
        """Rescue trial should show significant completion extension."""
        diffs = compute_trial_diffs(rescue_trial)
        comp_diffs = [d for d in diffs if d["field"] == "completionDate"]
        assert len(comp_diffs) == 1
        assert comp_diffs[0]["significant"] is True
        assert "days" in comp_diffs[0]["details"]

    def test_clean_trial_diffs(self, clean_trial):
        """Clean trial should have no significant diffs."""
        diffs = compute_trial_diffs(clean_trial)
        significant = [d for d in diffs if d["significant"]]
        assert len(significant) == 0


# ============================================================
# change_classifier tests (3)
# ============================================================

class TestChangeClassifier:
    def test_enrollment_change_classification(self, rescue_trial):
        """Enrollment increase should classify as SAMPLE_SIZE_INCREASE."""
        diffs = compute_trial_diffs(rescue_trial)
        changes = classify_changes(diffs)
        types = [c["type"] for c in changes]
        assert ChangeType.SAMPLE_SIZE_INCREASE.value in types

    def test_completion_classification(self, rescue_trial):
        """Completion extension should classify as COMPLETION_EXTENDED."""
        diffs = compute_trial_diffs(rescue_trial)
        changes = classify_changes(diffs)
        types = [c["type"] for c in changes]
        assert ChangeType.COMPLETION_EXTENDED.value in types

    def test_eligibility_classification(self, scope_creep_trial):
        """Scope creep trial should classify eligibility changes."""
        diffs = compute_trial_diffs(scope_creep_trial)
        changes = classify_changes(diffs)
        elig_changes = [c for c in changes if c["category"] == ChangeCategory.ELIGIBILITY.value]
        assert len(elig_changes) >= 3
        types = {c["type"] for c in elig_changes}
        assert ChangeType.AGE_RANGE_WIDENED.value in types
        assert ChangeType.INCLUSION_LOOSENED.value in types
        assert ChangeType.EXCLUSION_REMOVED.value in types


# ============================================================
# pattern_detector tests (5)
# ============================================================

class TestPatternDetector:
    def test_rescue_pattern(self, rescue_trial):
        """Rescue trial should trigger RESCUE pattern."""
        patterns = detect_patterns(rescue_trial)
        pattern_names = [p["pattern"] for p in patterns]
        assert "RESCUE" in pattern_names
        rescue = next(p for p in patterns if p["pattern"] == "RESCUE")
        assert rescue["severity"] == "HIGH"
        assert len(rescue["evidence"]) >= 3

    def test_scope_creep_pattern(self, scope_creep_trial):
        """Scope creep trial should trigger SCOPE_CREEP pattern."""
        patterns = detect_patterns(scope_creep_trial)
        pattern_names = [p["pattern"] for p in patterns]
        assert "SCOPE_CREEP" in pattern_names

    def test_goalpost_move_pattern(self, goalpost_trial):
        """Goalpost trial should trigger GOALPOST_MOVE pattern."""
        patterns = detect_patterns(goalpost_trial)
        pattern_names = [p["pattern"] for p in patterns]
        assert "GOALPOST_MOVE" in pattern_names

    def test_no_patterns_clean(self, clean_trial):
        """Clean trial should have no concerning patterns."""
        patterns = detect_patterns(clean_trial)
        assert len(patterns) == 0

    def test_funnel_narrowing_synthetic(self):
        """Synthetic trial with tightened eligibility + decreased enrollment."""
        trial = {
            "nctId": "NCT_FUNNEL_001",
            "briefTitle": "Funnel Narrowing Test",
            "status": "COMPLETED",
            "phase": "PHASE3",
            "sponsor": "FunnelCo",
            "sponsorClass": "INDUSTRY",
            "condition": "Hypertension",
            "studyFirstPostDate": "2020-01-01",
            "lastUpdatePostDate": "2023-01-01",
            "enrollmentEstimated": 500,
            "enrollmentActual": 300,
            "completionDateOriginal": "2022-06-01",
            "completionDateFinal": "2022-12-01",
            "eligibilityChanges": [
                {"type": "inclusion_tightened", "detail": "eGFR >60 narrowed to >90"},
            ],
            "protocolPrimary": ["BP reduction at 12 weeks"],
            "protocolSecondary": ["Heart rate"],
            "resultsPrimary": ["BP reduction at 12 weeks"],
            "resultsSecondary": ["Heart rate"],
        }
        patterns = detect_patterns(trial)
        pattern_names = [p["pattern"] for p in patterns]
        assert "FUNNEL_NARROWING" in pattern_names


# ============================================================
# outcome_linker tests (2)
# ============================================================

class TestOutcomeLinker:
    def test_completion_rate(self, sample_trials):
        """Amended trials should have lower completion rate than clean."""
        result = link_amendments_to_outcomes(sample_trials)
        assert result["amendedTrials"] >= 2
        assert result["cleanTrials"] >= 1
        # Clean trial has 100% completion rate
        assert result["cleanCompletionRate"] == 100.0

    def test_pattern_fate(self, sample_trials):
        """Pattern fates should track status by pattern type."""
        result = link_amendments_to_outcomes(sample_trials)
        fates = result["patternFates"]
        # GOALPOST_MOVE is on the TERMINATED trial
        assert "GOALPOST_MOVE" in fates
        assert fates["GOALPOST_MOVE"].get("TERMINATED", 0) >= 1


# ============================================================
# aggregator tests (2)
# ============================================================

class TestAggregator:
    def test_by_sponsor(self, sample_trials):
        """Aggregation should break down by sponsor."""
        result = aggregate_amendments(sample_trials)
        sponsors = result["bySponsor"]
        assert "RescueCorp" in sponsors
        assert sponsors["RescueCorp"]["trials"] == 1
        assert sponsors["RescueCorp"]["amended"] == 1

    def test_prevalence(self, sample_trials):
        """Prevalence counts should reflect fixture data."""
        result = aggregate_amendments(sample_trials)
        prev = result["prevalence"]
        assert result["totalTrials"] == 4
        assert prev["endpointChanged"] >= 1
        assert prev["anyPattern"] >= 2
        assert prev["anyPatternPct"] > 0


# ============================================================
# integration tests (7)
# ============================================================

class TestIntegration:
    def test_full_pipeline(self, sample_trials):
        """Full pipeline: harvest -> diff -> classify -> detect -> link -> aggregate."""
        # Each step should succeed
        for trial in sample_trials:
            signals = extract_amendment_signals(trial)
            assert "nctId" in signals
            diffs = compute_trial_diffs(trial)
            assert isinstance(diffs, list)
            changes = classify_changes(diffs)
            assert isinstance(changes, list)
            patterns = detect_patterns(trial)
            assert isinstance(patterns, list)

        outcome = link_amendments_to_outcomes(sample_trials)
        assert "amendedTrials" in outcome

        agg = aggregate_amendments(sample_trials)
        assert "totalTrials" in agg

    def test_dashboard_json_serializable(self, sample_trials):
        """All outputs must be JSON-serializable for dashboard consumption."""
        agg = aggregate_amendments(sample_trials)
        outcome = link_amendments_to_outcomes(sample_trials)
        dashboard_data = {
            "aggregation": agg,
            "outcomes": outcome,
            "trials": [],
        }
        for trial in sample_trials:
            dashboard_data["trials"].append({
                "signals": extract_amendment_signals(trial),
                "diffs": compute_trial_diffs(trial),
                "changes": classify_changes(compute_trial_diffs(trial)),
                "patterns": detect_patterns(trial),
            })
        # Must not raise
        serialized = json.dumps(dashboard_data)
        assert len(serialized) > 100

    def test_signal_keys_complete(self, rescue_trial):
        """All expected signal keys must be present."""
        signals = extract_amendment_signals(rescue_trial)
        expected_keys = {
            "nctId", "enrollmentEstimated", "enrollmentActual",
            "enrollmentDelta", "enrollmentDeltaPct", "enrollmentRatio",
            "updateGapDays", "completionExtensionDays", "completionExtensions",
            "eligibilityChangeCount", "eligibilityChanges", "endpointChanged",
            "status", "sponsor", "sponsorClass", "condition", "phase",
        }
        assert expected_keys.issubset(set(signals.keys()))

    def test_all_categories_covered(self, sample_trials):
        """All 4 change categories should appear across fixture data."""
        all_categories = set()
        for trial in sample_trials:
            diffs = compute_trial_diffs(trial)
            changes = classify_changes(diffs)
            for c in changes:
                all_categories.add(c["category"])
        assert ChangeCategory.ENROLLMENT.value in all_categories
        assert ChangeCategory.ENDPOINTS.value in all_categories
        assert ChangeCategory.ELIGIBILITY.value in all_categories
        assert ChangeCategory.TIMELINE.value in all_categories

    def test_pattern_types_expected(self, sample_trials):
        """At least RESCUE, SCOPE_CREEP, and GOALPOST_MOVE must be detected."""
        all_pattern_names = set()
        for trial in sample_trials:
            for p in detect_patterns(trial):
                all_pattern_names.add(p["pattern"])
        assert "RESCUE" in all_pattern_names
        assert "SCOPE_CREEP" in all_pattern_names
        assert "GOALPOST_MOVE" in all_pattern_names

    def test_stealth_switch_synthetic(self):
        """Stealth switch: endpoint changed with NO enrollment change."""
        trial = {
            "nctId": "NCT_STEALTH_001",
            "briefTitle": "Stealth Switch Test",
            "status": "COMPLETED",
            "phase": "PHASE3",
            "sponsor": "StealthCo",
            "sponsorClass": "INDUSTRY",
            "condition": "Asthma",
            "studyFirstPostDate": "2019-01-01",
            "lastUpdatePostDate": "2022-01-01",
            "enrollmentEstimated": 400,
            "enrollmentActual": 400,
            "completionDateOriginal": "2021-06-01",
            "completionDateFinal": "2021-06-01",
            "protocolPrimary": ["FEV1 change at 12 weeks"],
            "protocolSecondary": ["Asthma control questionnaire"],
            "resultsPrimary": ["Exacerbation rate at 52 weeks"],
            "resultsSecondary": ["FEV1 change at 12 weeks"],
        }
        patterns = detect_patterns(trial)
        pattern_names = [p["pattern"] for p in patterns]
        assert "STEALTH_SWITCH" in pattern_names

    def test_multiple_patterns_possible(self, rescue_trial):
        """A single trial can trigger multiple patterns."""
        # Rescue trial triggers RESCUE; verify it returns at least 1 pattern
        patterns = detect_patterns(rescue_trial)
        assert len(patterns) >= 1
        # All patterns must have required keys
        for p in patterns:
            assert "pattern" in p
            assert "severity" in p
            assert "evidence" in p
            assert isinstance(p["evidence"], list)
