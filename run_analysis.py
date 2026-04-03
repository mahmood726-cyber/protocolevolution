"""Run the ProtocolEvolution pipeline on fixture data and produce JSON output."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.amendment_harvester import extract_amendment_signals
from src.diff_engine import compute_trial_diffs
from src.change_classifier import classify_changes
from src.pattern_detector import detect_patterns
from src.outcome_linker import link_amendments_to_outcomes
from src.aggregator import aggregate_amendments


def main():
    # Load fixture data
    fixtures_path = os.path.join(
        os.path.dirname(__file__), "data", "fixtures", "sample_amendments.json"
    )
    with open(fixtures_path, "r", encoding="utf-8") as f:
        trials = json.load(f)

    print(f"Loaded {len(trials)} trials from fixtures.\n")

    # Per-trial analysis
    all_trial_results = []
    for trial in trials:
        signals = extract_amendment_signals(trial)
        diffs = compute_trial_diffs(trial)
        changes = classify_changes(diffs)
        patterns = detect_patterns(trial)

        result = {
            "nctId": signals["nctId"],
            "signals": signals,
            "diffs": diffs,
            "changes": changes,
            "patterns": patterns,
        }
        all_trial_results.append(result)

        print(f"--- {signals['nctId']} ---")
        print(f"  Enrollment: {signals['enrollmentEstimated']} -> {signals['enrollmentActual']}"
              f" (delta {signals['enrollmentDeltaPct']}%)")
        print(f"  Completion extension: {signals['completionExtensionDays']} days")
        print(f"  Endpoint changed: {signals['endpointChanged']}")
        print(f"  Eligibility changes: {signals['eligibilityChangeCount']}")
        print(f"  Diffs: {len(diffs)} ({sum(1 for d in diffs if d['significant'])} significant)")
        print(f"  Changes: {len(changes)}")
        if patterns:
            print(f"  PATTERNS: {', '.join(p['pattern'] for p in patterns)}")
        else:
            print("  PATTERNS: none")
        print()

    # Cross-trial analysis
    outcomes = link_amendments_to_outcomes(trials)
    aggregation = aggregate_amendments(trials)

    print("=== OUTCOME LINKAGE ===")
    print(f"  Amended trials: {outcomes['amendedTrials']}")
    print(f"  Clean trials: {outcomes['cleanTrials']}")
    print(f"  Amended completion rate: {outcomes['amendedCompletionRate']}%")
    print(f"  Clean completion rate: {outcomes['cleanCompletionRate']}%")
    print(f"  Delta: {outcomes['completionRateDelta']} pp")
    print(f"  Pattern fates: {json.dumps(outcomes['patternFates'], indent=2)}")
    print()

    print("=== AGGREGATION ===")
    print(f"  Total trials: {aggregation['totalTrials']}")
    print(f"  Prevalence: {json.dumps(aggregation['prevalence'], indent=2)}")
    print()

    # Write output JSON for dashboard
    output = {
        "trials": all_trial_results,
        "outcomes": outcomes,
        "aggregation": aggregation,
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "data", "processed", "analysis_output.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to {out_path}")


if __name__ == "__main__":
    main()
