"""Shared test fixtures for ProtocolEvolution."""

import json
import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures")


@pytest.fixture
def sample_trials():
    """Load sample amendment fixtures."""
    path = os.path.join(FIXTURES_DIR, "sample_amendments.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def clean_trial(sample_trials):
    """NCT_AMEND_001 — Clean trial, no amendments."""
    return sample_trials[0]


@pytest.fixture
def rescue_trial(sample_trials):
    """NCT_AMEND_002 — Rescue pattern."""
    return sample_trials[1]


@pytest.fixture
def scope_creep_trial(sample_trials):
    """NCT_AMEND_003 — Scope creep pattern."""
    return sample_trials[2]


@pytest.fixture
def goalpost_trial(sample_trials):
    """NCT_AMEND_004 — Goalpost move pattern."""
    return sample_trials[3]
