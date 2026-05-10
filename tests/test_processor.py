"""
test_processor.py — Tests for the pandas processing pipeline.

Demonstrates:
- @pytest.mark.parametrize: same transformation logic tested against
  multiple distinct DataFrame shapes (empty-ish repo, mega-popular repo,
  abandoned repo with no issues).
- Direct assertion on dtype downcasting to prove memory optimization works.
"""

import numpy as np
import pandas as pd
import pytest

from source.processor import (
    build_dataframe,
    clean_repo_metrics,
    optimize_memory,
    prepare_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable raw record shapes
# ---------------------------------------------------------------------------

RECORD_NEW_REPO = {
    "repo": "alice/brand-new",
    "stars": 5,
    "forks": 0,
    "open_issues": 1,
    "total_issues": 1,
    "contributor_count": 1,
    "weekly_commits": [0] * 51 + [3],
    "repo_age_days": 7,
    "language": "Python",
    "description": "Just started",
    "html_url": "https://github.com/alice/brand-new",
}

RECORD_MEGA_POPULAR = {
    "repo": "pytorch/pytorch",
    "stars": 89_000,
    "forks": 24_000,
    "open_issues": 800,
    "total_issues": 15_000,
    "contributor_count": 3_200,
    "weekly_commits": [120] * 52,
    "repo_age_days": 2920,
    "language": "C++",
    "description": "Deep learning framework",
    "html_url": "https://github.com/pytorch/pytorch",
}

RECORD_ABANDONED = {
    "repo": "ghost/dead-project",
    "stars": 2_100,
    "forks": 40,
    "open_issues": 300,
    "total_issues": 300,
    "contributor_count": 2,
    "weekly_commits": [0] * 52,
    "repo_age_days": 1460,
    "language": "JavaScript",
    "description": "",
    "html_url": "https://github.com/ghost/dead-project",
}


# ---------------------------------------------------------------------------
# build_dataframe tests
# ---------------------------------------------------------------------------

def test_build_dataframe_raises_on_empty_input():
    """build_dataframe must raise ValueError when given an empty list."""
    with pytest.raises(ValueError, match="empty"):
        build_dataframe([])


def test_build_dataframe_creates_correct_columns():
    """All expected columns should be present after building."""
    df = build_dataframe([RECORD_NEW_REPO])
    for col in ["stars", "forks", "open_issues", "total_issues",
                "contributor_count", "repo_age_days"]:
        assert col in df.columns, f"Missing expected column: {col}"


# ---------------------------------------------------------------------------
# clean_repo_metrics tests — parametrized across three repo archetypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("record,expected_verdict_feature", [
    (
        RECORD_NEW_REPO,
        "star_velocity",   # Velocity should be computable even for tiny repos.
    ),
    (
        RECORD_MEGA_POPULAR,
        "issue_resolution_rate",  # Large repo with lots of closed issues.
    ),
    (
        RECORD_ABANDONED,
        "fork_substance_ratio",   # Non-zero forks relative to stars.
    ),
])
def test_clean_repo_metrics_derives_features(record, expected_verdict_feature):
    """
    After cleaning, each derived feature column must exist and be a finite float.
    We parametrize to ensure the logic handles the full range of real-world
    repo shapes: tiny new repos, massive active projects, and abandoned ones.
    """
    df = build_dataframe([record])
    cleaned = clean_repo_metrics(df)
    assert expected_verdict_feature in cleaned.columns
    assert np.isfinite(cleaned[expected_verdict_feature].iloc[0]), (
        f"{expected_verdict_feature} should be a finite number, "
        f"got {cleaned[expected_verdict_feature].iloc[0]}"
    )


@pytest.mark.parametrize("record,expected_rate", [
    (RECORD_NEW_REPO, 0.0),       # 1 open / 1 total → 0 % resolved.
    (RECORD_MEGA_POPULAR, pytest.approx(0.9467, abs=0.01)),  # 14200/15000.
    (RECORD_ABANDONED, 0.0),      # All issues are open → 0 % resolved.
])
def test_issue_resolution_rate(record, expected_rate):
    """
    Issue resolution rate must match the hand-computed expected value for each
    repo archetype, proving the formula (closed / total) is applied correctly.
    """
    df = build_dataframe([record])
    cleaned = clean_repo_metrics(df)
    actual = float(cleaned["issue_resolution_rate"].iloc[0])
    if isinstance(expected_rate, float):
        assert actual == pytest.approx(expected_rate, abs=0.01)
    else:
        assert actual == expected_rate


# ---------------------------------------------------------------------------
# optimize_memory tests
# ---------------------------------------------------------------------------

def test_optimize_memory_downcasts_integers():
    """Integer count columns must be downcast from int64 to int32."""
    df = build_dataframe([RECORD_MEGA_POPULAR])
    cleaned = clean_repo_metrics(df)
    optimized = optimize_memory(cleaned)
    for col in ["stars", "forks", "open_issues"]:
        assert optimized[col].dtype == np.int32, (
            f"Expected int32 for '{col}', got {optimized[col].dtype}"
        )


def test_optimize_memory_downcasts_floats():
    """Derived float columns must be downcast from float64 to float32."""
    df = build_dataframe([RECORD_MEGA_POPULAR])
    cleaned = clean_repo_metrics(df)
    optimized = optimize_memory(cleaned)
    for col in ["star_velocity", "issue_resolution_rate", "fork_substance_ratio"]:
        assert optimized[col].dtype == np.float32, (
            f"Expected float32 for '{col}', got {optimized[col].dtype}"
        )


# ---------------------------------------------------------------------------
# prepare_pipeline integration test
# ---------------------------------------------------------------------------

def test_prepare_pipeline_end_to_end():
    """
    The full pipeline must produce a DataFrame with the correct shape and types
    without raising any exceptions on a multi-repo input.
    """
    records = [RECORD_NEW_REPO, RECORD_MEGA_POPULAR, RECORD_ABANDONED]
    df = prepare_pipeline(records)
    assert len(df) == 3
    assert "star_velocity" in df.columns
    assert "commit_consistency" in df.columns
    assert df["stars"].dtype == np.int32
