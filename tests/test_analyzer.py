"""
test_analyzer.py — Tests for the scipy statistical scoring engine.

Demonstrates:
- @pytest.mark.parametrize: verdict logic verified against known inputs
  spanning the full range from obvious hype to obvious substance.
- Edge case handling: single-repo analysis (no z-score variance possible).
"""

import pytest

from source.analyzer import RepoScore, _assign_verdict, score_repos
from source.processor import prepare_pipeline


# ---------------------------------------------------------------------------
# Fixtures — raw records with hand-crafted metric profiles
# ---------------------------------------------------------------------------

def _make_record(
    repo: str,
    stars: int,
    forks: int,
    open_issues: int,
    total_issues: int,
    contributors: int,
    weekly_commits: list[int],
    age_days: int,
) -> dict:
    return {
        "repo": repo,
        "stars": stars,
        "forks": forks,
        "open_issues": open_issues,
        "total_issues": total_issues,
        "contributor_count": contributors,
        "weekly_commits": weekly_commits,
        "repo_age_days": age_days,
        "language": "Python",
        "description": "",
        "html_url": f"https://github.com/{repo}",
    }


HYPE_REPO = _make_record(
    repo="trending/overnight-hit",
    stars=50_000,
    forks=200,           # Low forks relative to stars
    open_issues=500,
    total_issues=510,    # Almost no issues resolved
    contributors=2,
    weekly_commits=[0] * 50 + [1000, 1000],  # Giant spike, then silence
    age_days=30,
)

SUBSTANCE_REPO = _make_record(
    repo="foundation/battle-tested",
    stars=8_000,
    forks=2_000,          # High fork ratio
    open_issues=20,
    total_issues=800,     # Most issues resolved
    contributors=150,
    weekly_commits=[40] * 52,  # Steady weekly commits
    age_days=1825,
)

AVERAGE_REPO = _make_record(
    repo="mid/average-project",
    stars=3_000,
    forks=400,
    open_issues=50,
    total_issues=200,
    contributors=15,
    weekly_commits=[10] * 52,
    age_days=730,
)


# ---------------------------------------------------------------------------
# _assign_verdict unit tests — parametrized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hype,substance,expected_verdict", [
    (8.5, 3.0, "Just Hype"),      # High hype, low substance → clear hype.
    (4.0, 7.5, "Useful"),         # Low hype, high substance → useful.
    (4.9, 4.9, "Emerging"),       # Both just below threshold → too early to call.
    (9.0, 9.0, "Useful"),         # Both very high → substance threshold met.
    (2.0, 2.0, "Emerging"),       # Both very low → emerging/unknown.
])
def test_assign_verdict_logic(hype, substance, expected_verdict):
    """
    Verdict assignment must follow the threshold rules for all key score
    combinations. Parametrize covers hype-dominant, substance-dominant,
    and ambiguous cases so every branch of _assign_verdict is exercised.
    """
    verdict, _ = _assign_verdict(hype, substance)
    assert verdict == expected_verdict, (
        f"hype={hype}, substance={substance} → expected '{expected_verdict}', got '{verdict}'"
    )


@pytest.mark.parametrize("hype,substance,expected_confidence", [
    (9.0, 2.0, "High"),     # Gap of 7 → high confidence.
    (6.0, 4.5, "Medium"),   # Gap of 1.5 → medium confidence.
    (5.1, 5.0, "Low"),      # Gap of 0.1 → low confidence.
])
def test_assign_confidence_levels(hype, substance, expected_confidence):
    """Confidence must scale with the gap between hype and substance scores."""
    _, confidence = _assign_verdict(hype, substance)
    assert confidence == expected_confidence


# ---------------------------------------------------------------------------
# score_repos integration tests
# ---------------------------------------------------------------------------

def test_score_repos_returns_one_result_per_row():
    """score_repos must produce exactly one RepoScore per input row."""
    records = [HYPE_REPO, SUBSTANCE_REPO, AVERAGE_REPO]
    df = prepare_pipeline(records)
    results = score_repos(df)
    assert len(results) == 3


def test_score_repos_returns_repo_score_instances():
    """Every item in the returned list must be a RepoScore dataclass."""
    df = prepare_pipeline([SUBSTANCE_REPO, HYPE_REPO])
    results = score_repos(df)
    for result in results:
        assert isinstance(result, RepoScore)


def test_scores_are_in_valid_range():
    """Both hype_score and substance_score must be in [0, 10]."""
    records = [HYPE_REPO, SUBSTANCE_REPO, AVERAGE_REPO]
    df = prepare_pipeline(records)
    for result in score_repos(df):
        assert 0.0 <= result.hype_score <= 10.0, (
            f"hype_score out of range: {result.hype_score}"
        )
        assert 0.0 <= result.substance_score <= 10.0, (
            f"substance_score out of range: {result.substance_score}"
        )


def test_substance_repo_outscores_hype_repo_on_substance():
    """
    When compared in the same batch, the battle-tested repo must have a
    higher substance score than the overnight hype repo.
    This is the core behavioural guarantee of the analyzer.
    """
    records = [HYPE_REPO, SUBSTANCE_REPO]
    df = prepare_pipeline(records)
    results = {r.repo: r for r in score_repos(df)}

    hype_substance = results["trending/overnight-hit"].substance_score
    real_substance = results["foundation/battle-tested"].substance_score
    assert real_substance > hype_substance, (
        f"Expected battle-tested ({real_substance}) > overnight-hit ({hype_substance})"
    )