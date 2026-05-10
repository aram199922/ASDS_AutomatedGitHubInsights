"""
test_ingestion.py — Tests for the async GitHub API fetching layer.

Demonstrates unittest.mock usage: every aiohttp network call is intercepted
and replaced with a fake response object. This means:
- Tests run instantly (no real HTTP requests).
- Tests are deterministic (no flakiness from rate limits or network issues).
- We can simulate GitHub returning partial/empty data gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from source.ingestion import _compute_age_days, fetch_repo_data, fetch_trending_repos


# ---------------------------------------------------------------------------
# Helpers — build fake aiohttp response objects
# ---------------------------------------------------------------------------

def _make_response(status: int, json_data) -> MagicMock:
    """
    Create a mock that behaves like an aiohttp response context manager.

    The aiohttp pattern is:
        async with session.get(url) as response:
            data = await response.json()

    We need both __aenter__/__aexit__ (for `async with`) and an async .json().
    """
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data)

    # Make it usable as `async with session.get(...) as response`
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Fake GitHub API payloads
# ---------------------------------------------------------------------------

FAKE_METADATA = {
    "stargazers_count": 12_000,
    "forks_count": 1_500,
    "open_issues_count": 45,
    "closed_issues": 300,
    "created_at": "2021-06-01T00:00:00Z",
    "language": "Python",
    "description": "A great library",
    "html_url": "https://github.com/test/repo",
}

FAKE_COMMIT_STATS = [{"total": i * 5} for i in range(52)]

FAKE_OPEN_ISSUES = [{"id": 1, "title": "Bug fix"}]

FAKE_CONTRIBUTORS = [{"login": f"user{i}"} for i in range(20)]


# ---------------------------------------------------------------------------
# fetch_repo_data tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_repo_data_returns_expected_fields():
    """
    fetch_repo_data must correctly map GitHub API JSON into our metric dict.

    All four API endpoints are mocked via unittest.mock.patch so no real
    network requests are made — the test completes instantly.
    """
    responses = [
        _make_response(200, FAKE_METADATA),
        _make_response(200, FAKE_COMMIT_STATS),
        _make_response(200, FAKE_OPEN_ISSUES),
        _make_response(200, FAKE_CONTRIBUTORS),
    ]

    call_count = 0

    def side_effect(url, headers=None):
        nonlocal call_count
        resp = responses[call_count % len(responses)]
        call_count += 1
        return resp

    with patch("aiohttp.ClientSession.get", side_effect=side_effect):
        result = await fetch_repo_data("test/repo")

    assert result["repo"] == "test/repo"
    assert result["stars"] == 12_000
    assert result["forks"] == 1_500
    assert result["contributor_count"] == 20
    assert result["language"] == "Python"
    assert isinstance(result["weekly_commits"], list)
    assert len(result["weekly_commits"]) == 52


@pytest.mark.asyncio
async def test_fetch_repo_data_handles_empty_api_responses():
    """
    When GitHub returns non-200 (e.g. 202 for stats not ready), fetch_repo_data
    must degrade gracefully and return zero-value defaults rather than crashing.
    """
    responses = [
        _make_response(200, FAKE_METADATA),
        _make_response(202, {}),   # Commit stats not ready
        _make_response(200, []),
        _make_response(200, []),
    ]

    call_count = 0

    def side_effect(url, headers=None):
        nonlocal call_count
        resp = responses[call_count % len(responses)]
        call_count += 1
        return resp

    with patch("aiohttp.ClientSession.get", side_effect=side_effect):
        result = await fetch_repo_data("test/repo")

    # weekly_commits should be empty list, not an exception.
    assert result["weekly_commits"] == []
    assert result["contributor_count"] == 0


@pytest.mark.asyncio
async def test_fetch_trending_repos_returns_list():
    """
    fetch_trending_repos must return a list of metric dicts.

    The GitHub search API call is mocked via aiohttp, and each individual
    fetch_repo_data coroutine is replaced with an AsyncMock so the test
    runs instantly without any real network I/O. Mocking at this level also
    avoids dealing with interleaved session calls across concurrent tasks.
    """
    fake_search = {
        "items": [
            {"full_name": f"org/repo{i}"} for i in range(3)
        ]
    }

    fake_metric = {
        "repo": "org/repo0",
        "stars": 5000,
        "forks": 200,
        "open_issues": 10,
        "total_issues": 100,
        "contributor_count": 30,
        "weekly_commits": [10] * 52,
        "repo_age_days": 500,
        "language": "Python",
        "description": "Test",
        "html_url": "https://github.com/org/repo0",
    }

    search_response = _make_response(200, fake_search)

    with patch("aiohttp.ClientSession.get", return_value=search_response), \
         patch(
             "source.ingestion.fetch_repo_data",
             new=AsyncMock(return_value=fake_metric),
         ):
        results = await fetch_trending_repos(language="python", limit=3)

    assert isinstance(results, list)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# _compute_age_days unit tests
# ---------------------------------------------------------------------------

def test_compute_age_days_known_date():
    """A repo created about 5 years ago should report > 1800 days."""
    age = _compute_age_days("2021-01-01T00:00:00Z")
    assert age > 1800, f"Expected > 1800 days, got {age}"


def test_compute_age_days_none_returns_one():
    """Missing created_at must return 1 (avoids division by zero downstream)."""
    assert _compute_age_days(None) == 1
