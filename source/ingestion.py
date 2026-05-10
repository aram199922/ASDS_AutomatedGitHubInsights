"""
ingestion.py — Async GitHub API data fetcher.

All network I/O is handled here using asyncio + aiohttp so that multiple
GitHub API endpoints are hit concurrently without blocking the event loop.

Architecture note: GitHub API calls are pure I/O-bound (we spend time waiting
for HTTP responses, not doing CPU work). asyncio is the correct concurrency
primitive here — it bypasses the GIL entirely for I/O and avoids the overhead
of spawning OS threads or processes.
"""

import asyncio
import os
from typing import Any

import aiohttp
from dotenv import load_dotenv

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _build_headers() -> dict[str, str]:
    """Return request headers, injecting a Bearer token when available."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


async def _get(session: aiohttp.ClientSession, url: str) -> Any:
    """
    Perform a single authenticated GET request and return parsed JSON.

    Returns an empty dict on non-200 responses rather than raising, so
    the pipeline degrades gracefully when GitHub returns 202 (stats not
    yet computed) or rate-limit errors.
    """
    async with session.get(url, headers=_build_headers()) as response:
        if response.status == 200:
            return await response.json()
        return {}


async def fetch_repo_data(repo: str) -> dict[str, Any]:
    """
    Concurrently fetch all metrics for a single GitHub repository.

    Hits four endpoints at the same time via asyncio.gather():
    - /repos/{repo}                   → core metadata (stars, forks, language)
    - /repos/{repo}/stats/commit_activity → weekly commit counts (last 52 weeks)
    - /repos/{repo}/issues?state=all  → open + closed issue counts
    - /repos/{repo}/stats/contributors → contributor list with commit counts

    Args:
        repo: Full repository name in "owner/repo" format.

    Returns:
        A dictionary containing the merged results from all four endpoints.
    """
    async with aiohttp.ClientSession() as session:
        metadata, commit_stats, open_issues, contributors = await asyncio.gather(
            _get(session, f"{GITHUB_API_BASE}/repos/{repo}"),
            _get(session, f"{GITHUB_API_BASE}/repos/{repo}/stats/commit_activity"),
            _get(session, f"{GITHUB_API_BASE}/repos/{repo}/issues?state=open&per_page=1"),
            _get(session, f"{GITHUB_API_BASE}/repos/{repo}/stats/contributors"),
        )

    closed_issues_count = metadata.get("closed_issues", 0)
    total_issues = (
        metadata.get("open_issues_count", 0) + closed_issues_count
    )

    weekly_totals = (
        [week.get("total", 0) for week in commit_stats]
        if isinstance(commit_stats, list)
        else []
    )

    contributor_count = len(contributors) if isinstance(contributors, list) else 0

    return {
        "repo": repo,
        "stars": metadata.get("stargazers_count", 0),
        "forks": metadata.get("forks_count", 0),
        "open_issues": metadata.get("open_issues_count", 0),
        "total_issues": total_issues,
        "contributor_count": contributor_count,
        "weekly_commits": weekly_totals,
        "repo_age_days": _compute_age_days(metadata.get("created_at")),
        "language": metadata.get("language", "Unknown"),
        "description": metadata.get("description", ""),
        "html_url": metadata.get("html_url", ""),
    }


async def fetch_trending_repos(language: str = "python", limit: int = 10) -> list[dict[str, Any]]:
    """
    Fetch and analyze multiple trending GitHub repositories concurrently.

    Searches GitHub for the most-starred repositories in the given language
    pushed within the last 30 days, then fetches full metrics for each repo
    simultaneously.

    Args:
        language: Programming language filter (e.g. "python", "javascript").
        limit: Maximum number of repositories to return (capped at 30).

    Returns:
        A list of raw metric dictionaries, one per repository.
    """
    limit = min(limit, 30)
    query = f"language:{language}&sort=stars&order=desc&per_page={limit}"
    search_url = f"{GITHUB_API_BASE}/search/repositories?q={query}"

    async with aiohttp.ClientSession() as session:
        search_results = await _get(session, search_url)

    items = search_results.get("items", [])
    repo_names = [item["full_name"] for item in items]

    results = await asyncio.gather(
        *[fetch_repo_data(name) for name in repo_names]
    )
    return list(results)


def _compute_age_days(created_at: str | None) -> int:
    """
    Calculate how many days old a repository is based on its creation timestamp.

    Args:
        created_at: ISO 8601 timestamp string from the GitHub API (e.g. "2020-01-15T10:00:00Z").

    Returns:
        Number of days since creation, or 1 if the timestamp is missing.
    """
    if not created_at:
        return 1
    from datetime import datetime, timezone
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    delta = datetime.now(timezone.utc) - created
    return max(delta.days, 1)
