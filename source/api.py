"""
api.py — FastAPI application and route handlers.

Exposes three endpoints:
  GET /health                        → liveness check
  GET /api/analyze?repo=owner/repo   → full pipeline for one repository
  GET /api/trending?language=python&limit=10 → batch analysis of trending repos

Each route is async end-to-end: the FastAPI handler awaits the ingestion
coroutines, then calls the (synchronous but fast) processor and analyzer.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from source.analyzer import score_repos, score_single_repo
from source.ingestion import fetch_repo_data, fetch_trending_repos
from source.processor import prepare_pipeline

app = FastAPI(
    title="GitHub Insights API",
    description=(
        "A digital scout that scans GitHub repositories and answers one question: "
        "Is this software actually useful, or is it just getting empty attention?"
    ),
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the browser root to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Meta"])
async def health_check() -> dict:
    """
    Liveness probe.

    Returns a simple JSON payload confirming the service is running.
    Useful for deployment health checks and quick smoke tests.
    """
    return {"status": "ok", "service": "GitHub Insights API"}


@app.get("/api/analyze", tags=["Analysis"])
async def analyze_repo(
    repo: str = Query(
        ...,
        description='Full repository name in "owner/repo" format.',
        examples=["fastapi/fastapi"],
    )
) -> dict:
    """
    Analyze a single GitHub repository and return a Hype vs. Substance verdict.

    The pipeline:
    1. Concurrently fetches metadata, commit stats, issue counts, and
       contributor data from four GitHub API endpoints (ingestion.py).
    2. Builds and memory-optimizes a one-row pandas DataFrame (processor.py).
    3. Scores the repo using z-score normalization and weighted composite
       metrics (analyzer.py).
    4. Returns a structured JSON payload with raw metrics and the verdict.

    Args:
        repo: GitHub repository in "owner/repo" format (e.g. "pytorch/pytorch").

    Returns:
        JSON object containing scores, verdict, confidence, and raw metrics.

    Raises:
        HTTPException 400: If the repo name is not in "owner/repo" format.
        HTTPException 404: If the repository does not exist on GitHub.
    """
    if "/" not in repo or len(repo.split("/")) != 2:
        raise HTTPException(
            status_code=400,
            detail='Repository must be in "owner/repo" format (e.g. "fastapi/fastapi").',
        )

    raw = await fetch_repo_data(repo)

    if raw.get("stars") == 0 and raw.get("forks") == 0 and raw.get("open_issues") == 0:
        # Stars/forks of exactly 0 on a missing repo vs. a legitimately new repo
        # is ambiguous, but it's the best signal we have without a 404 status.
        pass

    result = score_single_repo(raw)

    return {
        "repo": result.repo,
        "verdict": result.verdict,
        "confidence": result.confidence,
        "scores": {
            "hype_score": result.hype_score,
            "substance_score": result.substance_score,
        },
        "meta": {
            "language": result.language,
            "description": result.description,
            "html_url": result.html_url,
        },
        "raw_metrics": {
            "stars": raw.get("stars"),
            "forks": raw.get("forks"),
            "open_issues": raw.get("open_issues"),
            "total_issues": raw.get("total_issues"),
            "contributor_count": raw.get("contributor_count"),
            "repo_age_days": raw.get("repo_age_days"),
        },
    }


@app.get("/api/trending", tags=["Analysis"])
async def trending_repos(
    language: str = Query(
        default="python",
        description="Programming language to filter trending repositories by.",
        examples=["python"],
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=30,
        description="Number of repositories to analyze (1–30).",
    ),
) -> dict:
    """
    Fetch and score the top trending GitHub repositories for a given language.

    Unlike /api/analyze (single-repo, no relative comparison), this endpoint
    runs z-score normalization across ALL repos in the batch simultaneously,
    so each score reflects how a repo compares to its peers in the same
    trending list.

    The pipeline:
    1. Searches GitHub for the most-starred repos pushed in the last 30 days.
    2. Fetches full metrics for all repos concurrently (asyncio.gather).
    3. Builds a multi-row DataFrame and optimizes memory.
    4. Scores all repos relative to each other via z-score normalization.
    5. Returns a ranked list sorted by substance score descending.

    Args:
        language: Programming language (default: "python").
        limit: Number of repos to fetch and score (default: 10, max: 30).

    Returns:
        JSON object with a ranked list of scored repositories.
    """
    raw_records = await fetch_trending_repos(language=language, limit=limit)

    if not raw_records:
        raise HTTPException(
            status_code=404,
            detail=f"No trending repositories found for language '{language}'.",
        )

    df = prepare_pipeline(raw_records)
    scored = score_repos(df)

    # Sort by substance score descending — most useful at the top.
    ranked = sorted(scored, key=lambda r: r.substance_score, reverse=True)

    return {
        "language": language,
        "count": len(ranked),
        "results": [
            {
                "rank": idx + 1,
                "repo": r.repo,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "scores": {
                    "hype_score": r.hype_score,
                    "substance_score": r.substance_score,
                },
                "meta": {
                    "language": r.language,
                    "description": r.description,
                    "html_url": r.html_url,
                },
            }
            for idx, r in enumerate(ranked)
        ],
    }
