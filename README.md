# ASDS — Automated GitHub Insights & Predictive Analytics API

A "digital scout" that scans GitHub repositories and answers one question:

> **Is this new software actually useful, or is it just getting empty attention?**

It ingests data from the GitHub API concurrently, processes it with memory-efficient pandas, scores it using scipy statistical analysis, and serves the results through a FastAPI endpoint as a structured JSON verdict.

---

## Architecture

```
source/
├── ingestion.py   — async GitHub API fetching (asyncio + aiohttp)
├── processor.py   — pandas cleaning + memory optimization (dtype downcasting)
├── analyzer.py    — scipy z-score scoring → Hype Score / Substance Score
└── api.py         — FastAPI endpoints

tests/
├── test_ingestion.py  — unittest.mock fakes all HTTP calls
├── test_processor.py  — @pytest.mark.parametrize across 3 repo archetypes
└── test_analyzer.py   — verdict logic + integration tests
```

### Pipeline Flow

```
User Request
    │
    ▼
api.py (FastAPI)
    │
    ▼
ingestion.py ──► asyncio.gather() hits 4 GitHub endpoints simultaneously
    │               /repos/{repo}
    │               /repos/{repo}/stats/commit_activity
    │               /repos/{repo}/issues
    │               /repos/{repo}/stats/contributors
    ▼
processor.py ──► build DataFrame → derive features → downcast dtypes
    │               star_velocity, commit_consistency,
    │               issue_resolution_rate, fork_substance_ratio
    ▼
analyzer.py ──► scipy.stats.zscore → weighted scores → verdict
    │               Hype Score   (0–10)
    │               Substance Score (0–10)
    │               Verdict: "Useful" | "Just Hype" | "Emerging"
    ▼
JSON Response
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/ASDS_AutomatedGitHubInsights.git
cd ASDS_AutomatedGitHubInsights
python -m venv venv
```

Activate it:
- Windows: `.\venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure GitHub token (optional but recommended)

Without a token you get 60 API requests/hour. With a token: 5,000/hour.

```bash
cp .env.example .env
# Edit .env and add your token:
# GITHUB_TOKEN=ghp_yourtoken
```

Create a token at <https://github.com/settings/tokens> — no scopes needed for public repos.

### 4. Start the API

```bash
uvicorn source.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive docs: `http://127.0.0.1:8000/docs`

---

## API Endpoints

### `GET /health`
Liveness check.

```json
{ "status": "ok", "service": "GitHub Insights API" }
```

---

### `GET /api/analyze?repo=owner/repo`
Analyze a single repository and return its Hype vs. Substance verdict. Additionally, the endpoint automatically detects the repository's primary language, searches GitHub for `{repo_name} language:{lang}` sorted by stars (e.g. `pothole_detection language:Jupyter Notebook`), fetches the top 10 results, and runs the same batch z-score statistical analysis used by `/api/trending`. The ranked results appear in `similar_repos`.

**Example:**
```
GET http://127.0.0.1:8000/api/analyze?repo=aram199922/pothole_detection
```

**Response:**
```json
{
  "repo": "aram199922/pothole_detection",
  "verdict": "Emerging",
  "confidence": "Low",
  "scores": {
    "hype_score": 5.0,
    "substance_score": 5.0
  },
  "meta": {
    "language": "Jupyter Notebook",
    "description": "...",
    "html_url": "https://github.com/aram199922/pothole_detection"
  },
  "raw_metrics": {
    "stars": 3,
    "forks": 1,
    "open_issues": 0,
    "total_issues": 0,
    "contributor_count": 1,
    "repo_age_days": 180
  },
  "similar_repos": {
    "search_query": "pothole_detection language:Jupyter Notebook",
    "count": 10,
    "results": [
      {
        "rank": 1,
        "repo": "org/pothole_detection",
        "verdict": "Useful",
        "confidence": "High",
        "scores": { "hype_score": 4.8, "substance_score": 8.3 },
        "meta": {
          "language": "Jupyter Notebook",
          "description": "...",
          "html_url": "https://github.com/org/pothole_detection"
        }
      },
      ...
    ]
  }
}
```

`similar_repos` is `null` when the repository's primary language cannot be determined by GitHub.

---

### `GET /api/trending?language=python&limit=10`
Score the top trending repos for a language, ranked by Substance Score.

**Example:**
```
GET http://127.0.0.1:8000/api/trending?language=python&limit=5
```

**Response:**
```json
{
  "language": "python",
  "count": 5,
  "results": [
    {
      "rank": 1,
      "repo": "org/battle-tested-lib",
      "verdict": "Useful",
      "confidence": "High",
      "scores": { "hype_score": 5.2, "substance_score": 8.9 },
      "meta": { ... }
    },
    ...
  ]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **29 passed** in under 1 second (all tests are fully mocked — no network calls).

---

## Architecture Defense Notes

### Why asyncio and not multiprocessing or threading?

GitHub API calls are **I/O-bound**: the CPU sits idle while waiting for HTTP responses. The GIL is not a bottleneck for I/O — the real bottleneck is network latency.

- **asyncio** suspends coroutines during I/O and switches to other work at `await` points — perfect for making 4+ HTTP calls simultaneously with a single thread.
- **multiprocessing** spawns separate Python processes to bypass the GIL — only beneficial for CPU-heavy work (number crunching). Using it here would waste process-spawning overhead for no gain.
- **threading** can parallelize I/O but adds synchronization complexity; asyncio achieves the same result with cleaner, more readable code.

### Where memory was deliberately optimized

In `processor.py`, `optimize_memory()` explicitly downcasts all numeric columns after feature engineering:

```python
df[int_targets] = df[int_targets].astype(np.int32)    # int64 → int32 (50% savings)
df[float_targets] = df[float_targets].astype(np.float32)  # float64 → float32 (50% savings)
```

Star counts and issue counts fit comfortably in `int32` (max ~2.1 billion). Ratios and scores are `float32`-safe. On a 30-repo DataFrame the savings are modest, but the pattern scales correctly to batch jobs.

### How z-score normalization works in the analyzer

Raw metrics have incompatible units: stars are in the tens of thousands, ratios live between 0 and 1. Adding them directly would let raw magnitude dominate.

`scipy.stats.zscore` rescales each column to mean=0, std=1:

```
z = (x - mean) / std
```

Every metric now lives on the same ±3 scale. A z-score of +2 means "this repo is 2 standard deviations above average on this metric" — regardless of whether the metric was stars or a fork ratio. We then apply domain-informed weights and map the result to a 0–10 scale.

### Why two scores instead of one?

A single "quality" score would conflate two separate phenomena. A repo can have:

- **High Hype + Low Substance** → trending overnight, but no maintainer activity or community
- **Low Hype + High Substance** → battle-tested library that the community uses heavily but that doesn't go viral

Separating them into a Hype Score and a Substance Score lets us give a more accurate answer to the core question.