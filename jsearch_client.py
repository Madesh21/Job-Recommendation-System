"""JSearch API client for fetching job listings (via RapidAPI)."""
import requests
from typing import List, Optional

from config import JSEARCH_RAPIDAPI_KEY, MAX_RESULTS

JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"
JSEARCH_HOST = "jsearch.p.rapidapi.com"


def fetch_jobs(
    search_query: str,
    location: Optional[str] = None,
    country: str = "us",
    max_results: int = MAX_RESULTS,
) -> List[dict]:
    """
    Fetch job listings from JSearch API (Google for Jobs, Indeed, LinkedIn, etc.).
    Returns list of raw job dicts.
    """
    if not JSEARCH_RAPIDAPI_KEY:
        return []

    # Build query: "python developer in Bangalore" or "python developer"
    query = search_query
    if location:
        query = f"{search_query} in {location}"

    all_results = []
    page = 1
    num_pages = max(1, (max_results + 9) // 10)  # ~10 results per page

    for _ in range(num_pages):
        params = {
            "query": query,
            "page": str(page),
            "num_pages": "1",
        }
        if country:
            params["country"] = country

        headers = {
            "X-RapidAPI-Host": JSEARCH_HOST,
            "X-RapidAPI-Key": JSEARCH_RAPIDAPI_KEY,
        }

        try:
            response = requests.get(JSEARCH_URL, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            break
        except ValueError:
            break

        if data.get("status") != "OK":
            break

        jobs = data.get("data", [])
        if not jobs:
            break

        all_results.extend(jobs)
        if len(jobs) < 10:
            break
        page += 1

    return all_results[:max_results]


def job_to_display(job: dict) -> dict:
    """Normalize JSearch job dict for display."""
    employer = job.get("employer_name") or "N/A"
    if isinstance(employer, dict):
        employer = employer.get("display_name", "N/A")
    source = job.get("job_publisher") or "N/A"

    location_parts = [
        job.get("job_city"),
        job.get("job_state"),
        job.get("job_country"),
    ]
    location = ", ".join(filter(None, location_parts)) or "N/A"

    return {
        "id": job.get("job_id", ""),
        "title": job.get("job_title", "N/A"),
        "company": employer,
        "source": source,
        "location": location,
        "description": job.get("job_description", ""),
        "created": job.get("job_posted_at_datetime_utc"),
        "redirect_url": job.get("job_apply_link", ""),
    }
