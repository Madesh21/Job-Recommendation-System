import streamlit as st

from config import JSEARCH_RAPIDAPI_KEY
from jsearch_client import fetch_jobs, job_to_display
from resume_parser import extract_resume_text, parse_resume, get_search_query

MAX_KEYWORDS_SEPARATE = 8
JOBS_PER_KEYWORD = 5
JOBS_COMBINED = 10

st.set_page_config(page_title="Job Recommendation System", page_icon="💼", layout="centered")

st.title("💼 Job Recommendation System")
st.markdown("Upload your resume to get jobs matched per keyword")

# Sidebar for settings
with st.sidebar:
    st.subheader("Settings")
    location = st.text_input("Location (optional)", placeholder="e.g., Bangalore, Mumbai")
    st.divider()
    if not JSEARCH_RAPIDAPI_KEY:
        st.warning("Add JSearch RapidAPI key to `.env` file")
        st.caption("Get a free key at [RapidAPI JSearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)")


def render_job_list(jobs_display, section_key: str, limit=10):
    """Render a list of jobs (title, company, source, location, description, apply link)."""
    for i, job in enumerate(jobs_display[:limit], 1):
        with st.container():
            st.markdown(f"**{i}. {job['title']}**")
            st.caption(
                f"**{job['company']}** | Source: {job.get('source', 'N/A')} | {job['location']}"
            )
            expander_label = "Job Description" + ("\u200b" * (i + len(section_key)))
            with st.expander(expander_label):
                full_desc = job.get("description") or ""
                if full_desc.strip():
                    desc = full_desc[:800]
                    st.write(desc + ("..." if len(full_desc) > 800 else ""))
                else:
                    st.caption("No description was provided by the job source for this listing.")
            if job.get("redirect_url"):
                st.link_button("Apply ->", job["redirect_url"], type="primary")
            st.divider()


def format_display_label(value: str) -> str:
    """Format extracted skills/keywords for cleaner UI display."""
    return value.strip().title()


def show_extraction_messages(extraction: dict) -> None:
    """Show extraction warnings in one place."""
    if extraction["method"] != "failed":
        return

    diagnostics = extraction.get("diagnostics", [])
    diagnostic_text = "\n".join(f"- {item}" for item in diagnostics if item)
    st.warning(
        "Could not extract readable text from this PDF. "
        "It may be a scanned image or a non-standard PDF format.\n\n"
        f"{diagnostic_text}\n\n"
        "Please paste your resume text below so the app can still detect keywords."
    )


# File upload
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Analyzing your resume..."):
        try:
            extraction = extract_resume_text(uploaded_file)
        except Exception as e:
            extraction = {"method": "failed", "ocr_attempted": False, "diagnostics": [str(e)]}
            st.warning(f"Error reading PDF: {e}")

    text = extraction["text"]
    show_extraction_messages(extraction)

    if not text or not text.strip():
        pasted = st.text_area("Paste your resume text here", height=250)
        if not pasted.strip():
            st.stop()
        text = pasted

    parsed = parse_resume(text)

    # Let user add extra skills/keywords if something is missing
    extra_input = st.text_input(
        "Add extra skills/keywords (comma-separated)",
        placeholder="e.g., HTML, CSS, Java, Data structures",
    )
    extra_skills = []
    if extra_input.strip():
        extra_skills = [s.strip().lower() for s in extra_input.split(",") if s.strip()]

    # All detected skills (from resume) + user-added skills
    all_skills = list(parsed.get("skills", []))
    for sk in extra_skills:
        if sk and sk not in all_skills:
            all_skills.append(sk)

    # Prioritize core tech skills so html/css/python/sql etc. are always included first
    priority_order = [
        "python",
        "java",
        "javascript",
        "typescript",
        "html",
        "css",
        "sql",
        "react",
        "node",
        "django",
        "flask",
        "fastapi",
    ]
    skills_list: list[str] = []
    # Add prioritized skills that are actually present
    for sk in priority_order:
        if sk in all_skills and sk not in skills_list:
            skills_list.append(sk)
    # Fill remaining slots with any other detected skills
    for sk in all_skills:
        if sk not in skills_list:
            skills_list.append(sk)
    # Limit how many we actually search with
    skills_list = skills_list[:MAX_KEYWORDS_SEPARATE]

    combined_query = get_search_query(parsed)

    # Show extracted keywords
    with st.expander("Extracted keywords (used for separate + combined search)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            formatted_skills = [format_display_label(skill) for skill in all_skills]
            st.write("**All detected skills:**", ", ".join(formatted_skills) or "-")
        with col2:
            formatted_query = " ".join(format_display_label(term) for term in combined_query.split())
            st.write("**Combined search query:**", formatted_query or "-")

    if st.button("Find matching jobs", type="primary"):
        if not JSEARCH_RAPIDAPI_KEY:
            st.error(
                "Please add JSEARCH_RAPIDAPI_KEY to your `.env` file. "
                "Get a free key at https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch"
            )
        else:
            total_searches = max(1, len(skills_list)) + 1  # +1 for combined
            progress = st.progress(0, text="Starting search...")
            results_by_keyword = {}

            # 1) Search for each keyword separately
            for idx, keyword in enumerate(skills_list):
                progress.progress((idx + 1) / total_searches, text=f"Searching: {keyword}...")
                raw = fetch_jobs(
                    search_query=keyword,
                    location=location or None,
                    country="in",
                    max_results=JOBS_PER_KEYWORD,
                )
                results_by_keyword[keyword] = [job_to_display(j) for j in raw]

            # 2) Search with all keywords combined
            progress.progress(1.0, text="Searching: all keywords combined...")
            raw_combined = fetch_jobs(
                search_query=combined_query,
                location=location or None,
                country="in",
                max_results=JOBS_COMBINED,
            )
            results_by_keyword["All keywords combined"] = [job_to_display(j) for j in raw_combined]

            progress.empty()

            # Display: one section per keyword, then combined
            st.success("Jobs by keyword (separate) and combined.")

            for label, jobs_display in results_by_keyword.items():
                display_label = label if label == "All keywords combined" else format_display_label(label)
                st.subheader(f"{display_label} jobs")
                if not jobs_display:
                    st.caption(f'No jobs found for "{display_label}".')
                else:
                    render_job_list(
                        jobs_display,
                        section_key=label.lower().replace(" ", "-"),
                        limit=JOBS_PER_KEYWORD if label != "All keywords combined" else JOBS_COMBINED,
                    )
                st.markdown("---")

            if not skills_list:
                st.info("No skills extracted for per-keyword search. Only combined search was run.")

else:
    st.info("Upload your resume (PDF) to get started")
    st.markdown(
        """
    ### How it works
    1. **Upload** your resume in PDF format.
    2. **NLP** extracts skills (e.g. Python, Linux, SQL).
    3. Jobs are fetched **separately** for each keyword + **combined**.
    4. You see: *Python jobs*, *SQL jobs*, *Linux jobs*, and *All keywords combined*.
    """
    )
