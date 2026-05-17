"""Resume parser with PDF extraction and keyword extraction."""
import io
import re
from collections import Counter
from pathlib import Path
from typing import List, Set

import nltk
from PyPDF2 import PdfReader
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

COMMON_SKILLS = {
    "python", "java", "javascript", "typescript","react", "node", "sql", "mongodb",
    "aws", "docker", "kubernetes", "git", "linux", "agile", "scrum", "machine learning",
    "data analysis", "excel", "powerpoint", "communication", "leadership", "project management",
    "rest api", "graphql", "html", "css", "angular", "vue", "django", "flask", "fastapi",
    "tensorflow", "pytorch", "pandas", "numpy", "tableau", "power bi", "spark", "hadoop",
    "cybersecurity", "cloud", "azure", "gcp", "ci/cd", "devops", "testing", "unit test",
    "frontend", "backend", "fullstack", "full stack", "mobile", "ios", "android",
    "php", "ruby", "csharp", "c++", "r", "go", "golang", "rust", "kotlin", "swift",
    "redux", "next", "vuejs", "jquery", "bootstrap", "tailwind", "sass", "less",
    "postgresql", "mysql", "redis", "elasticsearch", "kafka", "microservices", "canva",
}

JOB_ROLE_PATTERNS = [
    r"(?:senior|junior|mid|lead|principal)?\s*(?:software|web|backend|frontend|full.?stack)\s*(?:engineer|developer)",
    r"(?:data\s+)?(?:scientist|engineer|analyst)",
    r"(?:machine\s+learning|ml)\s*(?:engineer|scientist)",
    r"(?:devops|sre)\s*engineer",
    r"(?:project|product)\s*manager",
    r"(?:ui|ux)\s*(?:designer|developer)",
    r"qa\s*(?:engineer|analyst|tester)",
    r"cloud\s*architect",
    r"(?:system|network)\s*administrator",
    r"business\s*analyst",
]

GLUED_SKILLS = sorted(
    {
        skill
        for skill in COMMON_SKILLS
        if " " not in skill and 3 <= len(skill) <= 6 and re.fullmatch(r"[a-z0-9+#/.]+", skill)
    },
    key=len,
    reverse=True,
)


def _read_source_bytes(source) -> bytes:
    """Read bytes from a file path or file-like object."""
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()

    if hasattr(source, "getvalue"):
        data = source.getvalue()
        return data if isinstance(data, bytes) else bytes(data)

    if hasattr(source, "read"):
        position = source.tell() if hasattr(source, "tell") else None
        data = source.read()
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
        return data

    raise TypeError("Unsupported PDF source type")


def _extract_text_pypdf2_from_bytes(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using PyPDF2."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page_text for page in reader.pages if (page_text := page.extract_text()))


def normalize_resume_text(text: str) -> str:
    """Clean extracted text before NLP and skill matching."""
    if not text:
        return ""

    normalized = text.replace("\r", "\n")
    for skill in GLUED_SKILLS:
        escaped = re.escape(skill)
        normalized = re.sub(rf"(?i)(?<=[A-Za-z0-9])({escaped})(?=[A-Za-z0-9])", r" \1 ", normalized)

    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_resume_text(source):
    """Extract resume text using standard PDF text extraction only."""
    diagnostics: List[str] = []
    try:
        raw_text = _extract_text_pypdf2_from_bytes(_read_source_bytes(source))
    except Exception as exc:
        raw_text = ""
        diagnostics.append(f"Standard PDF extraction failed: {exc}")

    text = normalize_resume_text(raw_text)
    if text:
        return {
            "text": text,
            "raw_text": raw_text,
            "method": "pypdf2",
            "diagnostics": diagnostics,
        }

    diagnostics.append("Standard PDF extraction returned no readable text.")
    return {
        "text": "",
        "raw_text": "",
        "method": "failed",
        "diagnostics": diagnostics,
    }


def extract_text_from_pdf(file_path: str) -> str:
    """Extract normalized text from a PDF file path."""
    return extract_resume_text(file_path)["text"]


def extract_text_from_upload(uploaded_file) -> str:
    """Extract normalized text from a Streamlit uploaded PDF."""
    return extract_resume_text(uploaded_file)["text"]


def extract_keywords_rake(text: str, top_n: int = 30) -> List[str]:
    """Extract keywords using RAKE."""
    rake = Rake(max_length=3, include_repeated_phrases=False)
    rake.extract_keywords_from_text(text.lower())
    return rake.get_ranked_phrases()[:top_n]


def extract_skills_from_text(text: str) -> Set[str]:
    """Extract skills by exact matching against the skill list."""
    text_lower = text.lower()
    return {
        skill
        for skill in COMMON_SKILLS
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)
    }


def extract_job_roles(text: str) -> Set[str]:
    """Extract job role mentions using regex patterns."""
    text_lower = text.lower()
    roles = set()
    for pattern in JOB_ROLE_PATTERNS:
        roles.update(match.strip() for match in re.findall(pattern, text_lower, re.IGNORECASE) if match.strip())
    return roles


def extract_tfidf_terms(text: str, top_n: int = 20) -> List[str]:
    """Extract important terms using TF-IDF."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        words = [word.lower() for word in nltk.word_tokenize(text) if word.isalnum() and len(word) > 2]
        return [word for word, _ in Counter(words).most_common(top_n)]

    vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().sum(axis=0)
    top_indices = scores.argsort()[::-1][:top_n]
    return [feature_names[index] for index in top_indices]


def parse_resume(text: str) -> dict:
    """Parse resume text and extract keywords for job matching."""
    text = normalize_resume_text(text)
    if not text:
        return {"skills": set(), "job_roles": set(), "description_keywords": [], "search_terms": []}

    skills = extract_skills_from_text(text)
    job_roles = extract_job_roles(text)
    rake_keywords = extract_keywords_rake(text, top_n=25)
    tfidf_terms = extract_tfidf_terms(text, top_n=15)

    all_keywords = list(skills) + list(job_roles)
    for keyword in rake_keywords + tfidf_terms:
        if keyword not in all_keywords and (keyword in tfidf_terms or len(keyword.split()) <= 2):
            all_keywords.append(keyword)

    search_terms = list(job_roles)[:3] + list(skills)[:5]
    if not search_terms:
        search_terms = rake_keywords[:5]

    return {
        "skills": skills,
        "job_roles": job_roles,
        "description_keywords": all_keywords,
        "search_terms": search_terms[:10],
    }


def get_search_query(parsed: dict) -> str:
    """Build a combined search query string from parsed resume keywords."""
    terms = parsed.get("search_terms", [])
    return "developer" if not terms else " ".join(terms[:5])
