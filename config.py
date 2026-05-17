"""Configuration for the Job Recommendation System."""
import os
from dotenv import load_dotenv

load_dotenv()

# JSearch (RapidAPI) settings
JSEARCH_RAPIDAPI_KEY = os.getenv("JSEARCH_RAPIDAPI_KEY", "")

# Other settings
MAX_RESULTS = 50
