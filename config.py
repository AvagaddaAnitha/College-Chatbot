# =============================================================
# config.py  — All project settings live here
# If you want to change anything (API model, number of results,
# keywords), you only need to edit THIS file.
# =============================================================

import os
from dotenv import load_dotenv

# Load the .env file so os.getenv() can read your API key
load_dotenv()

# ── API Key ───────────────────────────────────────────────────
# Reads key from .env file locally OR from Streamlit Secrets
# when deployed. Never hardcode the actual key here.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Gemini model ──────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"

# ── Embedding model ───────────────────────────────────────────
# all-MiniLM-L6-v2: small (80 MB), fast, accurate enough for us.
# It converts any text into a list of 384 numbers (an embedding).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── How many rows to retrieve per question ────────────────────
TOP_K = 5

# ── Folder paths ─────────────────────────────────────────────
DATA_DIR        = "data"
FAISS_INDEX_DIR = "faiss_index"

# ── Dataset file names ────────────────────────────────────────
# Keys are short names we use in code; values are actual filenames
DATASET_FILES = {
    "eamcet"       : "EAMCET_Cutoff_SVECW.csv",
    "companies"    : "Companies.csv",
    "faculty"      : "Faculty.csv",
    "hostels"      : "Hostels.csv",
    "clubs"        : "clubs.csv",
    "faqs"         : "college_FAQs.csv",
    "publications" : "Sorted_Publications_2020_2025.csv",
}

# ── Keyword router ────────────────────────────────────────────
# When a student asks a question, we check these keywords to decide
# which CSV(s) to search. No dropdown needed — fully automatic.
DATASET_KEYWORDS = {
    "eamcet": [
        "cutoff", "rank", "eamcet", "cse", "ece", "eee", "mech",
        "civil", "branch", "admission", "seat", "category",
        "obc", "sc", "st", "oc", "general",
    ],
    "companies": [
        "placement", "company", "package", "salary", "lpa",
        "recruit", "hired", "placed", "job", "offer", "campus",
        "tcs", "infosys", "wipro", "accenture", "cognizant",
    ],
    "faculty": [
        "faculty", "professor", "teacher", "hod", "staff",
        "department", "lecturer", "associate", "designation",
        "qualification", "phd", "doctor",
    ],
    "hostels": [
        "hostel", "accommodation", "room", "fees", "stay",
        "food", "mess", "warden", "boys", "girls", "boarding",
    ],
    "clubs": [
        "club", "activity", "event", "fest", "cultural",
        "technical", "nss", "ncc", "sports", "coding",
        "dance", "music", "extracurricular",
    ],
    "faqs": [
        "fee", "timing", "bus", "library", "exam", "rule",
        "contact", "phone", "address", "website", "affiliation",
        "naac", "nba", "accreditation", "principal", "college",
        "how", "what", "when", "where", "who",
    ],
    "publications": [
        "research", "paper", "publication", "journal",
        "conference", "author", "ieee", "springer", "scopus",
    ],
}

# ── Page config ───────────────────────────────────────────────
PAGE_TITLE = "SVECW College Chatbot"
PAGE_ICON  = "🎓"
