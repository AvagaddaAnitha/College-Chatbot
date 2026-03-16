# llm_handler.py
# Uses direct HTTP requests to call Gemini API.
# No library dependency — works regardless of what is installed.

import os
import json
import urllib.request
import urllib.error

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COLLEGE_NAME   = "SVECW (Shri Vishnu Engineering College for Women)"

# These are the current working Gemini model names (2025-2026)
MODELS_TO_TRY = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


def generate_answer(question: str, context: str) -> str:
    """
    Main function called by app.py.
    Sends question + context to Gemini via direct HTTP.
    Returns a clean, specific answer.
    """

    if not GOOGLE_API_KEY:
        return _fallback(
            context,
            "API key not set. Go to Streamlit Cloud → your app → "
            "Manage app → Settings → Secrets → add: GOOGLE_API_KEY = 'your-key'"
        )

    prompt = f"""You are a helpful college assistant for {COLLEGE_NAME}.

Student asked: "{question}"

College database data:
{context}

Rules:
- Answer ONLY what was asked. Be specific.
- If asked about CSE branch only, give only CSE data.
- If asked about companies starting with a letter, list only those.
- Write 2 to 4 clear sentences. No raw data copying.
- If info not available, say: "Please contact the college office."
"""

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode("utf-8")

    for model in MODELS_TO_TRY:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GOOGLE_API_KEY}"
        )
        req = urllib.request.Request(
            url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data   = json.loads(resp.read().decode("utf-8"))
                answer = data["candidates"][0]["content"]["parts"][0]["text"]
                return answer.strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if e.code == 404:
                continue   # model not found — try next one
            return _fallback(context, f"HTTP {e.code}: {body[:200]}")
        except Exception as e:
            return _fallback(context, str(e)[:200])

    return _fallback(context, "No working Gemini model found. Check your API key.")


def _fallback(context: str, error: str) -> str:
    return (
        f"⚠️ AI unavailable: {error}\n\n"
        f"**Raw data found:**\n\n{context}\n\n"
        f"*Please contact the college office for more help.*"
    )


# Alias
get_ai_response = generate_answer
