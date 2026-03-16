# llm_handler.py — fixed to match app.py imports exactly

import google.generativeai as genai
import os

# Read API key directly here — no config import needed
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COLLEGE_NAME   = "SVECW (Shri Vishnu Engineering College for Women)"


def generate_answer(question: str, context: str) -> str:
    """
    Main function called by app.py.
    Sends the question + retrieved CSV rows to Gemini.
    Returns a specific, focused answer.
    """

    prompt = f"""You are a helpful college assistant for {COLLEGE_NAME}.

A student asked: "{question}"

Relevant data from the college database:
{context}

Instructions:
- Answer ONLY what the student asked. Do not show all data.
- If they asked about ONE branch (CSE), give ONLY that branch.
- If they asked about ONE year, give ONLY that year.
- Write in friendly, simple language — 2 to 4 sentences only.
- Do NOT paste raw data lines. Convert them into a readable answer.
- If the exact info is not available, say: "I don't have that specific 
  information. Please contact the college office."
"""

    if not GOOGLE_API_KEY:
        return _fallback(context, "API key not set in Streamlit Secrets.")

    genai.configure(api_key=GOOGLE_API_KEY)

    # Try multiple model names — different API versions use different names
    for model_name in ["gemini-1.5-flash", "gemini-1.5-flash-latest",
                       "gemini-1.0-pro", "gemini-pro"]:
        try:
            model    = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower():
                continue   # try next model name
            return _fallback(context, err)

    return _fallback(context, "No working Gemini model found.")


def _fallback(context: str, error: str) -> str:
    """Show raw data when AI is unavailable — better than crashing."""
    return (
        f"⚠️ AI unavailable ({error})\n\n"
        f"**Raw data found:**\n\n{context}\n\n"
        f"*Please contact the college office for help.*"
    )


# Keep old name working too, just in case
get_ai_response = generate_answer
