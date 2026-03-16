# llm_handler.py — uses the NEW google-genai SDK (not deprecated google-generativeai)

import os
from google import genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COLLEGE_NAME   = "SVECW (Shri Vishnu Engineering College for Women)"


def generate_answer(question: str, context: str) -> str:
    """
    Called by app.py — sends question + CSV context to Gemini.
    Returns a specific, focused answer (not raw data).
    """

    prompt = f"""You are a helpful college assistant for {COLLEGE_NAME}.

A student asked: "{question}"

Relevant data from the college database:
{context}

Instructions:
- Answer ONLY what the student specifically asked.
- If they asked about ONE branch (like CSE), give ONLY that branch's data.
- If they asked about ONE year, give ONLY that year's data.
- If they asked about companies starting with a letter, list ONLY those companies.
- Write in friendly, simple language in 2 to 5 sentences.
- Do NOT paste raw data lines. Convert them into a readable answer.
- If the exact info is not in the data, say: "I don't have that specific
  information. Please contact the college office directly."
"""

    if not GOOGLE_API_KEY:
        return _fallback(context, "API key not set. Go to Streamlit Cloud → Settings → Secrets and add GOOGLE_API_KEY.")

    try:
        # New google-genai SDK (replaces deprecated google-generativeai)
        client   = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = prompt,
        )
        return response.text

    except Exception as e:
        err = str(e)
        # Try fallback model if first one fails
        try:
            client   = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model    = "gemini-1.5-flash",
                contents = prompt,
            )
            return response.text
        except Exception as e2:
            return _fallback(context, str(e2))


def _fallback(context: str, error: str) -> str:
    """Show raw data when AI is unavailable — better than crashing."""
    return (
        f"⚠️ AI unavailable: {error}\n\n"
        f"**Raw data found:**\n\n{context}\n\n"
        f"*Please contact the college office for more help.*"
    )


# Keep alias just in case
get_ai_response = generate_answer
