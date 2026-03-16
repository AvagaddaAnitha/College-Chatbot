# ============================================================
# llm_handler.py  — fixed version
# Changes from v1:
#   - Updated Gemini model name (fixes 404 error)
#   - Much better prompt that gives specific focused answers
#   - Tells AI to answer ONLY what was asked, not dump all data
# ============================================================

import google.generativeai as genai
from config import GOOGLE_API_KEY, COLLEGE_NAME


def get_ai_response(question: str, contexts: list[str]) -> str:
    """
    Send the question and retrieved CSV contexts to Gemini.
    Returns a specific, focused answer string.
    """

    # Build the context block
    context_block = "\n".join([f"• {ctx}" for ctx in contexts])

    # Improved prompt — tells AI to be specific and focused
    prompt = f"""You are a helpful college information assistant for {COLLEGE_NAME}.

A student asked: "{question}"

Here is the relevant data from the college database:
{context_block}

Your task:
1. Read the student's question carefully.
2. Answer ONLY what they specifically asked — do not show all the data.
3. If they asked about ONE branch (like CSE), give ONLY that branch's info.
4. If they asked about ONE year, give ONLY that year's info.
5. Convert the raw data into a clean, friendly readable answer.
6. Use simple language a first-year student can understand.
7. If the exact information is not in the data, say: "I don't have that specific information. Please contact the college office."

Write a clear, concise answer in 2-4 sentences. Do NOT paste raw data. Do NOT list every branch when only one was asked about."""

    # Try Gemini API with multiple model names as fallback
    gemini_models = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    if not GOOGLE_API_KEY:
        return _fallback(contexts, "GOOGLE_API_KEY is not set in Streamlit Secrets.")

    genai.configure(api_key=GOOGLE_API_KEY)

    for model_name in gemini_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                continue
            return _fallback(contexts, error_msg)

    return _fallback(contexts, "No compatible Gemini model found.")


def _fallback(contexts: list[str], error: str) -> str:
    """Show raw data when AI is unavailable — better than crashing."""
    context_block = "\n".join([f"• {ctx}" for ctx in contexts])
    return (
        f"⚠️ AI assistant unavailable right now ({error})\n\n"
        f"**Here is the raw data I found:**\n\n{context_block}\n\n"
        f"*Please contact the college office for more help.*"
    )
