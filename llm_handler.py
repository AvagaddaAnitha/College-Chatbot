# =============================================================
# llm_handler.py  — Calls the Gemini API and returns an answer
#
# WHY THIS IS A SEPARATE FILE:
# If you want to switch from Gemini to Claude or OpenAI later,
# you only change THIS file. The rest of the app stays the same.
#
# WHAT THE FALLBACK DOES:
# If the Gemini API key is expired or the network is down, instead
# of crashing the whole app, we show the raw retrieved rows so
# the student still gets useful information.
# =============================================================

import google.generativeai as genai
from config import GOOGLE_API_KEY, GEMINI_MODEL


# =============================================================
# The system prompt — tells Gemini how to behave
# =============================================================

SYSTEM_PROMPT = """You are a helpful and friendly college information assistant 
for SVECW (Shri Vishnu Engineering College for Women), Bhimavaram, Andhra Pradesh.

Your job is to answer questions from students who want to join the college.

Rules you must follow:
1. Answer ONLY using the context data provided below. Do not make up information.
2. If the context does not contain enough information, say:
   "I don't have complete information on that. Please contact the college office 
    or visit www.svecw.edu.in for accurate details."
3. Be friendly and clear. Use simple language suitable for students.
4. When giving numbers (ranks, fees, packages), be specific and accurate.
5. If multiple years of data are available, mention the trend briefly.
6. Keep your answer concise — under 150 words unless the question genuinely 
   requires a longer answer.
"""


# =============================================================
# Main function — generate answer from question + context
# =============================================================

def generate_answer(question: str, context: str) -> str:
    """
    Send the student's question + relevant CSV context to Gemini.
    Returns Gemini's answer as a string.

    Falls back to showing raw context if API fails.
    """

    # ── Check if API key exists ───────────────────────────────
    if not GOOGLE_API_KEY:
        return (
            "⚠️ **API key not configured.**\n\n"
            "Please add your GOOGLE_API_KEY to the `.env` file (local) "
            "or Streamlit Secrets (deployed).\n\n"
            f"**Raw data found:**\n{context}"
        )

    # ── Build the full prompt ─────────────────────────────────
    # We give Gemini:
    #   - The system instructions (how to behave)
    #   - The actual CSV data we retrieved (the context)
    #   - The student's question
    full_prompt = f"""{SYSTEM_PROMPT}

--- COLLEGE DATA (use only this to answer) ---
{context}
--- END OF DATA ---

Student Question: {question}

Please answer the question using the data above."""

    # ── Call Gemini ───────────────────────────────────────────
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        error_msg = str(e)

        # Friendly error messages for common problems
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            friendly = "❌ Your Gemini API key is invalid or expired. Please get a new key from https://aistudio.google.com/app/apikey"
        elif "quota" in error_msg.lower() or "429" in error_msg:
            friendly = "⚠️ Gemini API quota exceeded. Free tier allows ~60 requests/minute. Please wait a moment and try again."
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            friendly = "🌐 Network error. Please check your internet connection."
        else:
            friendly = f"⚠️ Could not get AI response ({error_msg})"

        # Even if AI fails, show the raw data so student is not left empty-handed
        return (
            f"{friendly}\n\n"
            "**Here is the raw data found in the college database:**\n\n"
            f"{context}"
        )
