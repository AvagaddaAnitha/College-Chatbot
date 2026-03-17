import os
import json
import urllib.request
import urllib.error

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
COLLEGE_NAME       = "SVECW (Shri Vishnu Engineering College for Women)"


def _build_prompt(question, context):
    return (
        "You are a helpful college assistant for " + COLLEGE_NAME + ".\n\n"
        "Student asked: " + question + "\n\n"
        "College database data:\n" + context + "\n\n"
        "Rules:\n"
        "- Answer ONLY what was specifically asked.\n"
        "- If asked about HOD or head of department, give only that person name and department.\n"
        "- If asked about ONE branch like CSE, give ONLY that branch data.\n"
        "- If asked about companies starting with a letter, list ONLY those.\n"
        "- Write 2 to 5 clear friendly sentences.\n"
        "- Do NOT copy raw data lines, convert them into a readable answer.\n"
        "- If info is not available say: Please contact the college office.\n"
    )


def _call_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = json.dumps({
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENROUTER_API_KEY,
            "HTTP-Referer": "https://college-chatbot.streamlit.app",
            "X-Title": "SVECW College Chatbot",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt):
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + model + ":generateContent?key=" + GOOGLE_API_KEY
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
    raise Exception("No working Gemini model found.")


def generate_answer(question, context):
    prompt = _build_prompt(question, context)
    openrouter_error = ""
    gemini_error = ""

    if OPENROUTER_API_KEY:
        try:
            return _call_openrouter(prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            openrouter_error = "OpenRouter HTTP " + str(e.code) + ": " + body[:150]
        except Exception as e:
            openrouter_error = "OpenRouter error: " + str(e)[:150]
    else:
        openrouter_error = "OPENROUTER_API_KEY not set in Streamlit Secrets"

    if GOOGLE_API_KEY:
        try:
            return _call_gemini(prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            gemini_error = "Gemini HTTP " + str(e.code) + ": " + body[:150]
        except Exception as e:
            gemini_error = "Gemini error: " + str(e)[:150]
    else:
        gemini_error = "GOOGLE_API_KEY not set"

    return _fallback(context, openrouter_error + " | " + gemini_error)


def _fallback(context, error):
    return (
        "AI temporarily unavailable.\n\n"
        "Here is the information found:\n\n" + context + "\n\n"
        "Error: " + error + "\n\n"
        "Please contact the college office for more help."
    )


get_ai_response = generate_answer
