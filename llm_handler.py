import os
import json
import urllib.request
import urllib.error

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COLLEGE_NAME   = "SVECW (Shri Vishnu Engineering College for Women)"


def _build_prompt(question, context):
    return (
        "You are a helpful college assistant for " + COLLEGE_NAME + ".\n\n"
        "Student asked: " + question + "\n\n"
        "College database data:\n" + context + "\n\n"
        "Rules:\n"
        "- Answer ONLY what was specifically asked.\n"
        "- If asked about ONE branch like CSE, give ONLY that branch data.\n"
        "- If asked about companies starting with a letter, list ONLY those.\n"
        "- Write 2 to 5 clear friendly sentences.\n"
        "- Do NOT copy raw data lines, convert them into a readable answer.\n"
        "- If info is not available say: Please contact the college office.\n"
    )


def _call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = json.dumps({
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + GROQ_API_KEY,
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
    groq_error = ""
    gemini_error = ""

    if GROQ_API_KEY:
        try:
            return _call_groq(prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            groq_error = "Groq HTTP " + str(e.code) + ": " + body[:150]
        except Exception as e:
            groq_error = "Groq error: " + str(e)[:150]
    else:
        groq_error = "GROQ_API_KEY not set in Streamlit Secrets"

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

    return _fallback(context, groq_error + " | " + gemini_error)


def _fallback(context, error):
    return (
        "AI temporarily unavailable.\n\n"
        "Here is the information found:\n\n" + context + "\n\n"
        "Error: " + error + "\n\n"
        "Please contact the college office for more help."
    )


get_ai_response = generate_answer
