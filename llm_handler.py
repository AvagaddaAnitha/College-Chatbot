import os
import json
import urllib.request
import urllib.error

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COLLEGE_NAME   = "SVECW (Shri Vishnu Engineering College for Women)"

def _build_prompt(question, context):
    return f"""You are a helpful college assistant for {COLLEGE_NAME}.
Student asked: "{question}"
College database data:
{context}
Rules:
- Answer ONLY what was specifically asked.
- If asked about ONE branch (CSE), give ONLY that branch data.
- If asked about companies starting with a letter, list ONLY those.
- Write 2 to 5 clear friendly sentences.
- Do NOT copy raw data lines — convert to readable answer.
- If info not available say: "Please contact the college office."
"""

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
            "Authorization": f"Bearer {GROQ_API_KEY}",
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
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GOOGLE_API_KEY}"
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

def generate_answer(question: str, context: str) -> str:
    prompt = _build_prompt(question, context)

    if GROQ_API_KEY:
        try:
            return _call_groq(prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            groq_error = f"Groq HTTP {e.code}: {body[:150]}"
        except Exception as e:
            groq_error = f"Groq error: {str(e)[:150]}"
    else:
        groq_error = "GROQ_API_KEY not set in Streamlit Secrets"

    if GOOGLE_API_KEY:
        try:
            return _call_gemini(prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            gemini_error = f"Gemini HTTP {e.code}: {body[:150]}"
        except Exception as e:
            gemini_error = f"Gemini error: {str(e)[:150]}"
    else:
        gemini_error = "GOOGLE_API_KEY not set"

    return _fallback(context, f"{groq_error} | {gemini_error}")

def _fallback(context: str, error: str) -> str:
    return (
        f"⚠️ AI temporarily unavailable.\n\n"
        f"**Here is the information found:**\n\n{context}\n\n"
        f"*Error: {error}*\n\n"
        f"*Please contact the college office for more help.*"
    )

get_ai_response = generate_answer
```

Scroll down → commit message: `Add Groq as primary AI` → click **Commit changes**

---

## Also add GROQ_API_KEY to Streamlit Secrets

Streamlit → Manage app → Settings → Secrets → add:
```
GROQ_API_KEY = "gsk_your-key-here"
