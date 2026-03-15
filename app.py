# =============================================================
# app.py  — The Streamlit chat interface
#
# This is the file you run: streamlit run app.py
#
# It does only 3 things:
#   1. Shows the chat UI (title, sidebar, message history)
#   2. When student types a question, calls rag_engine to get context
#   3. Passes context to llm_handler to get the final answer
# =============================================================

import streamlit as st
from config import PAGE_TITLE, PAGE_ICON
from rag_engine import load_all_indexes, retrieve_context
from llm_handler import generate_answer


# =============================================================
# PAGE SETUP — must be the very first Streamlit command
# =============================================================

st.set_page_config(
    page_title = PAGE_TITLE,
    page_icon  = PAGE_ICON,
    layout     = "centered",
)


# =============================================================
# CUSTOM STYLING
# =============================================================

st.markdown("""
<style>
    /* Page background — light cream colour */
    .stApp {
        background-color: #faf8f2;
    }

    /* Make the chat input box look cleaner */
    .stChatInput textarea {
        border-radius: 12px !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f0ede0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# LOAD ALL FAISS INDEXES (runs once, cached)
# This is the slow step on first run. After the first run,
# indexes are saved to disk and load in <1 second.
# =============================================================

with st.spinner("Loading college knowledge base... (first load may take ~30 seconds)"):
    indexes, contexts = load_all_indexes()


# =============================================================
# HEADER
# =============================================================

st.title("🎓 SVECW College Chatbot")
st.markdown(
    "Ask me anything about **admissions, EAMCET cutoffs, placements, "
    "hostels, faculty, clubs, or fees**. I'll find the answer from the "
    "college database."
)
st.divider()


# =============================================================
# SIDEBAR — Shows what topics the chatbot covers + example Qs
# =============================================================

with st.sidebar:
    st.header("What can I help with?")

    topics = {
        "📊 EAMCET & Admissions": [
            "What is the CSE cutoff rank for 2023?",
            "What was the OC category rank for ECE last year?",
        ],
        "🏢 Placements": [
            "Which companies visited for placements?",
            "What is the highest package offered?",
        ],
        "🏠 Hostels": [
            "What are the hostel fees?",
            "Is food provided in the hostel?",
        ],
        "👩‍🏫 Faculty": [
            "Who is the HOD of CSE department?",
            "How many PhD faculty are there?",
        ],
        "🎭 Clubs & Activities": [
            "What clubs are available in college?",
            "Are there any technical clubs?",
        ],
        "❓ General FAQs": [
            "What is the college fee structure?",
            "What is the college address?",
        ],
    }

    for topic, questions in topics.items():
        with st.expander(topic):
            for q in questions:
                # Clicking a sample question fills it into the chat
                if st.button(q, key=q, use_container_width=True):
                    st.session_state["prefill_question"] = q
                    st.rerun()

    st.divider()
    st.caption("Powered by RAG + Gemini AI")
    st.caption("Data: SVECW college database")

    # Button to clear chat history
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =============================================================
# CHAT HISTORY — stored in session_state
# session_state persists across reruns in the same browser tab
# =============================================================

# Initialise message list on first load
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Add a welcome message from the bot
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 Hello! I'm the SVECW College Chatbot.\n\n"
            "I can answer questions about:\n"
            "- EAMCET cutoff ranks (2020–2024)\n"
            "- Placement companies and packages\n"
            "- Hostel details and fees\n"
            "- Faculty information\n"
            "- College clubs and activities\n"
            "- General FAQs\n\n"
            "What would you like to know?"
        ),
    })

# Display all previous messages
for msg in st.session_state.messages:
    avatar = "🙋" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# =============================================================
# HANDLE SIDEBAR QUICK-QUESTION PREFILL
# =============================================================

prefill = st.session_state.pop("prefill_question", None)


# =============================================================
# CHAT INPUT — waits for the student to type
# =============================================================

user_input = st.chat_input("Ask anything about SVECW college...")

# If a sidebar question was clicked, use that as input
if prefill:
    user_input = prefill


# =============================================================
# PROCESS THE QUESTION
# =============================================================

if user_input:

    # 1. Show the student's question in the chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    # 2. Show a spinner while we work
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching college database..."):

            # Step A — Retrieve relevant rows from CSV indexes
            context = retrieve_context(user_input, indexes, contexts)

            # Step B — Ask Gemini to generate a friendly answer
            answer = generate_answer(user_input, context)

        # 3. Display the answer
        st.markdown(answer)

        # Optional: show the raw retrieved data in an expander
        # (useful for debugging; you can remove this in production)
        with st.expander("📄 View source data used to answer"):
            st.text(context)

    # 4. Save the answer to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
