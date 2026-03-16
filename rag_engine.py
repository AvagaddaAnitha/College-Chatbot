# =============================================================
# rag_engine.py  — The RAG brain of the chatbot
#
# What this file does (in order):
#   1. Loads each CSV file into a Pandas DataFrame
#   2. Builds a "context string" from ALL columns of each row
#      (This is the fix for why only 1 CSV worked before)
#   3. Converts every context string into a 384-number embedding
#      using SentenceTransformer
#   4. Stores all embeddings in a FAISS index (fast search index)
#   5. Saves indexes to disk so they are NOT rebuilt every restart
#   6. When a question comes in:
#      a. Detects which dataset(s) to search (keyword router)
#      b. Embeds the question
#      c. Searches FAISS for top-K most similar rows
#      d. Returns those rows as context text
# =============================================================

import os
import pickle
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR, FAISS_INDEX_DIR, DATASET_FILES,
    DATASET_KEYWORDS, EMBEDDING_MODEL, TOP_K
)


# =============================================================
# SECTION 1 — Load the embedding model
# @st.cache_resource means Streamlit loads it ONCE and reuses it.
# Without this, the 80 MB model would reload on every interaction.
# =============================================================

@st.cache_resource
def load_embedder():
    """Load the SentenceTransformer model (runs only once)."""
    return SentenceTransformer(EMBEDDING_MODEL)


# =============================================================
# SECTION 2 — Context builders
#
# WHY THIS EXISTS:
# Your old code did: df["context"] = df[df.columns[0]]
# That only uses the FIRST column, so FAISS had almost no info.
#
# Fix: for each dataset, we join ALL important columns into
# one descriptive sentence. FAISS now has the full picture.
# =============================================================

def build_context_for_row(dataset_key: str, row: pd.Series) -> str:
    """
    Given a dataset name and one row, return a rich text description
    using all relevant columns. If a column is missing or NaN,
    we skip it gracefully.
    """

    def safe(col):
        """Return column value as string, or empty string if missing."""
        val = row.get(col, "")
        if pd.isna(val):
            return ""
        return str(val).strip()

    # ── EAMCET cutoffs ────────────────────────────────────────
    if dataset_key == "eamcet":
        parts = []
        for col in row.index:
            v = safe(col)
            if v:
                parts.append(f"{col}: {v}")
        return " | ".join(parts)

    # ── Placement companies ───────────────────────────────────
    elif dataset_key == "companies":
        parts = []
        for col in row.index:
            v = safe(col)
            if v:
                parts.append(f"{col}: {v}")
        return " | ".join(parts)

    # ── Faculty ───────────────────────────────────────────────
    elif dataset_key == "faculty":
        parts = []
        for col in row.index:
            v = safe(col)
            if v:
                parts.append(f"{col}: {v}")
        return " | ".join(parts)

    # ── Hostels ───────────────────────────────────────────────
    elif dataset_key == "hostels":
        parts = []
        for col in row.index:
            v = safe(col)
            if v:
                parts.append(f"{col}: {v}")
        return " | ".join(parts)

    # ── Clubs ─────────────────────────────────────────────────
    elif dataset_key == "clubs":
        parts = []
        for col in row.index:
            v = safe(col)
            if v:
                parts.append(f"{col}: {v}")
        return " | ".join(parts)

    # ── FAQs — special case: already has Question + Answer ────
    elif dataset_key == "faqs":
        q = safe("Question")
        a = safe("Answer")
        if q and a:
            return f"Question: {q} | Answer: {a}"
        # fallback: join all columns
        parts = [f"{col}: {safe(col)}" for col in row.index if safe(col)]
        return " | ".join(parts)

    # ── Publications ──────────────────────────────────────────
    elif dataset_key == "publications":
        parts = []
        for col in row.index:
            v = safe(col)
            if v:
                parts.append(f"{col}: {v}")
        return " | ".join(parts)

    # ── Default fallback (covers any new CSV you add) ─────────
    else:
        parts = [f"{col}: {safe(col)}" for col in row.index if safe(col)]
        return " | ".join(parts)


# =============================================================
# SECTION 3 — Build or load FAISS index for one dataset
#
# First run  → builds index from CSV, saves to disk
# Later runs → loads from disk (much faster)
# =============================================================

def build_or_load_index(dataset_key: str, embedder: SentenceTransformer):
    """
    Returns (faiss_index, list_of_context_strings) for one dataset.
    Saves/loads from faiss_index/ folder automatically.
    """
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    index_path   = os.path.join(FAISS_INDEX_DIR, f"{dataset_key}.index")
    context_path = os.path.join(FAISS_INDEX_DIR, f"{dataset_key}.pkl")

    # ── If saved index exists, load it ────────────────────────
    if os.path.exists(index_path) and os.path.exists(context_path):
        index = faiss.read_index(index_path)
        with open(context_path, "rb") as f:
            contexts = pickle.load(f)
        return index, contexts

    # ── Otherwise build from scratch ──────────────────────────
    filename = DATASET_FILES.get(dataset_key)
    if not filename:
        return None, []

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        st.warning(f"⚠️ File not found: {filepath}")
        return None, []

    # Load CSV or Excel
    # try:
    #     if filepath.endswith(".xlsx"):
    #         df = pd.read_excel(filepath, engine="openpyxl")
    #     else:
    #         df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    # except Exception as e:
    #     st.error(f"❌ Could not read {filename}: {e}")
    #     return None, []

    try:
        if filepath.endswith(".xlsx"):
            df = pd.read_excel(filepath, engine="openpyxl")
        else:
            try:
                df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")
    except Exception as e:
        st.error(f"❌ Could not read {filename}: {e}")
        return None, []


    
    # Build context string for every row
    contexts = []
    for _, row in df.iterrows():
        ctx = build_context_for_row(dataset_key, row)
        if ctx.strip():          # skip empty rows
            contexts.append(ctx)

    if not contexts:
        st.warning(f"⚠️ No data found in {filename}")
        return None, []

    # Convert context strings → embeddings (list of 384-number arrays)
    embeddings = embedder.encode(contexts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (IndexFlatL2 = exact nearest-neighbour search)
    dimension = embeddings.shape[1]           # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save to disk so we don't rebuild next time
    faiss.write_index(index, index_path)
    with open(context_path, "wb") as f:
        pickle.dump(contexts, f)

    return index, contexts


# =============================================================
# SECTION 4 — Load ALL datasets at startup
# @st.cache_resource so this runs only ONCE per app session.
# =============================================================

@st.cache_resource
def load_all_indexes():
    """
    Load (or build) FAISS indexes for every dataset.
    Returns two dicts:
        indexes  = { "eamcet": faiss_index, "companies": faiss_index, ... }
        contexts = { "eamcet": ["row1 text", "row2 text", ...], ... }
    """
    embedder = load_embedder()
    indexes  = {}
    contexts = {}

    for key in DATASET_FILES:
        idx, ctx = build_or_load_index(key, embedder)
        if idx is not None:
            indexes[key]  = idx
            contexts[key] = ctx

    return indexes, contexts


# =============================================================
# SECTION 5 — Smart keyword router
#
# Reads the student's question and returns the list of dataset
# keys that are most relevant. No dropdown required.
# =============================================================

def route_query(question: str) -> list:
    """
    Given a question, return list of dataset keys to search.
    Example:
        "What is the hostel fee?" → ["hostels", "faqs"]
        "CSE cutoff 2023?"        → ["eamcet"]
        "Tell me about TCS"       → ["companies"]
    If nothing matches, return ALL keys (safe fallback).
    """
    q_lower = question.lower()
    matched = []

    for key, keywords in DATASET_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(key)

    # If nothing matched, search everything
    return matched if matched else list(DATASET_FILES.keys())


# =============================================================
# SECTION 6 — Retrieve top-K relevant context rows
#
# This is called every time the student asks a question.
# =============================================================

def retrieve_context(question: str, indexes: dict, contexts: dict) -> str:
    """
    Given a question and all loaded indexes:
      1. Decide which datasets to search
      2. Embed the question
      3. Search FAISS for the top-K most similar rows
      4. Return them joined as a single string for the LLM prompt

    Returns a string like:
        [From EAMCET data]
        Branch: CSE | Year: 2023 | Rank: 4521 | ...

        [From FAQs data]
        Question: What is the fee? | Answer: 1.2L per year
    """
    embedder     = load_embedder()
    dataset_keys = route_query(question)

    # Embed the question into a 384-number vector
    q_embedding  = embedder.encode([question], show_progress_bar=False)
    q_embedding  = np.array(q_embedding, dtype="float32")

    all_results = []

    for key in dataset_keys:
        if key not in indexes:
            continue

        index = indexes[key]
        ctx   = contexts[key]

        # Search for top-K nearest rows
        k = min(TOP_K, index.ntotal)   # can't ask for more than exist
        _, indices = index.search(q_embedding, k)

        rows = []
        for i in indices[0]:
            if 0 <= i < len(ctx):
                rows.append(ctx[i])

        if rows:
            label = key.replace("_", " ").upper()
            block = f"[From {label} data]\n" + "\n".join(rows)
            all_results.append(block)

    if not all_results:
        return "No relevant data found in the college database."

    return "\n\n".join(all_results)
