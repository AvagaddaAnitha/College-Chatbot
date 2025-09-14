import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# 🎨 Custom CSS for styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: #f8f5e6;
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .college-font {
        font-family: 'Times New Roman', serif;
        color: #2c5f2d;
    }
    .user-msg {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #2c5f2d !important;
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
    }
    .stChatInput {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🔑 Configure Gemini API (securely with secrets)
# -----------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
gemini = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# ⚡ Cache SentenceTransformer
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# -----------------------------
# 📂 Load multiple datasets and create FAISS indices
# -----------------------------
@st.cache_data
def load_data():
    datasets = {
        "Companies": "Companies.csv",
        "EAMCET_Cutoff": "EAMCET_Cutoff_SVECW.csv",
        "Faculty": "Faculty.csv",
        "Faculty Excel": "Faculty.xlsx",
        "Hostels": "Hostels.csv",
        "Publications": "Sorted_Publications_2020_2025.csv",
        "Clubs": "clubs.csv",
        "FAQs": "college_FAQs.csv"
    }
    
    data_dict = {}
    index_dict = {}
    
    for name, file in datasets.items():
        try:
            if file.endswith(".xlsx"):
                df = pd.read_excel(file, engine="openpyxl")
            else:
                df = pd.read_csv(file)
            
            # Handle datasets with different structures
            if "Question" in df.columns and "Answer" in df.columns:
                df["context"] = df.apply(
                    lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", 
                    axis=1
                )
            else:
                df["context"] = df[df.columns[0]].astype(str)
            
            embeddings = embedder.encode(df["context"].tolist())
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype("float32"))
            
            data_dict[name] = df
            index_dict[name] = index
        except Exception as e:
            st.error(f"❌ Failed to load dataset {name}. Error: {e}")
    
    return data_dict, index_dict

data_dict, index_dict = load_data()

# -----------------------------
# 🏫 App Header
# -----------------------------
st.markdown('<h1 class="college-font">🏫 Multi-Dataset College Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="college-font">Your Guide to College Information</h3>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# 🔽 Dataset selection
# -----------------------------
selected_dataset = st.selectbox("📂 Select a dataset", list(data_dict.keys()))

# -----------------------------
# 🔎 Find closest question using FAISS
# -----------------------------
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype("float32"), k=3)
    contexts = [df.iloc[i]["context"] for i in I[0]]
    return contexts

# -----------------------------
# 🤖 Generate response with Gemini
# -----------------------------
def generate_response(query, contexts):
    prompt = f"""You are a helpful and knowledgeable college chatbot. 
    Answer the following question using the provided context:

    Question: {query}
    Contexts: {contexts}

    - Provide a detailed and accurate answer.
    - If the question is unclear, ask for clarification.
    """
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"

# -----------------------------
# 💬 Chat Interface
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🏫"):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything about college..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("🔎 Searching for the best answer..."):
        try:
            df = data_dict[selected_dataset]
            faiss_index = index_dict[selected_dataset]
            contexts = find_closest_question(prompt, faiss_index, df)
            response = generate_response(prompt, contexts)
            response = f"**{selected_dataset} Information**:\n{response}"
        except Exception as e:
            response = f"❌ Sorry, I couldn't generate a response. Error: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
