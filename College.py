import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob

# Custom CSS for styling
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

# Configure Gemini AI Model
genai.configure(api_key="AIzaSyChdnIsx6-c36f1tU2P2BYqkrqBccTyhBE")
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Load Embedding Model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient embedding model

# Function to load multiple datasets dynamically
@st.cache_data
def load_all_data():
    try:
        # Find all CSV files in 'datasets/' folder
        dataset_files = glob.glob(os.path.join("Final Mini Project", "*.csv"))
        
        # Combine all datasets
        all_data = []
        for file in dataset_files:
            df = pd.read_csv(file)
            df['context'] = df.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
            all_data.append(df)
        
        # Merge all datasets
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # Generate embeddings and create FAISS index
        embeddings = embedder.encode(df_combined['context'].tolist(), convert_to_tensor=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings.cpu()).astype('float32'))
        
        return df_combined, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

# Load all datasets
df, faiss_index = load_all_data()

# App Header
st.markdown('<h1 class="college-font">🏫 Welcome to SVECW!!!</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="college-font">Your Guide to Our College Information</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find the closest matching question using FAISS
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    _, I = faiss_index.search(query_embedding.cpu().numpy().astype('float32'), k=5)  # Top 5 matches
    contexts = [df.iloc[i]['context'] for i in I[0]]
    return contexts

# Function to generate a response using Gemini AI
def generate_response(query, contexts):
    prompt = f"""You are a helpful chatbot for SVCEW College. Answer the following question using the provided context:
    Question: {query}
    Contexts: {contexts}
    - Provide a detailed and accurate answer.
    - If the question is unclear, ask for clarification.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🙋‍♀️" if message["role"] == "user" else "🏫"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about SVCEW College..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Finding the best answer..."):
        try:
            # Find closest matching questions using FAISS
            contexts = find_closest_question(prompt, faiss_index, df)
            
            # Generate a response using Gemini
            response = generate_response(prompt, contexts)
            response = f"**College Information**:\n{response}"
        except Exception as e:
            response = f"Sorry, I couldn't generate a response. Error: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
