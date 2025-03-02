import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# Configure models
genai.configure(api_key="AIzaSyBsq5Kd5nJgx2fejR77NT8v5Lk3PK4gbH8")
gemini = genai.GenerativeModel('gemini-1.5-flash')
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

# Load multiple datasets and create FAISS indices
@st.cache_data
def load_data():
    datasets = {
        "SVCEW": 'svcew_details.csv',
        "Dataset2": 'dataset2.csv',  # Add other dataset file names here
        "Dataset3": 'dataset3.csv',
        "Dataset4": 'dataset4.csv',
        "Dataset5": 'dataset5.csv',
        "Dataset6": 'dataset6.csv',
        "Dataset7": 'dataset7.csv',
    }
    
    data_dict = {}
    index_dict = {}
    
    for name, file in datasets.items():
        try:
            df = pd.read_csv(file)
            
            # Handle datasets with different structures
            if 'Question' in df.columns and 'Answer' in df.columns:
                # If the dataset has 'Question' and 'Answer' columns
                df['context'] = df.apply(
                    lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", 
                    axis=1
                )
            else:
                # If the dataset has a different structure, use the first column as context
                df['context'] = df[df.columns[0]].astype(str)  # Use the first column as context
            
            embeddings = embedder.encode(df['context'].tolist())
            index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
            index.add(np.array(embeddings).astype('float32'))
            
            data_dict[name] = df
            index_dict[name] = index
        except Exception as e:
            st.error(f"Failed to load dataset {name}. Error: {e}")
    
    return data_dict, index_dict

data_dict, index_dict = load_data()

# App Header
st.markdown('<h1 class="college-font">🏫 Multi-Dataset Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="college-font">Your Guide to Multiple Datasets</h3>', unsafe_allow_html=True)
st.markdown("---")

# Dataset selection dropdown
selected_dataset = st.selectbox("Select a dataset", list(data_dict.keys()))

# Function to find the closest matching question using FAISS
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=3)  # Top 3 matches
    contexts = [df.iloc[i]['context'] for i in I[0]]
    return contexts

# Function to generate a response using Gemini
def generate_response(query, contexts):
    prompt = f"""You are a helpful and knowledgeable chatbot. Answer the following question using the provided context:
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
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🏫"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Finding the best answer..."):
        try:
            # Get the selected dataset and FAISS index
            df = data_dict[selected_dataset]
            faiss_index = index_dict[selected_dataset]
            
            # Find closest matching questions using FAISS
            contexts = find_closest_question(prompt, faiss_index, df)
            
            # Generate a response using Gemini
            response = generate_response(prompt, contexts)
            response = f"**{selected_dataset} Information**:\n{response}"
        except Exception as e:
            response = f"Sorry, I couldn't generate a response. Error: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
