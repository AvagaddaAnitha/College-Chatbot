import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Configure models
genai.configure(api_key="AIzaSyChdnIsx6-c36f1tU2P2BYqkrqBccTyhBE")
gemini = genai.GenerativeModel('gemini-1.5-flash')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Store numerical and alphanumeric data separately
numerical_data = {}
alphanumeric_data = {}

# Function to load and process multiple datasets
@st.cache_data
def load_all_data():
    global numerical_data, alphanumeric_data
    try:
        dataset_files = glob.glob(os.path.join("Final Mini Project", "*.csv"))
        if not dataset_files:
            st.warning("No datasets found in the 'datasets/' folder.")
            return None, None

        all_data = []
        
        for file in dataset_files:
            df = pd.read_csv(file)
            
            if 'Question' in df.columns and 'Answer' in df.columns:
                # Handle Question-Answer datasets
                df['context'] = df.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
                all_data.append(df)
            else:
                # Handle Numerical or Alphanumeric datasets
                if df.select_dtypes(include=[np.number]).shape[1] > 0:
                    numerical_data[file] = df
                else:
                    alphanumeric_data[file] = df
        
        # If no FAQ datasets found, avoid concatenation error
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            embeddings = embedder.encode(df_combined['context'].tolist(), convert_to_tensor=True)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings.cpu()).astype('float32'))
            return df_combined, index
        else:
            st.warning("No FAQ datasets found (with 'Question' and 'Answer' columns).")
            return None, None

    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        return None, None

df, faiss_index = load_all_data()

# Function to search in FAISS for FAQs
def find_closest_question(query):
    if faiss_index is None or df is None:
        return []
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=3)
    return [df.iloc[i]['context'] for i in I[0]]

# Function to retrieve numerical or alphanumeric data
def find_structured_data(query):
    for filename, data in numerical_data.items():
        if query in data.columns:
            return f"**{query} Data from {filename}**:\n{data[query].to_string(index=False)}"
    for filename, data in alphanumeric_data.items():
        if query in data.columns:
            return f"**{query} Information from {filename}**:\n{data[query].to_string(index=False)}"
    return None

# Function to generate response using Gemini
def generate_response(query, contexts):
    prompt = f"""You are a chatbot for SVCEW College. Answer the following question using the provided context:
    Question: {query}
    Contexts: {contexts}
    - Provide a detailed and accurate answer.
    - If the question is unclear, ask for clarification.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Streamlit Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🙋‍♀️" if message["role"] == "user" else "🏫"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about SVCEW College..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Searching..."):
        try:
            structured_data_response = find_structured_data(prompt)
            
            if structured_data_response:
                response = structured_data_response
            else:
                # Find closest matching questions using FAISS
                contexts = find_closest_question(prompt)
                # Generate a response using Gemini
                response = generate_response(prompt, contexts) if contexts else "I'm not sure. Can you clarify?"
                response = f"**College Information**:\n{response}"
        except Exception as e:
            response = f"Sorry, I couldn't generate a response. Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
