# app.py
import streamlit as st
import json
import pandas as pd

# === Inline replacements for missing modules ===

def parse_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def parse_pdf(uploaded_file):
    return "PDF parsing not yet implemented."

def prepare_vectorstore(text: str):
    # Dummy placeholder: you should replace this with your actual vector store builder
    return text

class DummyAgent:
    def __init__(self, context_text):
        self.context = context_text

    def run(self, query):
        return f"Echoing query: {query}\n\nPreview:\n{self.context[:400]}..."

def build_sql_agent(vectorstore):
    return DummyAgent(vectorstore)

# === Streamlit UI ===

st.set_page_config(page_title="Transit Data Chatbot", layout="centered")
st.title("Transit Data Chatbot")

def flatten_json(y):
    out = []
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}_")
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, f"{name}{i}_")
        else:
            out.append(f"{name[:-1]}: {x}")
    flatten(y)
    return "\n".join(out)

# Load embedded file first
def load_embedded_json():
    try:
        with open("getvehicles.json", "r") as f:
            data = json.load(f)
            st.session_state['json_data'] = data
            st.success("Embedded JSON file loaded.")
            return data
    except Exception as e:
        st.warning(f"Could not load embedded JSON: {e}")
        return None

if "json_data" not in st.session_state:
    load_embedded_json()

# Optional user override
st.subheader("Optional: Upload a new file")
uploaded_file = st.file_uploader("Upload CSV, PDF, or JSON", type=["csv", "pdf", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = parse_csv(uploaded_file)
        if df is not None:
            st.session_state["data"] = df
            st.write("CSV preview:")
            st.dataframe(df.head())

    elif uploaded_file.name.endswith(".pdf"):
        pdf_text = parse_pdf(uploaded_file)
        st.session_state["data"] = pdf_text
        st.write("PDF content preview:")
        st.text(pdf_text[:500])

    elif uploaded_file.name.endswith(".json"):
        try:
            json_data = json.load(uploaded_file)
            st.session_state["json_data"] = json_data
            st.success("JSON file successfully loaded.")
            if st.checkbox("Show raw JSON"):
                st.json(json_data)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")

# Run agent if JSON available
if "json_data" in st.session_state:
    flattened = flatten_json(st.session_state["json_data"])
    vectorstore = prepare_vectorstore(flattened)
    agent = build_sql_agent(vectorstore)

    st.subheader("Ask a question about the data")
    query = st.text_input("Your question:")
    if query:
        answer = agent.run(query)
        st.write("Answer:", answer)
