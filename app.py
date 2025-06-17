# app.py
import streamlit as st
import json
import pandas as pd

# --- Helper functions (replacing file_handlers.py) ---
def parse_csv(uploaded_file):
    """Reads a CSV file and returns a pandas DataFrame."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

def parse_pdf(uploaded_file):
    """Dummy PDF parser placeholder."""
    return "PDF parsing not yet implemented. Please replace with actual logic."

# --- Helper functions (replacing vectorstore_utils.py and df_agent_utils.py) ---
def prepare_vectorstore(flattened_text):
    """Placeholder for vectorstore ingestion logic."""
    return flattened_text  # Simple pass-through

class DummyAgent:
    def run(self, query):
        return f"(Echo) You asked: '{query}' — replace this with LLM response logic."

def build_sql_agent(data):
    return DummyAgent()

# --- Streamlit app logic ---
st.title("Transit Data Chatbot")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "pdf", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = parse_csv(uploaded_file)
        st.session_state["data"] = df
        st.write("CSV data preview:")
        st.dataframe(df.head())

    elif uploaded_file.name.endswith(".pdf"):
        parsed_text = parse_pdf(uploaded_file)
        st.session_state["data"] = parsed_text
        st.write("PDF parsed preview:")
        st.text(parsed_text[:500])

    elif uploaded_file.name.endswith(".json"):
        try:
            json_data = json.load(uploaded_file)
            st.session_state["json_data"] = json_data
            st.success("JSON file successfully loaded.")
            if st.checkbox("Show raw JSON"):
                st.json(json_data)
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")

# --- JSON Flattening ---
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

if "json_data" in st.session_state:
    flattened = flatten_json(st.session_state["json_data"])
    vectorstore = prepare_vectorstore(flattened)
    agent = build_sql_agent(vectorstore)

    query = st.text_input("Ask a question about the JSON data:")
    if query:
        answer = agent.run(query)
        st.write("Answer:", answer)
