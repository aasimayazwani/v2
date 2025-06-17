# app.py
import streamlit as st
import json
from file_handlers import parse_pdf, parse_csv  # Assuming you already have these
from vectorstore_utils import prepare_vectorstore  # Assuming this prepares embeddings
from df_agent_utils import build_sql_agent  # For handling structured data questions

# --- File upload section ---
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

# --- Flatten and index JSON for Q&A ---
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
