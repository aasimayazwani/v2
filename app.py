# app.py
import streamlit as st
import json
import pandas as pd
from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from df_agent_utils import build_sql_agent

st.set_page_config(page_title="Transit Data Chatbot", layout="centered")
st.title("Transit Data Chatbot")

############ Helper Function to Flatten JSON ############
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

############ Automatically Load Embedded JSON File ############
def load_embedded_json():
    try:
        with open("getvehicles.json", "r") as f:
            data = json.load(f)
            st.session_state['json_data'] = data
            st.success("Embedded JSON file loaded successfully.")
            return data
    except Exception as e:
        st.error(f"Failed to load embedded JSON file: {e}")
        return None

if 'json_data' not in st.session_state:
    load_embedded_json()

############ Optional File Upload by User ############
st.subheader("Optional: Upload a new file")
uploaded_file = st.file_uploader("Upload a file", type=["json", "csv", "pdf"])

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

############ Run Agent on Flattened JSON ############
if "json_data" in st.session_state:
    flattened = flatten_json(st.session_state["json_data"])
    vectorstore = prepare_vectorstore(flattened)
    agent = build_sql_agent(vectorstore)

    st.subheader("Ask a question about the data")
    query = st.text_input("Your question:")
    if query:
        answer = agent.run(query)
        st.write("Answer:", answer)

############ Handle other types (CSV, PDF) if needed ############
# Optional: Add handling logic for st.session_state["data"] if not JSON

# End of app.py
