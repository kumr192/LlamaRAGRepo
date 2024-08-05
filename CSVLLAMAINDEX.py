import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Set the title of the app
st.title("CSV Reader using LlamaIndex")

# Create a file uploader for the user to upload their CSV files
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# If the user has uploaded files, save them to a temporary directory and create the index
if uploaded_files is not None and len(uploaded_files) > 0:
    with tempfile.TemporaryDirectory() as tmp_dir:
        documents = []
        for uploaded_file in uploaded_files:
            tmp_file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        documents.extend(SimpleDirectoryReader(tmp_dir).load_data())

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        # Set the OpenAI API key
        api_key = st.text_input("OpenAI API Key:", type="password")

        # If the user has entered an API key, create a text input for the user to enter their query
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

            query = st.text_input("Enter your query:")

            # If the user has entered a query, run it and display the response
            if query:
                response = query_engine.query(query)

                # Store the query and response in the session state
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append((query, response.response))

                # Display the previous 5 questions and answers
                st.subheader("Previous Questions and Answers:")
                for i, (q, a) in enumerate(st.session_state.history[-5:]):
                    st.write(f"Q{i+1}: {q}")
                    st.write(f"A{i+1}: {a}")
        else:
            st.warning("Please enter your OpenAI API key.")
