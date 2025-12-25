#%%
import os

import streamlit as st

from rag_util import process_document_to_chroma_db, answer_question

# Set the working directory
working_dir = os.path.dirname(os.path.abspath((__file__)))
st.title("Llama-3.3-70B - Document RAG")

#file uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Define save path
    save_path = os.path.join(working_dir, uploaded_file.name)
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chroma_db(uploaded_file.name)
    st.info("Document Processed Successfully!")

# Text widget to get user input
user_question = st.text_area("Ask your question about the document.")

if st.button("Answer"):
    answer = answer_question(user_question)

    st.markdown("### Llama-3.3-70B Response")
    st.markdown(answer)
