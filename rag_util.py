#%%
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

working_dir = os.getcwd()

# Load embedding model
embedding = HuggingFaceEmbeddings()

# Load LLama-3.3-70B model from Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def process_document_to_chroma_db(file_name):
    # Load the PDF document using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Store the document chunk in a Chroma vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=f"{working_dir}/vector_db",
    )
    return 0

def answer_question(user_question):
    # Load the persistent Chroma vector database
    vector_db = Chroma(
        persist_directory=f"{working_dir}/vector_db",
        embedding_function=embedding
    )

    # Create a retriever for document search
    retriever = vector_db.as_retriever()

    # Create a RetrievalQA chain to answer user questions using Llama-3.3-70B
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]
    return answer


#%%
