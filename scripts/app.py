
# Making necessary imports
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

# Loading .env for API keys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Loading the data
loader = CSVLoader(file_path=r"C:\Users\sarda\OneDrive\Desktop\PythonCodes\PotfolioProjects\LLM_Internal_Chatbot\data\raw\study_permit_goc.csv")
documents = loader.load()

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Text Splitter
#Splitting the text into smaller chunks
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

# Creating FAISS database
from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)

# Question Answering Chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# Setting up the model
llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.2, "max_length":1000})
chain = load_qa_chain(llm, chain_type="stuff")
def qna(query):
    docs = db.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

def main():
    st.set_page_config(
        page_title="Canada Study Permit Q&A", page_icon=":egg:")

    st.header("Canada Study Permit Q&A Chatbot")
    text_input = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input)>0:
            answer = qna(text_input)
            st.info(answer)

if __name__ == "__main__":
    main()

