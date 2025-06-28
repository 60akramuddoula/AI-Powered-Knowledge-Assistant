import streamlit as st
import os

import fitz  # PyMuPDF
import docx

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ================================
# CONFIG
# ================================

DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = Ollama(model="mistral")


# ================================
# HELPERS
# ================================
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")  # or page.get_text("text")
    return text

print(extract_text_from_pdf("sample.pdf"))


def extract_text_from_word(doc_path):
    text = ""
    doc = docx.Document(doc_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_word(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        return ""

def process_document(file_path):
    text = extract_text(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    return texts

def store_embeddings(texts):
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )
    vectorstore.add_texts(texts)
    return "✅ Embeddings stored!"

def search_and_summarize(query):
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.invoke(query)
    return response


# ================================
# STREAMLIT APP
# ================================

st.set_page_config(page_title="AI-Powered Knowledge Assistant", page_icon="🤖")
st.title("📚 AI-Powered Knowledge Assistant")

st.sidebar.header("📂 Upload Documents")
uploaded_file = st.sidebar.file_uploader(
    "Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"]
)

if uploaded_file:
    os.makedirs("uploaded_files", exist_ok=True)
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Process + embed
    texts = process_document(file_path)
    if texts:
        store_msg = store_embeddings(texts)
        st.sidebar.success(store_msg)
    else:
        st.sidebar.error("❌ Could not extract any text.")

# Chat UI
st.subheader("💬 Ask a Question")
user_query = st.text_input("Type your question:")

if st.button("Ask AI"):
    if user_query:
        response = search_and_summarize(user_query)
        st.markdown(f"**🤖 AI Response:** {response}")
    else:
        st.warning("Please enter a question.")
