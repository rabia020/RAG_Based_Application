import os
import requests
import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- LOAD ENVIRONMENT VARIABLES ----------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ---------------- SETTINGS ----------------
PDF_LINKS = [
    "https://drive.google.com/uc?id=1gqn5BuA4iVSdkd9KpvR99WXGSoN3lFRf"
]

VECTOR_DB_PATH = "faiss_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"




# ---------------- DOWNLOAD PDFs ----------------
def download_pdfs():
    os.makedirs("docs", exist_ok=True)

    for i, link in enumerate(PDF_LINKS):
        filename = f"docs/doc_{i}.pdf"

        if not os.path.exists(filename):
            r = requests.get(link)
            with open(filename, "wb") as f:
                f.write(r.content)


# ---------------- LOAD DOCUMENTS ----------------
def load_documents():
    docs = []

    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("docs", file))
            docs.extend(loader.load())

    return docs


# ---------------- CREATE VECTOR DB ----------------
def create_vector_db():
    download_pdfs()

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)


# ---------------- RAG ----------------
def ask_rag(question):

    if not os.path.exists(VECTOR_DB_PATH):
        create_vector_db()

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer using ONLY the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL,
    )

    return response.choices[0].message.content


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="RAG Assistant", page_icon="📚")

st.title("📚 Public PDF RAG Assistant")
st.write("Ask questions from your document knowledge base.")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Ask your question:")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            answer = ask_rag(question)

        st.session_state.chat.append((question, answer))

# Display chat history
for q, a in st.session_state.chat:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")