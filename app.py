import os
import shutil
import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

DOCS_PATH = "docs"
VECTOR_DB_PATH = "faiss_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

APP_NAME = "RAG-Based PDF Assistant"
APP_TAGLINE = "DOCUMENT ASSISTANT"

SUGGESTED_QUESTIONS = [
    "Summarize the main findings",
    "What conclusions are supported by the evidence?",
    "List the most important cited pages",
]


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide"
)


# ============================================================
# THEME / CSS
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        :root{
            --bg-main:#ece4d8;
            --bg-sidebar:#161512;
            --accent:#e2622c;
            --accent-hover:#c9531f;
            --text-dark:#1c1a17;
            --text-muted:#8a8378;
            --card-bg:#f4eee4;
            --border-soft:rgba(0,0,0,0.08);
        }

        .stApp{
            background-color: var(--bg-main);
        }

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"]{
            background-color: var(--bg-sidebar);
            border-right: 1px solid #000;
        }
        section[data-testid="stSidebar"] *{
            color: #ece7de;
        }
        section[data-testid="stSidebar"] .stMarkdown p{
            color: #9a9488;
        }
        section[data-testid="stSidebar"] hr{
            border-color: rgba(255,255,255,0.08);
        }

        .brand-row{
            display:flex;
            align-items:center;
            gap:10px;
            padding: 4px 0 18px 0;
        }
        .brand-dot{
            width:34px;height:34px;border-radius:8px;
            background:#2a2822;
            display:flex;align-items:center;justify-content:center;
            flex-shrink:0;
        }
        .brand-dot span{
            width:9px;height:9px;border-radius:50%;
            background: var(--accent);
            display:block;
        }
        .brand-title{
            font-weight:700;font-size:15px;letter-spacing:0.5px;color:#f5f1e9;
            line-height:1.1;
        }
        .brand-sub{
            font-size:10px;letter-spacing:1.5px;color:#7d7669;
            text-transform:uppercase;
        }

        .lib-label{
            display:flex;justify-content:space-between;align-items:center;
            font-size:11px;letter-spacing:1.5px;color:#8a8378;
            text-transform:uppercase;
            margin: 6px 0 10px 0;
        }

        .empty-lib{
            text-align:center;
            padding: 26px 10px;
            color:#7d7669;
        }
        .empty-lib .icon{font-size:22px;opacity:0.5;margin-bottom:6px;}
        .empty-lib .title{color:#e4dfd4;font-weight:600;font-size:13px;margin-bottom:4px;}
        .empty-lib .desc{font-size:12px;color:#8a8378;}

        .doc-row{
            display:flex;align-items:center;justify-content:space-between;
            padding:6px 2px;font-size:13px;color:#ece7de;
        }

        /* Sidebar buttons */
        section[data-testid="stSidebar"] .stButton > button{
            background-color: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight:600;
            padding: 0.55rem 0.75rem;
        }
        section[data-testid="stSidebar"] .stButton > button:hover{
            background-color: var(--accent-hover);
            color:white;
        }
        section[data-testid="stSidebar"] .secondary-btn button{
            background-color: transparent !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: #ece7de !important;
        }

        /* File uploader */
        section[data-testid="stSidebar"] [data-testid="stFileUploader"]{
            background:#1f1d19;
            border:1px dashed rgba(255,255,255,0.15);
            border-radius:10px;
            padding:6px;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
            background:transparent;
        }
        /* Upload documents label text -> black */
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p{
            color:#111111 !important;
            font-weight:600;
        }

        /* ---------- HISTORY PANEL ---------- */
        .history-btn button{
            background:transparent !important;
            border:1px solid var(--accent) !important;
            color: var(--accent) !important;
        }
        .history-panel{
            background: var(--card-bg);
            border:1px solid var(--border-soft);
            border-radius:12px;
            padding:14px 16px;
            margin-bottom:14px;
        }
        .history-panel-title{
            font-size:12px;letter-spacing:1.5px;text-transform:uppercase;
            color:var(--text-muted);margin-bottom:10px;
        }
        .history-empty{
            font-size:13px;color:var(--text-muted);
        }
        .history-panel .stButton > button{
            background:#fff;
            border:1px solid var(--border-soft);
            color: var(--text-dark);
            text-align:left;
            justify-content:flex-start;
            border-radius:8px;
            font-weight:500;
        }
        .history-panel .stButton > button:hover{
            border-color: var(--accent);
            color: var(--accent);
        }

        /* ---------- TOP BAR ---------- */
        .topbar{
            display:flex;align-items:center;justify-content:space-between;
            padding: 6px 0 18px 0;
            border-bottom: 1px solid var(--border-soft);
            margin-bottom: 10px;
        }
        .topbar-left{display:flex;align-items:center;gap:10px;}
        .topbar-title{font-weight:700;font-size:17px;color:var(--text-dark);}
        .chip{
            background:#e0d8c9;
            color:#6d6656;
            font-size:12px;
            padding:3px 10px;
            border-radius:999px;
        }

        /* ---------- HERO ---------- */
        .hero-icon{
            width:46px;height:46px;border-radius:10px;
            background: var(--text-dark);
            display:flex;align-items:center;justify-content:center;
            color:white;font-size:20px;
            margin-bottom:18px;
        }
        .hero-eyebrow{
            font-size:12px;letter-spacing:2px;color:var(--text-muted);
            text-transform:uppercase;margin-bottom:6px;
        }
        .hero-title{
            font-size:38px;font-weight:800;color:var(--text-dark);
            line-height:1.15;margin-bottom:14px;
        }
        .hero-sub{
            font-size:15px;color:#5c564b;max-width:560px;line-height:1.5;
        }

        .suggested-label{
            font-size:11px;letter-spacing:1.5px;color:var(--text-muted);
            text-transform:uppercase;margin: 26px 0 10px 0;
        }

        /* Suggested question pill buttons (main area) */
        div[data-testid="stHorizontalBlock"] .stButton > button{
            background:#f4eee4;
            color:#4a453b;
            border:1px solid var(--border-soft);
            border-radius:999px;
            font-size:13px;
            padding:0.4rem 0.9rem;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover{
            border-color: var(--accent);
            color: var(--accent);
        }

        /* ---------- CHAT ---------- */
        [data-testid="stChatMessage"]{
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border-soft);
        }

        /* Chat input pinned bottom */
        [data-testid="stChatInput"]{
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
        }
        [data-testid="stChatInput"] textarea{
            color: var(--text-dark);
        }
        .stChatInput button{
            background-color: var(--accent) !important;
        }

        .footnote{
            font-size:11px;color:var(--text-muted);margin-top:6px;
        }

        .status-banner{
            background:#e7e0d3;
            border:1px solid var(--border-soft);
            border-radius:10px;
            padding:8px 14px;
            font-size:13px;
            color:#4a453b;
            margin-bottom:14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_css()


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(DOCS_PATH, exist_ok=True)


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================

def get_uploaded_documents():
    """Return all uploaded PDF filenames."""

    if not os.path.exists(DOCS_PATH):
        return []

    return sorted(
        [
            file
            for file in os.listdir(DOCS_PATH)
            if file.lower().endswith(".pdf")
        ]
    )


def save_uploaded_files(uploaded_files):
    """Save uploaded PDFs to the docs directory."""

    saved_files = []

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            DOCS_PATH,
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(uploaded_file.name)

    return saved_files


def delete_document(filename):
    """Delete a PDF from the document directory."""

    file_path = os.path.join(
        DOCS_PATH,
        filename
    )

    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove existing vector database because
    # it no longer represents the current documents.
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    """Load all PDFs and preserve source/page metadata."""

    documents = []

    pdf_files = get_uploaded_documents()

    for filename in pdf_files:

        file_path = os.path.join(
            DOCS_PATH,
            filename
        )

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        for doc in docs:

            # Store the actual filename
            doc.metadata["source_file"] = filename

            # PyPDFLoader pages are zero-indexed.
            # Convert to human-readable page numbers.
            if "page" in doc.metadata:
                doc.metadata["page_number"] = (
                    doc.metadata["page"] + 1
                )

        documents.extend(docs)

    return documents


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_db():

    documents = load_documents()

    if not documents:
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    db.save_local(VECTOR_DB_PATH)

    return True


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vector_db():

    embeddings = get_embeddings()

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# ============================================================
# CONVERSATION-AWARE QUERY
# ============================================================

def rewrite_query(question):
    """
    Rewrite the current question using previous conversation
    so that follow-up questions can be retrieved correctly.
    """

    history = st.session_state.get("chat", [])

    if not history:
        return question

    recent_history = history[-6:]

    conversation_text = ""

    for item in recent_history:
        q, a = item[0], item[1]
        conversation_text += (
            f"User: {q}\n"
            f"Assistant: {a}\n\n"
        )

    prompt = f"""
You are a query rewriting assistant for a PDF question-answering system.

Rewrite the user's latest question into a standalone search query.

Use the conversation history to resolve references such as:
- it
- they
- this
- that
- the previous document
- the product mentioned earlier

Do not answer the question.

Return ONLY the rewritten search query.

Conversation history:
{conversation_text}

Latest user question:
{question}
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=LLM_MODEL,
        temperature=0
    )

    rewritten_query = response.choices[0].message.content.strip()

    return rewritten_query


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def ask_rag(question):

    # Create vector database if it does not exist.
    if not os.path.exists(VECTOR_DB_PATH):

        created = create_vector_db()

        if not created:
            return (
                "Please upload at least one PDF document first.",
                []
            )

    # Load FAISS
    db = load_vector_db()

    # Rewrite question using conversation history
    search_query = rewrite_query(question)

    # Retrieve relevant chunks
    docs = db.similarity_search(
        search_query,
        k=5
    )

    if not docs:
        return (
            "I could not find relevant information in the uploaded documents.",
            []
        )

    # Build context
    context_parts = []

    sources = []

    for i, doc in enumerate(docs):

        source_file = doc.metadata.get(
            "source_file",
            "Unknown document"
        )

        page_number = doc.metadata.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {i + 1}
Document: {source_file}
Page: {page_number}

Content:
{doc.page_content}
"""
        )

        sources.append(
            {
                "document": source_file,
                "page": page_number
            }
        )

    context = "\n\n".join(context_parts)

    # Generate grounded answer
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, clearly say that
the information was not found in the uploaded documents.

Do not use outside knowledge.

Document context:
{context}

User question:
{question}
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=LLM_MODEL,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return answer, sources


# ============================================================
# SESSION STATE
# ============================================================

if "chat" not in st.session_state:
    st.session_state.chat = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "show_history" not in st.session_state:
    st.session_state.show_history = False

if "spotlight_index" not in st.session_state:
    st.session_state.spotlight_index = None


def run_question(q):
    """Runs the RAG pipeline for a question and stores it in chat history."""

    with st.spinner("Searching documents and generating answer..."):
        answer, sources = ask_rag(q)

    st.session_state.chat.append((q, answer, sources))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div class="brand-row">
            <div class="brand-dot"><span></span></div>
            <div>
                <div class="brand-title">{APP_NAME.upper()}</div>
                <div class="brand-sub">{APP_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Upload PDFs
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(
        "📥  Upload documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible"
    )

    if uploaded_files:

        if st.button(
            "📥 Add Documents",
            use_container_width=True
        ):

            saved = save_uploaded_files(
                uploaded_files
            )

            # Existing FAISS index is now outdated.
            if os.path.exists(VECTOR_DB_PATH):
                shutil.rmtree(VECTOR_DB_PATH)

            st.success(
                f"Added {len(saved)} document(s)."
            )

            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Document list
    # --------------------------------------------------------

    documents = get_uploaded_documents()

    st.markdown(
        f"""<div class="lib-label"><span>LIBRARY</span><span>{len(documents)} files</span></div>""",
        unsafe_allow_html=True
    )

    if not documents:

        st.markdown(
            """
            <div class="empty-lib">
                <div class="icon">🗂️</div>
                <div class="title">Your library is empty</div>
                <div class="desc">Add PDF files to begin a cited conversation.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        for filename in documents:

            col1, col2 = st.columns(
                [4, 1]
            )

            with col1:
                st.markdown(
                    f"<div class='doc-row'>📄 {filename}</div>",
                    unsafe_allow_html=True
                )

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{filename}"
                ):

                    delete_document(filename)

                    st.success(
                        f"Deleted {filename}"
                    )

                    st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Rebuild vector database
    # --------------------------------------------------------

    if documents:

        if st.button(
            "🔄 Rebuild Knowledge Base",
            use_container_width=True
        ):

            with st.spinner(
                "Rebuilding vector database..."
            ):

                if os.path.exists(VECTOR_DB_PATH):
                    shutil.rmtree(
                        VECTOR_DB_PATH
                    )

                create_vector_db()

            st.success(
                "Knowledge base rebuilt successfully."
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
    if st.button(
        "↺ Clear chat",
        use_container_width=True,
        key="sidebar_clear_chat"
    ):
        st.session_state.chat = []
        st.session_state.show_history = False
        st.session_state.spotlight_index = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TOP BAR
# ============================================================

documents = get_uploaded_documents()

top_left, top_right = st.columns([3, 1])

with top_left:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-left">
                <span class="topbar-title">Reading Room</span>
                <span class="chip">{len(documents)} sources linked</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_right:
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="history-btn">', unsafe_allow_html=True)
        if st.button("🕘 History", use_container_width=True, key="history_btn"):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with b2:
        if st.button("↺ Clear chat", use_container_width=True, key="topbar_clear_chat"):
            st.session_state.chat = []
            st.session_state.show_history = False
            st.session_state.spotlight_index = None
            st.rerun()


# ============================================================
# HISTORY PANEL
# ============================================================

if st.session_state.show_history:

    st.markdown('<div class="history-panel">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="history-panel-title">Conversation History · {len(st.session_state.chat)} question(s)</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.chat:
        st.markdown(
            '<div class="history-empty">No questions asked yet in this session.</div>',
            unsafe_allow_html=True
        )
    else:
        for idx, item in enumerate(reversed(st.session_state.chat)):

            real_idx = len(st.session_state.chat) - 1 - idx
            q_text = item[0]
            label = q_text if len(q_text) <= 80 else q_text[:77] + "..."

            if st.button(f"🕘  {label}", key=f"hist_item_{real_idx}", use_container_width=True):
                st.session_state.spotlight_index = real_idx
                st.session_state.show_history = False
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# STATUS BANNER
# ============================================================

if documents:
    st.markdown(
        f'<div class="status-banner">✅ Knowledge base contains {len(documents)} PDF(s).</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="status-banner">ℹ️ Upload one or more PDF documents from the sidebar to get started.</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MAIN CONTENT
# ============================================================

def render_chat_item(item):
    """Renders a single (question, answer, sources) chat exchange."""

    # Support new 3-item format
    if len(item) == 3:
        q, answer, sources = item
    else:
        q, answer = item
        sources = []

    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant", avatar="📚"):
        st.markdown(answer)

        if sources:
            with st.expander("📑 Sources used"):

                displayed_sources = set()

                for source in sources:

                    document = source["document"]
                    page = source["page"]

                    source_key = (document, page)

                    if source_key in displayed_sources:
                        continue

                    displayed_sources.add(source_key)

                    st.markdown(
                        f"📄 **{document}** — Page **{page}**"
                    )


if st.session_state.spotlight_index is not None and st.session_state.chat:

    # ---------------- SPOTLIGHT (single item from history) ----------------
    idx = min(st.session_state.spotlight_index, len(st.session_state.chat) - 1)

    if st.button("← Back to full conversation", key="back_to_chat"):
        st.session_state.spotlight_index = None
        st.rerun()

    render_chat_item(st.session_state.chat[idx])

elif not st.session_state.chat:

    # ---------------- HERO ----------------
    st.markdown('<div class="hero-icon">📖</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-eyebrow">Your research desk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Read across every document<br/>without losing the source.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-sub">Add one or more PDFs, then ask a question. '
        f'{APP_NAME} keeps every answer connected to its original page.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="suggested-label">Suggested questions</div>', unsafe_allow_html=True)

    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, suggestion in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.pending_question = suggestion

else:

    # ---------------- FULL CHAT HISTORY ----------------
    for item in st.session_state.chat:
        render_chat_item(item)


# ============================================================
# QUESTION INPUT (pinned bottom, chat-style)
# ============================================================

chat_question = st.chat_input("Ask your documents a question...")

st.markdown(
    '<div class="footnote">Answers stay tied to the pages they came from.</div>',
    unsafe_allow_html=True
)


# ============================================================
# HANDLE QUESTION SUBMISSION
# ============================================================

question_to_run = None

if chat_question and chat_question.strip():
    question_to_run = chat_question.strip()
elif st.session_state.pending_question:
    question_to_run = st.session_state.pending_question
    st.session_state.pending_question = None

if question_to_run:

    if not documents:
        st.warning("Please upload at least one PDF first.")
    else:
        st.session_state.spotlight_index = None
        st.session_state.show_history = False
        run_question(question_to_run)
        st.rerun()