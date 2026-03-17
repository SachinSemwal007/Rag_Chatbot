import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import streamlit as st

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG PDF Chatbot")
st.caption("Upload a PDF or text file and ask questions about it.")
 
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

def load_document(file_path):
    """Load a text or PDF document"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s) from document.")
    return documents

def create_vector_store(documents):
    """Split document into chunks and store in ChromaDB"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")

    # Free HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Store in ChromaDB locally
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Vector store created successfully.")
    return vector_store

def create_rag_chain(vector_store):
    """Create RAG chain using modern LangChain LCEL syntax"""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context below.
If you don't know the answer from the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready!")
    return rag_chain

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
 
    if uploaded_file and not st.session_state.file_processed:
        with st.spinner("Processing document..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
 
            documents = load_document(tmp_path)
            vector_store = create_vector_store(documents)
            st.session_state.rag_chain = create_rag_chain(vector_store)
            st.session_state.file_processed = True
            st.session_state.chat_history = []
            os.unlink(tmp_path)
 
        st.success(f"✅ Loaded: {uploaded_file.name}")
 
    if st.session_state.file_processed:
        if st.button("🗑️ Clear & Upload New File"):
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.file_processed = False
            st.rerun()
 
# --- Chat Interface ---
if not st.session_state.file_processed:
    st.info("👈 Upload a PDF or TXT file from the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
 
    user_input = st.chat_input("Ask something about your document...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(user_input)
            st.write(response)
 
        st.session_state.chat_history.append({"role": "assistant", "content": response})
 