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

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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

def chat(rag_chain):
    """Interactive chat loop"""
    print("\n" + "="*50)
    print("🤖 RAG Chatbot Ready! Ask questions about your document.")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye! 👋")
            break
        if not question:
            continue

        answer = rag_chain.invoke(question)
        print(f"\n🤖 Bot: {answer}\n")

def main():
    file_path = "document.txt"  # or "document.pdf"

    print("🚀 Starting RAG Chatbot...\n")
    documents = load_document(file_path)
    vector_store = create_vector_store(documents)
    rag_chain = create_rag_chain(vector_store)
    chat(rag_chain)

if __name__ == "__main__":
    main()