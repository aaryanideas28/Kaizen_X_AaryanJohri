import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configuration Constants
DATA_DIR = "./data"       
DB_DIR = "./chroma_db"             
LLM_MODEL = "gemini-2.5-flash"

def format_docs(docs):
    """Combines the text of retrieved documents into a single block."""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}' folder. Please drop some files into it!")
        return None
        
    print("🔄 Loading documents from local folder...")
    txt_loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    
    docs = txt_loader.load() + pdf_loader.load()
    
    if not docs:
        print("⚠️ No documents found in your './data' folder. Add some files and rerun.")
        return None

    print("✂️ Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    print("🧠 Initializing Local Embedding Model (No API limits/network bugs)...")
    # Uses a highly efficient model running directly on your CPU/GPU locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🗄️ Indexing chunks into local ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) 
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    
    template = """You are an expert assistant for answering questions based on provided context files.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer or if it's not in the context, say explicitly that the information is not available in the local files. Do not make up an answer.

Context:
{context}

Question: {question}
Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def start_chat():
    rag_chain = initialize_rag()
    if not rag_chain:
        return

    print("\n✅ RAG System Ready! Type 'exit' to quit.\n" + "="*50)
    
    while True:
        query = input("\n🧑‍💻 Ask a question about your files: ")
        if query.strip().lower() == 'exit':
            print("Goodbye!")
            break
            
        if not query.strip():
            continue
            
        print("🔍 Searching and generating answer...")
        answer = rag_chain.invoke(query)
        
        print("\n🤖 Answer:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    start_chat()