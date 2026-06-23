# KAIZEN X - RAG-Powered File Chatbot

An intelligent, context-aware RAG (Retrieval-Augmented Generation) chatbot built using LangChain, ChromaDB, and Google Gemini. The system reads local text and PDF documents, extracts their content into a local vector database using lightweight local sentence-transformers, and provides a conversational terminal interface to query information directly from your documents.

## 🚀 Features
- **Flexible Document Processing:** Automatically parses both `.txt` and `.pdf` files from a targeted directory.
- **Local Embedding Pipeline:** Uses `sentence-transformers/all-MiniLM-L6-v2` running locally on device for deterministic chunk indexing.
- **Persistent Vector Storage:** Leverages ChromaDB to index text splits onto disk so data doesn't re-index every session.
- **Grounded LLM Invocations:** Pipelines retrieved context splits into a strict grounding prompt sent to `gemini-2.5-flash` to prevent hallucinations.

---

## 🛠️ Setup Instructions

### 1. Clone the Workspace
```bash
git clone https://github.com/aaryanideas28/Kaizen_X_AaryanJohri.git
cd Kaizen_X_AaryanJohri
