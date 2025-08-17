# Policy RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Project-RAG%20Chatbot-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-Private-lightgrey?style=for-the-badge" />
</p>

A secure Retrieval-Augmented Generation (RAG) chatbot that allows users to ask natural-language questions over policy and regulatory PDF documents (e.g., military manuals, DoD regulations, SOPs). Designed for private on-premise environments where sensitive data cannot be sent to the cloud.

---
<img width="836" height="695" alt="Screenshot 2025-08-17 142947" src="https://github.com/user-attachments/assets/e45aa990-b18c-44b0-9f0f-23a98dd09360" />

## 🧠 Features

- Upload policy & procedure PDFs
- Vector-based semantic chunking
- LLM-powered Q&A with context grounding
- FastAPI backend with React/Vite frontend
- Deployable locally, air-gapped, or cloud

---

## 💡 Architecture

| Layer      | Tool / Framework                     |
|-----------|---------------------------------------|
| LLM       | Ollama (Llama-3), OpenAI GPT          |
| Embedding | Sentence-Transformers (`MiniLM`)      |
| Vector DB | Pinecone                              |
| Backend   | FastAPI (Python)                      |
| Frontend  | React + Vite                          |
| PDF       | PyPDF2                                |

---

## 🚀 Local Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # (on Windows)
pip install -r requirements.txt
cp .env.example .env       # add your Pinecone & OpenAI keys
python main.py
