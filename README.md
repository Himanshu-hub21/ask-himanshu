# Ask Himanshu - Personal AI Assistant

AI assistant trained on my resume and research papers using RAG (Retrieval Augmented Generation).

## 🚀 Live Demo
[https://ask-himanshu.streamlit.app](https://ask-himanshu.streamlit.app)

## 🛠️ Tech Stack
- **LLM:** Llama-3.1-8B (Groq API)
- **Vector DB:** Qdrant Cloud
- **Embeddings:** TF-IDF + LSA
- **Frontend:** Streamlit

## 🔒 Privacy
Personal contact information is protected. Contact requests redirect to email.

## 💻 Local Setup
```bash
pip install -r requirements.txt
# Add .env with API keys
python ingest_data.py  # One-time
streamlit run app.py
```

## 📧 Contact
hsramteke21@gmail.com