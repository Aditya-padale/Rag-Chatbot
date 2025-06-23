# Gemini RAG Chatbot (FastAPI + PDF Knowledge Base)

A modern Retrieval-Augmented Generation (RAG) chatbot using FastAPI, LangChain, Gemini (Google Generative AI), and FAISS. The chatbot answers questions using knowledge extracted from a PDF file, with a beautiful web UI inspired by ChatGPT.

---

## Features
- **Chatbot UI**: Clean, responsive, and modern web interface (HTML/CSS/JS, TailwindCSS style)
- **PDF Knowledge Base**: Upload or use a local PDF as the source of truth for answers
- **RAG Pipeline**: Uses LangChain, Gemini LLM, and FAISS for retrieval-augmented generation
- **File Upload**: Supports PDF, DOCX, and image uploads (text extracted and appended to knowledge base)
- **Fast Startup**: FAISS vectorstore is cached for instant reloads
- **API**: `/chat` endpoint for programmatic access

---

## Getting Started

### 1. Clone the repository
```bash
# Replace with your repo URL
git clone https://github.com/yourusername/gemini-rag-chatbot.git
cd gemini-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key
- Create a `.env` file in the project root:
  ```env
  GOOGLE_API_KEY=your-gemini-api-key-here
  ```

### 4. Add your knowledge base PDF
- Place your `sample.pdf` in the project directory.

### 5. Run the app
```bash
uvicorn rag:app --reload
```
- Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Usage
- **Chat**: Type your question in the chat UI and get answers from your PDF knowledge base.
- **Upload**: Use the upload form to add new PDFs, DOCX, or images. Extracted text is appended to the knowledge base.
- **API**: Send a POST request to `/chat` with `{ "user_id": "your_id", "message": "your question" }`.

---

## Project Structure
```
├── rag.py              # Main FastAPI app
├── requirements.txt    # Python dependencies
├── .env                # Gemini API key
├── sample.pdf          # Your knowledge base PDF
├── templates/
│   └── chat.html       # Web UI
│   └── styles.css      # Custom styles
├── faiss_index/        # Cached FAISS vectorstore (auto-generated)
└── uploads/            # Uploaded files
```

---

## Credits
- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- UI inspired by ChatGPT, Claude, and Gemini

---

## License
MIT
