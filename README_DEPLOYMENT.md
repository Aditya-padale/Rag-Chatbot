# 🤖 Modern AI Assistant - RAG Chatbot

A sleek, modern, and aesthetic Retrieval-Augmented Generation (RAG) chatbot with a beautiful UI, powered by FastAPI, LangChain, Google Gemini, and FAISS.

![AI Assistant](https://img.shields.io/badge/AI-Assistant-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green) ![RAG](https://img.shields.io/badge/RAG-Technology-purple)

## ✨ Features

### 🎨 **Modern & Aesthetic UI**
- **Sleek glass-morphism design** with animated gradients
- **Responsive layout** that works on all devices
- **Real-time typing indicators** and smooth animations
- **Connection status monitoring** with visual indicators
- **Dark theme** with purple/blue accent colors
- **Interactive elements** with hover effects and transitions

### 🧠 **Advanced AI Capabilities**
- **RAG Technology** - Retrieval-Augmented Generation for accurate responses
- **Google Gemini Integration** - Powered by latest AI models
- **Document Processing** - PDF, DOCX, and image text extraction
- **FAISS Vector Store** - Fast similarity search and caching
- **Context-Aware Conversations** - Maintains chat history

### 🚀 **Production Ready**
- **Error Handling** - Comprehensive error management with retry logic
- **Rate Limiting** - API protection and quota management
- **CORS Configuration** - Ready for cross-origin deployment
- **Health Monitoring** - Real-time server status checks
- **Scalable Architecture** - Optimized for cloud deployment

---

## 🏗️ **Deployment Guide**

### **Backend Deployment (Render)**

1. **Fork/Clone this repository**
2. **Connect to Render**:
   - Go to [render.com](https://render.com) and create account
   - Connect your GitHub repository
   - Choose "Web Service"

3. **Configuration**:
   ```
   Build Command: ./build.sh
   Start Command: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
   ```

4. **Environment Variables** (Set in Render Dashboard):
   ```
   GOOGLE_API_KEY=your-gemini-api-key-here
   GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key-here
   PYTHON_VERSION=3.11.0
   ```

5. **Deploy**: Render will automatically deploy your app!

### **Frontend Deployment (Vercel)**

1. **Deploy the `/vercel-frontend` folder to Vercel**:
   ```bash
   cd vercel-frontend
   npx vercel --prod
   ```

2. **Update the API URL**:
   - After backend is deployed, update line 270 in `index.html`:
   ```javascript
   const API_BASE_URL = 'https://your-render-app.onrender.com';
   ```

3. **Custom Domain** (Optional):
   - Configure custom domain in Vercel dashboard

---

## 🛠️ **Local Development**

### **Prerequisites**
- Python 3.11+
- Google Gemini API Key ([Get one here](https://ai.google.dev/))

### **Setup**

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Aditya-padale/Rag-Chatbot.git
   cd Rag-Chatbot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**:
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your-api-key-here" > .env
   echo "GOOGLE_GENERATIVE_AI_API_KEY=your-api-key-here" >> .env
   ```

4. **Add Your Knowledge Base**:
   - Place your `aditya_full_personal_profile.txt` in project root
   - Or update the file path in `main.py`

5. **Run Application**:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

6. **Access**: Open `http://127.0.0.1:8000` in your browser

---

## 📁 **Project Structure**

```
Rag-Chatbot/
├── 🔧 Backend (FastAPI)
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables
│   ├── build.sh               # Render build script
│   ├── Procfile               # Heroku/Render process
│   └── render.yaml            # Render configuration
│
├── 🎨 Templates
│   ├── chat.html              # Modern UI template
│   └── styles.css             # Additional styles
│
├── 🌐 Frontend (Vercel)
│   ├── vercel-frontend/
│   │   ├── index.html         # Standalone frontend
│   │   ├── vercel.json        # Vercel configuration
│   │   └── package.json       # Node.js metadata
│
├── 📊 Data
│   ├── faiss_index/           # Vector store cache
│   ├── uploads/               # Uploaded files
│   └── *.txt                  # Knowledge base files
│
└── 📚 Documentation
    └── README.md              # This file
```

---

## 🎯 **Key Improvements Made**

### **🎨 UI/UX Enhancements**
- **Glass-morphism Design**: Modern frosted glass effect with backdrop blur
- **Animated Gradients**: Dynamic background with color transitions
- **Interactive Elements**: Hover effects, button animations, and micro-interactions
- **Real-time Status**: Connection monitoring with colored indicators
- **Responsive Design**: Mobile-first approach with breakpoints
- **Typography**: Premium fonts (Inter + JetBrains Mono) for better readability

### **⚡ Performance Optimizations**
- **Rate Limiting**: Prevents API quota exhaustion
- **Memory Management**: Limited conversation history (10 exchanges)
- **Async Operations**: Non-blocking API calls with proper error handling
- **Caching**: FAISS vector store caching for instant reloads
- **Lazy Loading**: Optimized resource loading

### **🔒 Production Readiness**
- **Error Boundaries**: Comprehensive error handling with user-friendly messages
- **CORS Configuration**: Proper cross-origin setup for deployment
- **Health Checks**: Monitoring endpoints for uptime tracking
- **Logging**: Structured logging for debugging and monitoring
- **Environment Management**: Secure configuration handling

---

## 🚦 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the chat interface |
| `/chat` | POST | Process chat messages |
| `/health` | GET | Health check endpoint |
| `/upload` | POST | Upload documents |

### **Chat API Example**:
```javascript
POST /chat
{
  "user_id": "web_user",
  "message": "What is RAG technology?"
}

Response:
{
  "response": "RAG (Retrieval-Augmented Generation) is..."
}
```

---

## 🔑 **Environment Variables**

```env
# Required
GOOGLE_API_KEY=your-gemini-api-key
GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key

# Optional (for other integrations)
OPENAI_API_KEY=your-openai-key
HUGGINGFACEHUB_API_TOKEN=your-hf-token
```

---

## 🌟 **Technology Stack**

### **Backend**
- **FastAPI**: Modern Python web framework
- **LangChain**: AI application framework
- **Google Gemini**: Large language model
- **FAISS**: Vector similarity search
- **Uvicorn**: ASGI server
- **Gunicorn**: Production WSGI server

### **Frontend**
- **Vanilla JavaScript**: No framework dependencies
- **TailwindCSS**: Utility-first CSS framework
- **Font Awesome**: Icon library
- **Google Fonts**: Premium typography

### **Deployment**
- **Render**: Backend hosting
- **Vercel**: Frontend hosting
- **Git**: Version control

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 **Author**

**Aditya Padale**
- GitHub: [@Aditya-padale](https://github.com/Aditya-padale)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

---

## ⭐ **Show your support**

Give a ⭐️ if this project helped you!

---

## 📞 **Support**

If you have any questions or need help with deployment:
- Open an issue on GitHub
- Contact via LinkedIn
- Email: your-email@domain.com

---

**Happy Coding! 🚀✨**
