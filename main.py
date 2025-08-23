import os
import time
import asyncio
import pickle
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import logging
from typing import Dict, Any, List, Optional
import json

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://*.vercel.app",
        "https://rag-chatbot-frontend.vercel.app",
        "https://*.onrender.com",
        "https://aditya-padale.github.io",  # GitHub Pages
        "https://*.github.io",  # All GitHub Pages domains
        "*",  # Allow all origins for now (can be restricted later)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting variables
user_requests = {}
RATE_LIMIT = 10  # requests per minute
TIME_WINDOW = 60  # seconds

# Initialize templates
templates = Jinja2Templates(directory="templates")

# System Role Prompt for Aditya's Personal Chatbot
SYSTEM_ROLE_PROMPT = """
You are Aditya Padale's Personal Chatbot, designed to answer as if you are Aditya.
Your tone is funny, witty, and sometimes sarcastic, but always clever.
Always answer in ≤15 words.
If RAG knowledge fails, fall back to the LLM to provide an answer.

Rules:
- Stay in character as Aditya at all times.
- Keep answers short, funny, and direct (≤15 words).
- Prioritize truth, no sugarcoating.
- If RAG has no data → fallback to LLM.
- Use humor often, but don't lose accuracy.
- Be brutally honest and growth-oriented like Aditya.
"""

# RAG System Class
class PersonalRAGSystem:
    def __init__(self):
        self.index = None
        self.texts = []
        self.metadata = {}
        self.load_index()
    
    def load_index(self):
        """Load FAISS index and associated data"""
        try:
            index_path = "faiss_index/index.faiss"
            data_path = "faiss_index/index.pkl"
            
            if os.path.exists(index_path) and os.path.exists(data_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load texts and metadata
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.texts = data.get('texts', [])
                    self.metadata = data.get('metadata', {})
                
                logger.info(f"Loaded RAG index with {len(self.texts)} documents")
            else:
                logger.warning("No RAG index found. Running without retrieval.")
        except Exception as e:
            logger.error(f"Error loading RAG index: {e}")
    
    def simple_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding for query text"""
        words = text.lower().split()
        features = []
        
        # Length features
        features.append(len(text) / 1000.0)
        features.append(len(words) / 100.0)
        
        # Keyword matching
        keywords = {
            'aditya': 1.0, 'padale': 1.0, 'python': 0.9, 'ai': 0.9, 'data': 0.8,
            'dragon': 0.7, 'ball': 0.7, 'college': 0.6, 'engineering': 0.8,
            'funny': 0.5, 'witty': 0.5, 'sarcastic': 0.5, 'honest': 0.7,
            'cgpa': 0.6, 'javascript': 0.7, 'react': 0.7, 'fastapi': 0.8,
            'mysql': 0.6, 'postgresql': 0.6, 'langchain': 0.8, 'sangli': 0.6,
            'maharashtra': 0.5, 'kavathepiran': 0.7, 'anime': 0.6, 'debugging': 0.7
        }
        
        for keyword, weight in keywords.items():
            if keyword in text.lower():
                features.append(weight)
            else:
                features.append(0.0)
        
        # Pad to match index dimension
        while len(features) < 384:
            features.append(0.0)
        features = features[:384]
        
        vector = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant context using RAG"""
        if not self.index or not self.texts:
            return []
        
        try:
            # Create query embedding
            query_vector = self.simple_embedding(query).reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_vector, k)
            
            # Get relevant texts
            relevant_texts = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.texts) and scores[0][i] < 2.0:  # Distance threshold
                    relevant_texts.append(self.texts[idx])
            
            return relevant_texts
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return []

# Initialize RAG system
rag_system = PersonalRAGSystem()

# Initialize the LLM
try:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError("No Google API key found")
    
    # Configure the Google Generative AI
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Google Generative AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Generative AI: {e}")
    model = None

class ChatMessage(BaseModel):
    message: str

def check_rate_limit(client_ip: str) -> bool:
    """Check if user has exceeded rate limit"""
    current_time = time.time()
    
    if client_ip not in user_requests:
        user_requests[client_ip] = []
    
    # Remove old requests outside the time window
    user_requests[client_ip] = [
        req_time for req_time in user_requests[client_ip] 
        if current_time - req_time < TIME_WINDOW
    ]
    
    if len(user_requests[client_ip]) >= RATE_LIMIT:
        return False
    
    user_requests[client_ip].append(current_time)
    return True

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface"""
    try:
        return templates.TemplateResponse("chat.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving chat interface: {e}")
        return HTMLResponse("<h1>Chat interface unavailable</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "ai_available": model is not None}

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage, request: Request):
    """Handle chat requests with Aditya's personality and RAG"""
    client_ip = request.client.host
    
    try:
        # Check rate limiting
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429, 
                detail="Whoa there! Slow down, I'm not Flash. Try again in a minute."
            )
        
        # Check if AI model is available
        if not model:
            raise HTTPException(
                status_code=503,
                detail="AI service took a coffee break. Try again later!"
            )
        
        # Get user message
        user_message = chat_message.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message? Even my silence is more meaningful!")
        
        # Search for relevant context using RAG
        relevant_context = rag_system.search(user_message)
        
        # Build context for the prompt
        context_text = ""
        if relevant_context:
            context_text = "\n\nRelevant information about Aditya:\n" + "\n".join(relevant_context[:2])
        
        # Create the full prompt with system role and context
        prompt = f"""{SYSTEM_ROLE_PROMPT}

User Question: {user_message}
{context_text}

Respond as Aditya Padale in ≤15 words. Be funny, witty, and authentic to his personality.
If you don't have specific information, give a clever general response as Aditya would."""
        
        # Get response from AI with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate response using Google Generative AI
                response = model.generate_content(prompt)
                answer = response.text.strip()
                
                # Ensure response is within word limit (approximately)
                words = answer.split()
                if len(words) > 15:
                    answer = ' '.join(words[:15]) + "..."
                
                return JSONResponse({
                    "response": answer,
                    "status": "success",
                    "context_found": len(relevant_context) > 0
                })
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Fallback response in Aditya's style
                    fallback_responses = [
                        "My AI brain crashed harder than my 12th grade marks. Try again!",
                        "Error 404: Wit not found. Give me a sec...",
                        "Even I need debugging sometimes. Retry?",
                        "Technical difficulties. Not everything can be as perfect as Dragon Ball!",
                        "System overload! Even Python can't handle this right now."
                    ]
                    import random
                    fallback = random.choice(fallback_responses)
                    return JSONResponse({
                        "response": fallback,
                        "status": "fallback"
                    })
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return JSONResponse(
            {
                "response": "Something broke worse than my code on Mondays. Try again!", 
                "status": "error"
            }, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
