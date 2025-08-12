import os
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import logging
from typing import Dict, Any
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

# Initialize the LLM
try:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError("No Google API key found")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        google_api_key=api_key,
        max_output_tokens=1000
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

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
    return {"status": "healthy", "llm_available": llm is not None}

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage, request: Request):
    """Handle chat requests"""
    client_ip = request.client.host
    
    try:
        # Check rate limiting
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please wait before sending another message."
            )
        
        # Check if LLM is available
        if not llm:
            raise HTTPException(
                status_code=503,
                detail="AI service is currently unavailable. Please try again later."
            )
        
        # Get user message
        user_message = chat_message.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Create a simple prompt for the LLM
        prompt = f"""You are a helpful AI assistant. Please provide a clear and helpful response to the following question:

Question: {user_message}

Please provide a comprehensive answer."""
        
        # Get response from LLM with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Run in executor to handle potential blocking calls
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
                
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
                
                return JSONResponse({
                    "response": answer,
                    "status": "success"
                })
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return JSONResponse(
            {"error": "An error occurred while processing your request", "status": "error"}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
