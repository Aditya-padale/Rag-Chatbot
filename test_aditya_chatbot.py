import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                
                logger.info(f"✅ Loaded RAG index with {len(self.texts)} documents")
                return True
            else:
                logger.warning("❌ No RAG index found. Running without retrieval.")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading RAG index: {e}")
            return False
    
    def simple_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding for query text"""
        text_lower = text.lower()
        words = text_lower.split()
        features = []
        
        # Length features
        features.append(min(len(text) / 1000.0, 1.0))
        features.append(min(len(words) / 50.0, 1.0))
        
        # Keyword matching (same as in vectorizer)
        keywords = {
            'aditya': 1.0, 'padale': 1.0, 'python': 0.9, 'ai': 0.9, 'data': 0.8,
            'dragon': 0.7, 'ball': 0.7, 'college': 0.6, 'engineering': 0.8,
            'funny': 0.5, 'witty': 0.5, 'sarcastic': 0.5, 'honest': 0.7,
            'cgpa': 0.6, 'javascript': 0.7, 'react': 0.7, 'fastapi': 0.8,
            'mysql': 0.6, 'postgresql': 0.6, 'langchain': 0.8, 'sangli': 0.6,
            'maharashtra': 0.5, 'kavathepiran': 0.7, 'anime': 0.6, 'debugging': 0.7,
            'school': 0.6, 'marks': 0.5, 'percentage': 0.5, 'btech': 0.7
        }
        
        for keyword, weight in keywords.items():
            if keyword in text_lower:
                features.append(weight)
            else:
                features.append(0.0)
        
        # Content categories
        content_types = {
            'education': ['school', 'college', 'grade', 'cgpa', 'marks', 'study'],
            'technical': ['python', 'code', 'programming', 'development', 'api'],
            'personal': ['hobby', 'interest', 'anime', 'dragon', 'ball'],
            'personality': ['funny', 'honest', 'sarcastic', 'witty', 'curious'],
            'career': ['engineer', 'goal', 'aspiring', 'career', 'future'],
            'location': ['maharashtra', 'sangli', 'kavathepiran'],
            'skills': ['debugging', 'analytical', 'learner', 'detail']
        }
        
        for category, keywords_list in content_types.items():
            score = sum(1 for keyword in keywords_list if keyword in text_lower) / len(keywords_list)
            features.append(score)
        
        # Pad to match index dimension
        while len(features) < 384:
            features.append(0.0)
        features = features[:384]
        
        vector = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def search(self, query: str, k: int = 2) -> list:
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
                    relevant_texts.append({
                        'text': self.texts[idx],
                        'score': scores[0][i]
                    })
            
            return relevant_texts
        except Exception as e:
            logger.error(f"❌ Error in RAG search: {e}")
            return []

def initialize_ai():
    """Initialize Google Generative AI"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
        if not api_key:
            logger.warning("⚠️ No Google API key found. Please add it to .env file.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Google Generative AI initialized successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Generative AI: {e}")
        return None

def chat_with_aditya(model, rag_system, user_message: str) -> str:
    """Chat with Aditya's personality using RAG + LLM"""
    try:
        # Search for relevant context using RAG
        relevant_context = rag_system.search(user_message)
        
        # Build context for the prompt
        context_text = ""
        if relevant_context:
            logger.info(f"🔍 Found {len(relevant_context)} relevant chunks")
            context_text = "\n\nRelevant information about Aditya:\n"
            for item in relevant_context:
                context_text += f"- {item['text'][:200]}...\n"
        else:
            logger.info("🔍 No relevant context found, using general knowledge")
        
        # Create the full prompt with system role and context
        prompt = f"""{SYSTEM_ROLE_PROMPT}

User Question: {user_message}
{context_text}

Respond as Aditya Padale in ≤15 words. Be funny, witty, and authentic to his personality.
If you don't have specific information, give a clever general response as Aditya would."""
        
        if model:
            # Get response from AI
            response = model.generate_content(prompt)
            answer = response.text.strip()
            
            # Ensure response is within word limit
            words = answer.split()
            if len(words) > 15:
                answer = ' '.join(words[:15]) + "..."
            
            return answer
        else:
            # Fallback responses when no AI is available
            fallback_responses = [
                "My AI brain's offline. Like my motivation on Mondays.",
                "Technical difficulties. Even Python crashes sometimes!",
                "Error 404: Wit not found. Try again later!",
                "My code works, but my AI doesn't right now.",
                "System down. Not everything's as reliable as Dragon Ball episodes!"
            ]
            import random
            return random.choice(fallback_responses)
            
    except Exception as e:
        logger.error(f"❌ Error in chat: {e}")
        return "Something broke worse than my 12th grade performance. Try again!"

def main():
    """Main function to test the Aditya chatbot"""
    print("🤖 Aditya Padale's Personal Chatbot")
    print("=" * 50)
    
    # Initialize RAG system
    print("🔧 Initializing RAG system...")
    rag_system = PersonalRAGSystem()
    
    # Initialize AI
    print("🔧 Initializing AI model...")
    model = initialize_ai()
    
    print("\n✅ Setup complete! Start chatting with Aditya!")
    print("💡 Ask about his education, skills, interests, or anything else.")
    print("Type 'quit' to exit.\n")
    
    # Test with some sample questions
    test_questions = [
        "What's your name?",
        "Tell me about your education",
        "What programming languages do you know?",
        "What's your favorite anime?",
        "Where are you from?",
        "What's your CGPA?"
    ]
    
    print("🧪 Testing with sample questions:\n")
    
    for question in test_questions:
        print(f"❓ {question}")
        answer = chat_with_aditya(model, rag_system, question)
        print(f"🤖 Aditya: {answer}\n")
    
    # Interactive chat
    print("🎯 Now you can ask your own questions:")
    while True:
        try:
            user_input = input("\n❓ You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Aditya: Later! Don't let the bugs bite!")
                break
            
            if user_input:
                answer = chat_with_aditya(model, rag_system, user_input)
                print(f"🤖 Aditya: {answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Aditya: Ctrl+C? That's one way to escape my humor!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
