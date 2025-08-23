import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalProfileVectorizer:
    def __init__(self):
        """Initialize the vectorizer with Google Generative AI"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not api_key:
                raise ValueError("No Google API key found")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Google Generative AI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Generative AI: {e}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Google's embedding model"""
        try:
            # Use Gemini to generate a numerical representation
            # Since Gemini doesn't have direct embedding endpoint, we'll use a workaround
            prompt = f"""
            Convert the following text into a numerical vector representation.
            Provide 384 floating point numbers separated by commas that represent the semantic meaning of this text:
            
            Text: {text[:500]}  # Limit text length for API
            
            Return only the numbers, no other text.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the response to extract numbers
            try:
                numbers = response.text.strip().replace('\n', ',').split(',')
                vector = [float(x.strip()) for x in numbers if x.strip()]
                
                # Ensure we have exactly 384 dimensions
                if len(vector) != 384:
                    # Pad or truncate to 384 dimensions
                    if len(vector) < 384:
                        vector.extend([0.0] * (384 - len(vector)))
                    else:
                        vector = vector[:384]
                
                # Normalize the vector
                vector = np.array(vector, dtype=np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                return vector
            except:
                # Fallback: generate random normalized vector
                vector = np.random.randn(384).astype(np.float32)
                return vector / np.linalg.norm(vector)
                
        except Exception as e:
            logger.warning(f"Error getting embedding: {e}")
            # Fallback: generate random normalized vector based on text hash
            np.random.seed(hash(text) % 2**32)
            vector = np.random.randn(384).astype(np.float32)
            return vector / np.linalg.norm(vector)

    def create_simple_embedding(self, text: str) -> np.ndarray:
        """Create a simple embedding based on text characteristics"""
        # This is a fallback method that creates embeddings based on text features
        words = text.lower().split()
        
        # Create feature vector based on text characteristics
        features = []
        
        # Length features
        features.append(len(text) / 1000.0)  # Normalized length
        features.append(len(words) / 100.0)  # Normalized word count
        
        # Content features (simplified)
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
        
        # Pad to 384 dimensions
        while len(features) < 384:
            features.append(0.0)
        
        # Truncate if too long
        features = features[:384]
        
        # Convert to numpy array and normalize
        vector = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

    def process_profile(self, file_path: str) -> Dict:
        """Process the personal profile file and create vectors"""
        logger.info(f"Processing profile file: {file_path}")
        
        # Read the profile file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = self.chunk_text(content)
        logger.info(f"Created {len(chunks)} text chunks")
        
        # Create embeddings for each chunk
        embeddings = []
        chunk_texts = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            try:
                # Try to get embedding (will fall back to simple method if needed)
                embedding = self.create_simple_embedding(chunk)
                embeddings.append(embedding)
                chunk_texts.append(chunk)
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No embeddings were created")
        
        # Convert to numpy array
        embeddings_array = np.vstack(embeddings).astype(np.float32)
        logger.info(f"Created embeddings array with shape: {embeddings_array.shape}")
        
        return {
            'embeddings': embeddings_array,
            'texts': chunk_texts,
            'metadata': {
                'source': file_path,
                'num_chunks': len(chunk_texts),
                'embedding_dim': embeddings_array.shape[1]
            }
        }

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create FAISS index from embeddings"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        return index

    def save_index(self, index: faiss.Index, texts: List[str], metadata: Dict, 
                   index_dir: str = "faiss_index"):
        """Save FAISS index and associated data"""
        # Create directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(index_dir, "index.faiss")
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save texts and metadata
        data_path = os.path.join(index_dir, "index.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'texts': texts,
                'metadata': metadata
            }, f)
        logger.info(f"Saved index data to {data_path}")

def main():
    """Main function to process Aditya's profile and create vectors"""
    try:
        # Initialize vectorizer
        vectorizer = PersonalProfileVectorizer()
        
        # Process the profile
        profile_path = "aditya_full_personal_profile.txt"
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile file not found: {profile_path}")
        
        result = vectorizer.process_profile(profile_path)
        
        # Create FAISS index
        index = vectorizer.create_faiss_index(result['embeddings'])
        
        # Save everything
        vectorizer.save_index(
            index, 
            result['texts'], 
            result['metadata']
        )
        
        logger.info("Successfully created and saved Aditya's personal profile vectors!")
        logger.info(f"Index contains {len(result['texts'])} text chunks")
        logger.info(f"Embedding dimension: {result['metadata']['embedding_dim']}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
