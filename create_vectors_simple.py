import os
import pickle
import faiss
import numpy as np
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorizer:
    def __init__(self):
        """Initialize the vectorizer without external APIs"""
        self.embedding_dim = 384
        logger.info("Simple vectorizer initialized")

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        sentences = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            words = sentence.split()
            if current_length + len(words) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_length += len(words)
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding based on text characteristics and keywords"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Feature vector
        features = []
        
        # Basic text statistics (normalized)
        features.append(min(len(text) / 1000.0, 1.0))  # Text length
        features.append(min(len(words) / 50.0, 1.0))   # Word count
        features.append(min(len(set(words)) / len(words) if words else 0, 1.0))  # Vocabulary diversity
        
        # Personal information keywords with weights
        personal_keywords = {
            # Names
            'aditya': 1.0, 'padale': 1.0,
            
            # Education
            'sangli': 0.8, 'highschool': 0.7, 'college': 0.7, 'engineering': 0.8,
            'annasaheb': 0.7, 'dange': 0.7, 'ashta': 0.7, 'cgpa': 0.8,
            'btech': 0.8, 'artificial': 0.8, 'intelligence': 0.8,
            
            # Location
            'kavathepiran': 0.9, 'maharashtra': 0.7,
            
            # Technical skills
            'python': 0.9, 'javascript': 0.8, 'sql': 0.7, 'react': 0.8,
            'nodejs': 0.7, 'fastapi': 0.8, 'postgresql': 0.7, 'mysql': 0.7,
            'langchain': 0.8, 'nfc': 0.6, 'ar': 0.6, 'vr': 0.6,
            
            # Interests
            'dragon': 0.8, 'ball': 0.8, 'anime': 0.7, 'programming': 0.8,
            
            # Personality
            'funny': 0.6, 'witty': 0.6, 'sarcastic': 0.6, 'honest': 0.7,
            'curious': 0.6, 'ambitious': 0.7, 'debugging': 0.8,
            
            # Academic performance
            '89': 0.5, '68': 0.5, '8.9': 0.6, 'marks': 0.5, 'percentage': 0.5,
            
            # Career
            'engineer': 0.7, 'ai': 0.8, 'data': 0.8, 'science': 0.7,
            'real-world': 0.6, 'systems': 0.6,
            
            # Strengths
            'analytical': 0.6, 'learner': 0.6, 'detail': 0.6, 'focused': 0.6
        }
        
        # Score based on keyword presence
        for keyword, weight in personal_keywords.items():
            if keyword in text_lower:
                features.append(weight)
            else:
                features.append(0.0)
        
        # Add semantic features based on content type
        content_types = {
            'education': ['school', 'college', 'grade', 'cgpa', 'marks', 'study'],
            'technical': ['python', 'code', 'programming', 'development', 'api'],
            'personal': ['hobby', 'interest', 'anime', 'dragon', 'ball'],
            'personality': ['funny', 'honest', 'sarcastic', 'witty', 'curious'],
            'career': ['engineer', 'goal', 'aspiring', 'career', 'future'],
            'location': ['maharashtra', 'sangli', 'kavathepiran'],
            'skills': ['debugging', 'analytical', 'learner', 'detail']
        }
        
        for category, keywords in content_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            features.append(score)
        
        # Pad to exact dimension
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        # Truncate if too long
        features = features[:self.embedding_dim]
        
        # Convert to numpy array and normalize
        vector = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            # Fallback to small random vector
            vector = np.random.randn(self.embedding_dim).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
        
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
                embedding = self.create_embedding(chunk)
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
        vectorizer = SimpleVectorizer()
        
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
        
        logger.info("✅ Successfully created and saved Aditya's personal profile vectors!")
        logger.info(f"📊 Index contains {len(result['texts'])} text chunks")
        logger.info(f"🔢 Embedding dimension: {result['metadata']['embedding_dim']}")
        
        # Print some sample chunks for verification
        logger.info("\n📝 Sample chunks created:")
        for i, text in enumerate(result['texts'][:3]):
            logger.info(f"Chunk {i+1}: {text[:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
