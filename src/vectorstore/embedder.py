from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class PaperEmbedder:
    """Generates embeddings for research papers"""
    
    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
    def create_paper_text(self, paper: Dict) -> str:
        """Combine title and abstract for embedding"""
        title = paper.get('title', '').strip()
        abstract = paper.get('abstract', '').strip()
        return f"Title: {title} Abstract: {abstract}"
    
    def embed_papers(self, papers: List[Dict]) -> np.ndarray:
        """Generate embeddings for multiple papers"""
        texts = [self.create_paper_text(p) for p in papers]
        
        self.logger.info(f"Embedding {len(texts)} papers")
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=config.EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True
            )
            self.logger.info(f"Generated embeddings with shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for user query"""
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            self.logger.error(f"Error embedding query: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = PaperEmbedder()
    
    sample_paper = {
        'title': 'Attention Is All You Need',
        'abstract': 'We propose a new simple network architecture'
    }
    
    embeddings = embedder.embed_papers([sample_paper])
    print(f"Embedding shape: {embeddings.shape}")
