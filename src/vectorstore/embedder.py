from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import logging

class PaperEmbedder:
    """
    Generates embeddings for research papers
    Uses sentence-transformers for academic text
    """
    
    def __init__(self, model_name='allenai/specter'):
        """
        Args:
            model_name: Options include:
                - 'allenai/specter' (specialized for scientific papers)
                - 'sentence-transformers/all-MiniLM-L6-v2' (fast, general)
                - 'malteos/scincl' (scientific papers with contrastive learning)
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
    def create_paper_text(self, paper: Dict) -> str:
        """
        Combine title and abstract for embedding
        Format optimized for academic retrieval
        """
        title = paper.get('title', '').strip()
        abstract = paper.get('abstract', '').strip()
        
        # Format: "Title: [TITLE] Abstract: [ABSTRACT]"
        text = f"Title: {title} Abstract: {abstract}"
        return text
    
    def embed_papers(self, papers: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for multiple papers
        
        Returns:
            numpy array of shape (n_papers, embedding_dim)
        """
        texts = [self.create_paper_text(p) for p in papers]
        
        self.logger.info(f"Embedding {len(texts)} papers with {self.model_name}")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for user query"""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )
        return embedding[0]

# Test embedder
if __name__ == "__main__":
    import json
    
    # Load sample papers
    papers = []
    with open('data/papers.jsonl', 'r') as f:
        for line in f:
            papers.append(json.loads(line))
            if len(papers) >= 10:  # Test with 10 papers
                break
    
    embedder = PaperEmbedder()
    embeddings = embedder.embed_papers(papers)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
