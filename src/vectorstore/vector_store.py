import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
from collections import OrderedDict
import logging

class FAISSVectorStore:
    """
    FAISS-based vector store with LRU eviction
    Supports fast approximate nearest neighbor search
    """
    
    def __init__(self, embedding_dim=768, max_size=10000):
        """
        Args:
            embedding_dim: Dimension of embeddings
            max_size: Maximum number of vectors (LRU eviction)
        """
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        
        # Initialize FAISS index (Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Metadata storage with LRU
        self.metadata_store = OrderedDict()  # {idx: paper_dict}
        self.current_idx = 0
        
        self.logger = logging.getLogger(__name__)
        
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
    
    def add_papers(self, papers: List[Dict], embeddings: np.ndarray):
        """
        Add papers and their embeddings to the store
        Implements LRU eviction if max_size exceeded
        """
        embeddings_normalized = self.normalize_vectors(embeddings)
        
        for paper, embedding in zip(papers, embeddings_normalized):
            # Check if we need to evict
            if len(self.metadata_store) >= self.max_size:
                oldest_idx = next(iter(self.metadata_store))
                self.metadata_store.pop(oldest_idx)
                self.logger.info(f"LRU eviction: removed idx {oldest_idx}")
            
            # Add to FAISS
            self.index.add(np.array([embedding]))
            
            # Add metadata
            paper['vector_idx'] = self.current_idx
            self.metadata_store[self.current_idx] = paper
            self.current_idx += 1
            
        self.logger.info(f"Added {len(papers)} papers. Total: {len(self.metadata_store)}")
    
    def search(self, query_embedding: np.ndarray, 
               k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Search for similar papers
        
        Returns:
            List of (paper_dict, similarity_score) tuples
        """
        query_normalized = self.normalize_vectors(
            np.array([query_embedding])
        )
        
        # Search FAISS index
        distances, indices = self.index.search(query_normalized, k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx in self.metadata_store:
                paper = self.metadata_store[idx]
                results.append((paper, float(score)))
                
        return results
    
    def save(self, filepath='data/vector_store.pkl'):
        """Save index and metadata"""
        data = {
            'index': faiss.serialize_index(self.index),
            'metadata_store': dict(self.metadata_store),
            'current_idx': self.current_idx,
            'embedding_dim': self.embedding_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved vector store to {filepath}")
    
    def load(self, filepath='data/vector_store.pkl'):
        """Load index and metadata"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.index = faiss.deserialize_index(data['index'])
        self.metadata_store = OrderedDict(data['metadata_store'])
        self.current_idx = data['current_idx']
        self.embedding_dim = data['embedding_dim']
        self.logger.info(f"Loaded vector store from {filepath}")

# Test vector store
if __name__ == "__main__":
    from embedder import PaperEmbedder
    import json
    
    # Load papers
    papers = []
    with open('data/papers.jsonl', 'r') as f:
        for line in f:
            papers.append(json.loads(line))
    
    # Create embeddings
    embedder = PaperEmbedder()
    embeddings = embedder.embed_papers(papers)
    
    # Initialize and populate vector store
    vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_papers(papers, embeddings)
    
    # Test search
    query = "machine learning for natural language processing"
    query_embedding = embedder.embed_query(query)
    results = vector_store.search(query_embedding, k=5)
    
    print("\nTop 5 results:")
    for i, (paper, score) in enumerate(results):
        print(f"{i+1}. [{score:.3f}] {paper['title']}")
