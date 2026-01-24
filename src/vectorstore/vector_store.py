import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
from collections import OrderedDict
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS-based vector store with LRU eviction"""
    
    def __init__(self):
        self.embedding_dim = config.EMBEDDING_DIM
        self.max_size = config.VECTOR_STORE_MAX_SIZE
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata_store = OrderedDict()
        self.current_idx = 0
        self.logger = logging.getLogger(__name__)
        
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
    
    def add_papers(self, papers: List[Dict], embeddings: np.ndarray):
        """Add papers and embeddings to the store"""
        embeddings_normalized = self.normalize_vectors(embeddings)
        
        for paper, embedding in zip(papers, embeddings_normalized):
            if len(self.metadata_store) >= self.max_size:
                oldest_idx = next(iter(self.metadata_store))
                self.metadata_store.pop(oldest_idx)
                self.logger.info(f"LRU eviction: removed idx {oldest_idx}")
            
            self.index.add(np.array([embedding]))
            paper['vector_idx'] = self.current_idx
            self.metadata_store[self.current_idx] = paper
            self.current_idx += 1
            
        self.logger.info(f"Added {len(papers)} papers. Total: {len(self.metadata_store)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for similar papers"""
        if len(self.metadata_store) == 0:
            self.logger.warning("Vector store is empty")
            return []
            
        query_normalized = self.normalize_vectors(np.array([query_embedding]))
        
        k = min(k, len(self.metadata_store))
        distances, indices = self.index.search(query_normalized, k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx in self.metadata_store and idx != -1:
                paper = self.metadata_store[idx]
                results.append((paper, float(score)))
                
        return results
    
    def save(self, filepath=None):
        """Save index and metadata"""
        if filepath is None:
            filepath = config.VECTOR_STORE_PATH
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'index': faiss.serialize_index(self.index),
            'metadata_store': dict(self.metadata_store),
            'current_idx': self.current_idx,
            'embedding_dim': self.embedding_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved vector store to {filepath}")
    
    def load(self, filepath=None):
        """Load index and metadata"""
        if filepath is None:
            filepath = config.VECTOR_STORE_PATH
            
        if not os.path.exists(filepath):
            self.logger.warning(f"No saved vector store found at {filepath}")
            return
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.index = faiss.deserialize_index(data['index'])
        self.metadata_store = OrderedDict(data['metadata_store'])
        self.current_idx = data['current_idx']
        self.embedding_dim = data['embedding_dim']
        self.logger.info(f"Loaded vector store from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from embedder import PaperEmbedder
    
    embedder = PaperEmbedder()
    vector_store = FAISSVectorStore()
    
    sample_papers = [
        {'title': 'Test Paper 1', 'abstract': 'Machine learning research', 'arxiv_id': '1'},
        {'title': 'Test Paper 2', 'abstract': 'Deep learning models', 'arxiv_id': '2'}
    ]
    
    embeddings = embedder.embed_papers(sample_papers)
    vector_store.add_papers(sample_papers, embeddings)
    
    query_embedding = embedder.embed_query("machine learning")
    results = vector_store.search(query_embedding, k=2)
    
    print(f"Found {len(results)} results")
    for paper, score in results:
        print(f"  {paper['title']}: {score:.3f}")
