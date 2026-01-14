import numpy as np
from typing import Optional, Dict, Tuple
import json
import time
from datetime import datetime
import logging
from src.vectorstore.embedder import PaperEmbedder

class SemanticCache:
    """
    Semantic caching system for LLM responses
    Uses cosine similarity to match similar queries
    """
    
    def __init__(self, embedder: PaperEmbedder, 
                 similarity_threshold: float = 0.92,
                 max_cache_size: int = 1000):
        """
        Args:
            embedder: Embedder for query vectorization
            similarity_threshold: Min similarity for cache hit
            max_cache_size: Maximum cached responses
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Cache storage
        self.cache_queries = []  # List of query strings
        self.cache_embeddings = None  # numpy array of embeddings
        self.cache_responses = []  # List of response dicts
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, query: str) -> Optional[Dict]:
        """
        Retrieve cached response if similar query exists
        
        Returns:
            Cached response dict or None
        """
        self.stats['total_queries'] += 1
        
        if len(self.cache_queries) == 0:
            self.stats['misses'] += 1
            return None
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Compute similarities
        similarities = np.dot(
            self.cache_embeddings, 
            query_embedding
        ) / (
            np.linalg.norm(self.cache_embeddings, axis=1) * 
            np.linalg.norm(query_embedding)
        )
        
        # Find best match
        max_similarity = np.max(similarities)
        
        if max_similarity >= self.similarity_threshold:
            best_idx = np.argmax(similarities)
            cached_response = self.cache_responses[best_idx]
            
            self.stats['hits'] += 1
            self.logger.info(
                f"Cache HIT (sim={max_similarity:.3f}): {query[:50]}..."
            )
            
            # Update metadata
            cached_response['cache_metadata'] = {
                'cached_query': self.cache_queries[best_idx],
                'similarity': float(max_similarity),
                'hit_time': datetime.now().isoformat()
            }
            
            return cached_response
        
        self.stats['misses'] += 1
        return None
    
    def set(self, query: str, response: Dict, model_type: str = 'flagship'):
        """
        Store a high-quality response in cache
        
        Args:
            query: Original query string
            response: Response dict with 'answer' and metadata
            model_type: 'flagship' or 'low_cost'
        """
        # Generate embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Check cache size limit
        if len(self.cache_queries) >= self.max_cache_size:
            # Remove oldest entry (FIFO)
            self.cache_queries.pop(0)
            self.cache_responses.pop(0)
            self.cache_embeddings = np.delete(
                self.cache_embeddings, 0, axis=0
            )
            self.logger.info("Cache full: removed oldest entry")
        
        # Add to cache
        self.cache_queries.append(query)
        self.cache_responses.append({
            'answer': response.get('answer', ''),
            'model': response.get('model', 'unknown'),
            'model_type': model_type,
            'cached_at': datetime.now().isoformat(),
            'token_count': response.get('token_count', 0)
        })
        
        if self.cache_embeddings is None:
            self.cache_embeddings = np.array([query_embedding])
        else:
            self.cache_embeddings = np.vstack([
                self.cache_embeddings, 
                query_embedding
            ])
        
        self.logger.info(f"Cached response for: {query[:50]}...")
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        hit_rate = (
            self.stats['hits'] / self.stats['total_queries'] 
            if self.stats['total_queries'] > 0 else 0
        )
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache_queries)
        }
    
    def save(self, filepath='data/semantic_cache.json'):
        """Persist cache to disk"""
        data = {
            'queries': self.cache_queries,
            'embeddings': self.cache_embeddings.tolist() if self.cache_embeddings is not None else [],
            'responses': self.cache_responses,
            'stats': self.stats
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        self.logger.info(f"Saved cache to {filepath}")
    
    def load(self, filepath='data/semantic_cache.json'):
        """Load cache from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.cache_queries = data['queries']
        self.cache_embeddings = np.array(data['embeddings']) if data['embeddings'] else None
        self.cache_responses = data['responses']
        self.stats = data['stats']
        self.logger.info(f"Loaded cache from {filepath}")
