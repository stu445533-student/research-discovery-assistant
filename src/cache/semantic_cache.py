import numpy as np
from typing import Optional, Dict
import json
from datetime import datetime
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class SemanticCache:
    """Semantic caching system for LLM responses"""
    
    def __init__(self, embedder):
        self.embedder = embedder
        self.similarity_threshold = config.CACHE_SIMILARITY_THRESHOLD
        self.max_cache_size = config.CACHE_MAX_SIZE
        
        self.cache_queries = []
        self.cache_embeddings = None
        self.cache_responses = []
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, query: str) -> Optional[Dict]:
        """Retrieve cached response if similar query exists"""
        self.stats['total_queries'] += 1
        
        if len(self.cache_queries) == 0:
            self.stats['misses'] += 1
            return None
        
        query_embedding = self.embedder.embed_query(query)
        
        # Compute similarities
        similarities = np.dot(self.cache_embeddings, query_embedding) / (
            np.linalg.norm(self.cache_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        max_similarity = np.max(similarities)
        
        if max_similarity >= self.similarity_threshold:
            best_idx = np.argmax(similarities)
            cached_response = self.cache_responses[best_idx].copy()
            
            self.stats['hits'] += 1
            self.logger.info(f"Cache HIT (sim={max_similarity:.3f})")
            
            cached_response['cache_metadata'] = {
                'cached_query': self.cache_queries[best_idx],
                'similarity': float(max_similarity),
                'hit_time': datetime.now().isoformat()
            }
            
            return cached_response
        
        self.stats['misses'] += 1
        return None
    
    def set(self, query: str, response: Dict, model_type: str = 'flagship'):
        """Store a high-quality response in cache"""
        query_embedding = self.embedder.embed_query(query)
        
        if len(self.cache_queries) >= self.max_cache_size:
            self.cache_queries.pop(0)
            self.cache_responses.pop(0)
            self.cache_embeddings = np.delete(self.cache_embeddings, 0, axis=0)
            self.logger.info("Cache full: removed oldest entry")
        
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
            self.cache_embeddings = np.vstack([self.cache_embeddings, query_embedding])
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        hit_rate = self.stats['hits'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache_queries)
        }
    
    def save(self, filepath=None):
        """Persist cache to disk"""
        if filepath is None:
            filepath = config.CACHE_PATH
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'queries': self.cache_queries,
            'embeddings': self.cache_embeddings.tolist() if self.cache_embeddings is not None else [],
            'responses': self.cache_responses,
            'stats': self.stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved cache to {filepath}")
    
    def load(self, filepath=None):
        """Load cache from disk"""
        if filepath is None:
            filepath = config.CACHE_PATH
            
        if not os.path.exists(filepath):
            self.logger.warning(f"No saved cache found at {filepath}")
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.cache_queries = data['queries']
        self.cache_embeddings = np.array(data['embeddings']) if data['embeddings'] else None
        self.cache_responses = data['responses']
        self.stats = data['stats']
        self.logger.info(f"Loaded cache from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Semantic cache module loaded")
