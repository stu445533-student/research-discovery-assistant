"""
Redis Vector Store - The ONLY vector store implementation
"""
import redis
import numpy as np
import logging
import json
from typing import List, Dict, Tuple
import os
import sys

# Safe imports for Redis
try:
    from redis.commands.search.field import VectorField, TextField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError:
    from redis.commands.search.field import VectorField, TextField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Redis-based vector store
    This is the ONLY vector store implementation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.index_name = config.REDIS_INDEX_NAME
        self.dim = config.EMBEDDING_DIM
        
        try:
            self.client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                decode_responses=False 
            )
            self.client.ping()
            self.logger.info("✓ Connected to Redis")
            self._create_index_if_not_exists()
            
        except Exception as e:
            self.logger.error(f"✗ Failed to connect to Redis: {e}")
            raise

    def _create_index_if_not_exists(self):
        """Create the Search Index schema"""
        try:
            self.client.ft(self.index_name).info()
            self.logger.info(f"✓ Index '{self.index_name}' exists")
        except redis.exceptions.ResponseError:
            self.logger.info(f"Creating index '{self.index_name}'...")
            
            schema = (
                TextField("title"),
                TextField("abstract"),
                TextField("arxiv_id"),
                TextField("authors"),
                TextField("categories"),
                VectorField(
                    "vector",
                    "FLAT", 
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dim,
                        "DISTANCE_METRIC": "COSINE",
                    }
                ),
            )
            
            definition = IndexDefinition(
                prefix=["paper:"], 
                index_type=IndexType.HASH
            )
            
            self.client.ft(self.index_name).create_index(
                fields=schema, 
                definition=definition
            )
            self.logger.info(f"✓ Created index '{self.index_name}'")

    def add_papers(self, papers: List[Dict], embeddings: np.ndarray):
        """Add papers and embeddings to Redis"""
        pipeline = self.client.pipeline()
        
        for i, paper in enumerate(papers):
            key = f"paper:{paper['arxiv_id']}"
            embedding = embeddings[i].astype(np.float32).tobytes()
            
            data = {
                "title": paper.get('title', ''),
                "abstract": paper.get('abstract', ''),
                "arxiv_id": paper.get('arxiv_id', ''),
                "authors": json.dumps(paper.get('authors', [])),
                "categories": json.dumps(paper.get('categories', [])),
                "vector": embedding
            }
            
            pipeline.hset(key, mapping=data)
            
        pipeline.execute()
        self.logger.info(f"✓ Added {len(papers)} papers to Redis")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for similar papers using Redis Vector Search"""
        query_vec = query_embedding.astype(np.float32).tobytes()
        
        q = Query(f"*=>[KNN {k} @vector $vec AS score]")\
            .sort_by("score")\
            .return_fields("title", "abstract", "arxiv_id", "authors", "categories", "score")\
            .dialect(2)
            
        params = {"vec": query_vec}
        
        try:
            results = self.client.ft(self.index_name).search(q, query_params=params)
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

        formatted_results = []
        for doc in results.docs:
            distance = float(doc.score)
            similarity = 1 - distance 
            
            paper = {
                'title': self._decode(doc.title),
                'abstract': self._decode(doc.abstract),
                'arxiv_id': self._decode(doc.arxiv_id),
                'authors': json.loads(self._decode(doc.authors)),
                'categories': json.loads(self._decode(doc.categories))
            }
            formatted_results.append((paper, similarity))
            
        return formatted_results
    
    def _decode(self, value):
        """Helper to decode bytes to string"""
        if isinstance(value, bytes):
            return value.decode('utf-8')
        return value
    
    def load_all_papers(self) -> List[Dict]:
        """
        Load all papers from Redis
        Used for topic extraction and stats
        """
        papers = []
        
        try:
            cursor = 0
            count = 0
            
            while True:
                cursor, keys = self.client.scan(
                    cursor=cursor, 
                    match="paper:*", 
                    count=100
                )
                
                for key in keys:
                    try:
                        data = self.client.hgetall(key)
                        if data:
                            paper = {
                                'title': self._decode(data.get(b'title', b'')),
                                'abstract': self._decode(data.get(b'abstract', b'')),
                                'arxiv_id': self._decode(data.get(b'arxiv_id', b'')),
                                'authors': json.loads(self._decode(data.get(b'authors', b'[]'))),
                                'categories': json.loads(self._decode(data.get(b'categories', b'[]')))
                            }
                            papers.append(paper)
                            count += 1
                    except Exception as e:
                        self.logger.warning(f"Error loading paper {key}: {e}")
                        continue
                
                if cursor == 0:
                    break
            
            self.logger.info(f"✓ Loaded {len(papers)} papers from Redis")
            
        except Exception as e:
            self.logger.error(f"Error loading papers: {e}")
        
        return papers
    
    def count_papers(self) -> int:
        """Count total papers in Redis"""
        try:
            keys = self.client.keys("paper:*")
            return len(keys)
        except Exception as e:
            self.logger.error(f"Error counting papers: {e}")
            return 0
    
    def save(self, filepath=None):
        """Redis auto-persists - nothing to do"""
        self.logger.info("✓ Redis data persisted automatically")

    def load(self, filepath=None):
        """Redis auto-loads - nothing to do"""
        self.logger.info("✓ Redis data available")
    
    def clear_all(self):
        """Clear all papers from Redis (use with caution!)"""
        try:
            keys = self.client.keys("paper:*")
            if keys:
                self.client.delete(*keys)
                self.logger.info(f"✓ Cleared {len(keys)} papers from Redis")
            else:
                self.logger.info("No papers to clear")
        except Exception as e:
            self.logger.error(f"Error clearing papers: {e}")
