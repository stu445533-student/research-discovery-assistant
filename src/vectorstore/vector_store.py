import redis
import numpy as np
import logging
import json
from typing import List, Dict, Tuple
import os
import sys

# --- SAFE IMPORTS: Handle library version naming differences ---
try:
    # TRY NEWER PATH (Redis 7.x+) - snake_case
    from redis.commands.search.field import VectorField, TextField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError:
    try:
        # TRY OLDER PATH - CamelCase
        from redis.commands.search.field import VectorField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        from redis.commands.search.query import Query
    except ImportError:
        # If both fail, print the debug info
        print("CRITICAL ERROR: Could not import Redis Search modules.")
        print(f"Current Redis library version: {redis.__version__}")
        print("Try running: pip install --upgrade redis")
        sys.exit(1)
# ---------------------------------------------

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class RedisVectorStore:
    """Redis-based vector store implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.index_name = config.REDIS_INDEX_NAME
        self.dim = config.EMBEDDING_DIM
        
        try:
            # Connect to Redis
            self.client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                decode_responses=False 
            )
            self.client.ping()
            self.logger.info("Connected to Redis successfully")
            
            # Check/Create Index
            self._create_index_if_not_exists()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _create_index_if_not_exists(self):
        """Create the Search Index schema"""
        try:
            # Check if index exists
            self.client.ft(self.index_name).info()
            self.logger.info(f"Index '{self.index_name}' already exists")
        except redis.exceptions.ResponseError:
            # Index does not exist (Expected error), so we create it
            self.logger.info(f"Index '{self.index_name}' not found. Creating it...")
            
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
            
            definition = IndexDefinition(prefix=["paper:"], index_type=IndexType.HASH)
            
            self.client.ft(self.index_name).create_index(
                fields=schema, 
                definition=definition
            )
            self.logger.info(f"Created new index '{self.index_name}'")

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
        self.logger.info(f"Added {len(papers)} papers to Redis")

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
                'title': doc.title,
                'abstract': doc.abstract,
                'arxiv_id': doc.arxiv_id,
                'authors': json.loads(doc.authors) if hasattr(doc, 'authors') else [],
                'categories': json.loads(doc.categories) if hasattr(doc, 'categories') else []
            }
            formatted_results.append((paper, similarity))
            
        return formatted_results
        
    def count(self) -> int:
        """Return the number of papers in the Redis index"""
        try:
            info = self.client.ft(self.index_name).info()
            return int(info.get('num_docs', 0))
        except Exception:
            return 0
    
    
    def save(self, filepath=None):
        pass # Redis auto-saves

    def load(self, filepath=None):
        pass # Redis auto-loads
