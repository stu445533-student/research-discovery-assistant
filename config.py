import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# arXiv Configuration
ARXIV_POLL_INTERVAL = 300  # 5 minutes
ARXIV_MAX_RESULTS = 50  # Reduced for 8GB RAM
ARXIV_CATEGORIES = ['cs.AI', 'cs.CL', 'cs.LG', 'cs.CV']
ARXIV_LOOKBACK_DAYS = 1

# Embedding Configuration
EMBEDDING_MODEL = 'allenai/specter'
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 8  # Reduced for 8GB RAM

# Vector Store Configuration
VECTOR_STORE_MAX_SIZE = 3000  # Reduced for 8GB RAM
VECTOR_STORE_PATH = 'data/vector_store.pkl'

# Clustering Configuration
MIN_CLUSTER_SIZE = 3  # Reduced for smaller dataset
MIN_SAMPLES = 2

# Retrieval Configuration
RETRIEVAL_TOP_K = 10
SIMILARITY_THRESHOLD = 0.5

# Cache Configuration
CACHE_SIMILARITY_THRESHOLD = 0.92
CACHE_MAX_SIZE = 500  # Reduced for 8GB RAM
CACHE_PATH = 'data/semantic_cache.json'

# LLM Configuration
FLAGSHIP_MODEL = 'claude-sonnet-4-20250514'
FLAGSHIP_COST_PER_1K = 0.015
LOW_COST_MODEL = 'claude-haiku-4-20250514'
LOW_COST_COST_PER_1K = 0.001
MAX_TOKENS = 1500

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 8000

# Paths
DATA_DIR = 'data'
LOGS_DIR = 'logs'
PAPERS_FILE = 'data/papers.jsonl'
METRICS_LOG = 'data/metrics.jsonl'
