# Smart Research Topic Discovery System

An AI-powered research assistant system that helps postgraduate computer science students discover promising research topics using Retrieval-Augmented Generation (RAG), semantic caching, and one-shot prompting.

##  Features

- **Real-time arXiv Integration**: Continuously fetches and indexes latest CS papers
- **Semantic Search**: Vector-based retrieval using SPECTER embeddings
- **Topic Extraction**: Automatic clustering and trending topic identification
- **Intelligent Caching**: Semantic caching reduces LLM costs by reusing high-quality responses
- **Cost Optimization**: One-shot prompting enables low-cost models to match flagship quality
- **Conversational Interface**: Natural language interaction for research exploration

## 📋 Requirements

```txt
Python 3.9+
arxiv==2.1.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
openai==1.3.0
anthropic==0.7.0
hdbscan==0.8.33
fastapi==0.104.0
uvicorn==0.24.0
```

# Quick start 

## 1. installation
```bash
# Clone repository
git clone https://github.com/stu445533-student/research-discovery-assistant.git
cd research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## 2. configuration 
Create .env file:

```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```
## 3. Initialize System

```bash
# Create data directory
mkdir data

# Fetch initial papers
python src/main.py update
```
## 4. Run API Server
```bash
# Start FastAPI server
cd src/api
python app.py

```
Visit http://localhost:8000 to access the web interface.

# Usage

## Command Line Interface(CLI)

```bash
# Update paper database
python src/main.py update

# Process a query
python src/main.py query "What are emerging trends in transformer models?"

# Show trending topics
python src/main.py topics

# Show system statistics
python src/main.py stats

# Run as background daemon (continuous updates)
python src/main.py daemon

```
## API Endpoints
POST /query - Process research query
```json
{
  "query": "What are recent advances in NLP?",
  "use_cache": true
}
```
GET /topics?top_n=10 - Get trending topics

GET /stats - Get system statistics

POST /update - Manually trigger paper update

GET /health - Health check

## Python API
```python
from src.main import ResearchAssistantSystem

# Initialize system
system = ResearchAssistantSystem()
system.load_state()

# Update papers
system.update_paper_database()

# Process query
response = system.process_query(
    "What are emerging research areas in computer vision?"
)

print(response['answer'])
print(f"Cost: ${response['cost']:.4f}")
print(f"Retrieved papers: {len(response['retrieved_papers'])}")

# Get trending topics
topics = system.get_trending_topics(top_n=5)
for topic in topics:
    print(f"Topic: {topic['keywords'][:5]}")
    print(f"Papers: {topic['num_papers']}")

```
# Architecture
```
┌─────────────────┐
│  arXiv API      │
│  (5-min poll)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestion │
│  & Embedding    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  FAISS Vector   │────▶│   HDBSCAN    │
│  Store (LRU)    │     │   Clustering │
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │
│   Pipeline      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Semantic Cache │────▶│ LLM Manager  │
│  (Similarity)   │     │ (RAG + Shot) │
└─────────────────┘     └──────────────┘

```

# Testing
```bash
# Run unit tests
python tests/test_system.py

# Run integration tests
python tests/test_integration.py

# Run all tests
python -m pytest tests/

```

# Evaluation Metrics

the system track:

- **Retrieval Quality**: Hit Rate, nDCG@k, Recall@k.

- **Cache Efficiency**: Hit rate, latency reduction.

- **Cost Savings**: LLM API cost reduction percentage.

- **Response Quality**: LLM-as-judge evaluation.


# Configuration
Key parameters in src/main.py:
```python
# Vector store
embedding_dim = 768  # SPECTER dimension
max_store_size = 10000  # LRU limit

# Topic extraction
min_cluster_size = 5
min_samples = 3

# Retrieval
top_k = 10
similarity_threshold = 0.5

# Caching
cache_similarity_threshold = 0.92
max_cache_size = 1000

# Models
flagship_model = 'claude-sonnet-4-20250514'
low_cost_model = 'claude-haiku-4-20250514'

```
