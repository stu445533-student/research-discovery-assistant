# Smart Research Topic Discovery System

An AI-powered research assistant system that helps postgraduate computer science students discover promising research topics using Retrieval-Augmented Generation (RAG), semantic caching, and one-shot prompting.

## 🎯 Features

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

##Installation
```bash
# Clone repository
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

