from typing import List, Dict, Tuple
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    """End-to-end retrieval pipeline"""
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = config.RETRIEVAL_TOP_K
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        self.logger = logging.getLogger(__name__)
        
    def retrieve(self, query: str) -> List[Tuple[Dict, float]]:
        """Retrieve relevant papers for a query"""
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, k=self.top_k * 2)
        
        # Filter by threshold
        results = [(p, s) for p, s in results if s >= self.similarity_threshold]
        
        return results[:self.top_k]
    
    def format_results_for_llm(self, results: List[Tuple[Dict, float]], max_abstracts: int = 5) -> str:
        """Format retrieval results for LLM context"""
        context = "# Retrieved Research Papers\n\n"
        
        for i, (paper, score) in enumerate(results[:max_abstracts]):
            context += f"## Paper {i+1} (Relevance: {score:.2f})\n"
            context += f"**Title:** {paper['title']}\n"
            context += f"**Authors:** {', '.join(paper.get('authors', [])[:3])}\n"
            context += f"**Categories:** {', '.join(paper.get('categories', []))}\n"
            
            abstract = paper.get('abstract', '')
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            context += f"**Abstract:** {abstract}\n"
            context += f"**arXiv ID:** {paper.get('arxiv_id', 'N/A')}\n\n"
        
        return context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Retrieval pipeline module loaded")
