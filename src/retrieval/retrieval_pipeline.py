from typing import List, Dict, Tuple
import logging
from src.vectorstore.embedder import PaperEmbedder
from src.vectorstore.vector_store import FAISSVectorStore

class RetrievalPipeline:
    """
    End-to-end retrieval pipeline
    Handles query processing and result ranking
    """
    
    def __init__(self, vector_store: FAISSVectorStore, 
                 embedder: PaperEmbedder,
                 top_k: int = 10,
                 similarity_threshold: float = 0.5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        
    def retrieve(self, query: str, 
                filters: Dict = None) -> List[Tuple[Dict, float]]:
        """
        Retrieve relevant papers for a query
        
        Args:
            query: User's research query
            filters: Optional filters (categories, date range)
            
        Returns:
            List of (paper, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding, 
            k=self.top_k * 2  # Retrieve more for filtering
        )
        
        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)
        
        # Remove duplicates and low-quality matches
        results = self._deduplicate_results(results)
        results = [(p, s) for p, s in results if s >= self.similarity_threshold]
        
        # Return top-k
        return results[:self.top_k]
    
    def _apply_filters(self, results: List[Tuple[Dict, float]], 
                      filters: Dict) -> List[Tuple[Dict, float]]:
        """Apply category or date filters"""
        filtered = []
        
        for paper, score in results:
            # Category filter
            if 'categories' in filters:
                if not any(cat in paper['categories'] 
                          for cat in filters['categories']):
                    continue
            
            # Date filter (if needed)
            # Add date filtering logic here
            
            filtered.append((paper, score))
        
        return filtered
    
    def _deduplicate_results(self, results: List[Tuple[Dict, float]], 
                            title_similarity_threshold: float = 0.9
                            ) -> List[Tuple[Dict, float]]:
        """Remove near-duplicate papers based on title similarity"""
        from difflib import SequenceMatcher
        
        unique_results = []
        seen_titles = []
        
        for paper, score in results:
            title = paper['title'].lower()
            
            # Check similarity with seen titles
            is_duplicate = False
            for seen_title in seen_titles:
                similarity = SequenceMatcher(None, title, seen_title).ratio()
                if similarity > title_similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append((paper, score))
                seen_titles.append(title)
        
        return unique_results
    
    def format_results_for_llm(self, results: List[Tuple[Dict, float]], 
                               max_abstracts: int = 5) -> str:
        """
        Format retrieval results for LLM context
        
        Returns:
            Formatted string for prompt augmentation
        """
        context = "# Retrieved Research Papers\n\n"
        
        for i, (paper, score) in enumerate(results[:max_abstracts]):
            context += f"## Paper {i+1} (Relevance: {score:.2f})\n"
            context += f"**Title:** {paper['title']}\n"
            context += f"**Authors:** {', '.join(paper['authors'][:3])}\n"
            context += f"**Categories:** {', '.join(paper['categories'])}\n"
            context += f"**Abstract:** {paper['abstract'][:500]}...\n"
            context += f"**arXiv ID:** {paper['arxiv_id']}\n\n"
        
        return context

# Integration test
if __name__ == "__main__":
    import json
    from embedder import PaperEmbedder
    from vector_store import FAISSVectorStore
    
    # Load existing vector store
    embedder = PaperEmbedder()
    vector_store = FAISSVectorStore(embedding_dim=768)
    vector_store.load('data/vector_store.pkl')
    
    # Initialize pipeline
    pipeline = RetrievalPipeline(vector_store, embedder)
    
    # Test query
    query = "transformer architectures for computer vision"
    results = pipeline.retrieve(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results\n")
    
    # Format for LLM
    context = pipeline.format_results_for_llm(results)
    print(context)

