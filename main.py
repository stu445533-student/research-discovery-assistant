import logging
import time
from datetime import datetime
from typing import Dict, List
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config

# Configure logging
os.makedirs(config.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'system.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ResearchAssistantSystem:
    """Main system orchestrator with lazy loading"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Research Assistant System...")
        
        # Create directories
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        # Initialize components as None (lazy loading)
        self._crawler = None
        self._embedder = None
        self._vector_store = None
        self._retrieval_pipeline = None
        self._semantic_cache = None
        self._llm_manager = None
        
        self.system_stats = {
            'papers_indexed': 0,
            'queries_processed': 0,
            'last_update': None
        }
        
        self.logger.info("System initialized (lazy loading enabled)")
    
    @property
    def crawler(self):
        """Lazy load crawler"""
        if self._crawler is None:
            from src.crawler.arxiv_crawler import ArxivCrawler
            self._crawler = ArxivCrawler()
        return self._crawler
    
    @property
    def embedder(self):
        """Lazy load embedder (slowest component)"""
        if self._embedder is None:
            self.logger.info("Loading embedding model (this takes ~10s)...")
            from src.vectorstore.embedder import PaperEmbedder
            self._embedder = PaperEmbedder()
        return self._embedder
    
    @property
    def vector_store(self):
        """Lazy load vector store"""
        if self._vector_store is None:
            from src.vectorstore.vector_store import RedisVectorStore
            self._vector_store = RedisVectorStore()
        return self._vector_store
    
    @property
    def retrieval_pipeline(self):
        """Lazy load retrieval pipeline"""
        if self._retrieval_pipeline is None:
            from src.retrieval.retrieval_pipeline import RetrievalPipeline
            self._retrieval_pipeline = RetrievalPipeline(
                self.vector_store, 
                self.embedder
            )
        return self._retrieval_pipeline
    
    @property
    def semantic_cache(self):
        """Lazy load semantic cache"""
        if self._semantic_cache is None:
            from src.cache.semantic_cache import SemanticCache
            self._semantic_cache = SemanticCache(self.embedder)
            # Auto-load cache if exists
            self._semantic_cache.load()
        return self._semantic_cache
    
    @property
    def llm_manager(self):
        """Lazy load LLM manager"""
        if self._llm_manager is None:
            from src.llm.llm_manager import LLMManager
            self._llm_manager = LLMManager(self.semantic_cache)
        return self._llm_manager
    
    def update_paper_database(self):
        """Background task: Fetch and index new papers"""
        self.logger.info("Starting paper database update...")
        
        try:
            papers = self.crawler.fetch_recent_papers()
            
            if not papers:
                self.logger.info("No new papers found")
                return
            
            self.crawler.save_papers(papers)
            embeddings = self.embedder.embed_papers(papers)
            self.vector_store.add_papers(papers, embeddings)
            
            self.system_stats['papers_indexed'] += len(papers)
            self.system_stats['last_update'] = datetime.now().isoformat()
            
            self.save_state()
            
            self.logger.info(f"Updated with {len(papers)} papers. Total: {self.system_stats['papers_indexed']}")
            
        except Exception as e:
            self.logger.error(f"Error updating paper database: {e}")
    
    def process_query(self, query: str, use_cache: bool = True) -> Dict:
        """Process user query and generate response"""
        self.logger.info(f"Processing query: {query[:100]}...")
        
        start_time = time.time()
        
        try:
            results = self.retrieval_pipeline.retrieve(query)
            
            if not results:
                return {
                    'answer': "No relevant papers found. Please try updating the database or reformulating your query.",
                    'retrieved_papers': [],
                    'processing_time': time.time() - start_time,
                    'model': 'none',
                    'cost': 0.0
                }
            
            context = self.retrieval_pipeline.format_results_for_llm(results, max_abstracts=5)
            response = self.llm_manager.generate_response(query, context, use_cache=use_cache)
            
            response['retrieved_papers'] = [
                {
                    'title': paper['title'],
                    'arxiv_id': paper['arxiv_id'],
                    'relevance_score': score
                }
                for paper, score in results[:5]
            ]
            response['processing_time'] = time.time() - start_time
            
            self.system_stats['queries_processed'] += 1
            
            self.logger.info(f"Query processed in {response['processing_time']:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'error': True,
                'processing_time': time.time() - start_time,
                'cost': 0.0,
                'retrieved_papers': []
            }
    
    def get_trending_topics(self, top_n: int = 5) -> List[Dict]:
        """Get trending topics from indexed papers"""
        from src.retrieval.topic_extractor import TopicExtractor
        
        try:
            # Get papers from vector store
            if hasattr(self.vector_store, 'load_all_papers'):
                papers = self.vector_store.load_all_papers()
            else:
                papers = list(self.vector_store.metadata_store.values())
            
            if len(papers) < 5:
                self.logger.warning(f"Not enough papers: {len(papers)}")
                return []
            
            # Limit to recent 100 for performance
            papers = papers[-100:] if len(papers) > 100 else papers
            
            self.logger.info(f"Extracting topics from {len(papers)} papers...")
            embeddings = self.embedder.embed_papers(papers)
            
            min_cluster = max(3, len(papers) // 20)
            extractor = TopicExtractor(
                min_cluster_size=min_cluster,
                min_samples=2
            )
            
            topics = extractor.extract_topics(papers, embeddings)
            
            if not topics:
                self.logger.warning("No topics found")
                return []
            
            sorted_topics = sorted(
                topics.values(),
                key=lambda x: x['num_papers'],
                reverse=True
            )
            
            self.logger.info(f"Found {len(sorted_topics)} topics")
            return sorted_topics[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}", exc_info=True)
            return []
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        cache_stats = self.semantic_cache.get_stats()
        cost_report = self.llm_manager.get_cost_report()
        
        # Handle both Redis and FAISS
        if hasattr(self.vector_store, 'metadata_store'):
            store_size = len(self.vector_store.metadata_store)
        else:
            store_size = 0  # Redis doesn't expose this easily
        
        return {
            **self.system_stats,
            'cache_hit_rate': cache_stats['hit_rate'],
            'total_costs_saved': cost_report['total_costs_saved'],
            'vector_store_size': store_size,
            'cached_responses': len(self.semantic_cache.cache_queries)
        }
    
    def save_state(self):
        """Save system state to disk"""
        try:
            if self._vector_store:
                self.vector_store.save()
            if self._semantic_cache:
                self.semantic_cache.save()
            self.logger.info("System state saved")
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load system state from disk"""
        try:
            if self._vector_store:
                self.vector_store.load()
            if self._semantic_cache:
                self.semantic_cache.load()
            self.logger.info("System state loaded")
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")


def main():
    """Command-line interface"""
    
    # Measure initialization time
    init_start = time.time()
    system = ResearchAssistantSystem()
    init_time = time.time() - init_start
    
    print(f"\n✓ System ready in {init_time:.2f}s")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "update":
            print("\n" + "="*80)
            print("UPDATING PAPER DATABASE")
            print("="*80 + "\n")
            system.update_paper_database()
            print("\n✓ Update complete\n")
            
        elif command == "query":
            if len(sys.argv) < 3:
                print("Usage: python src/main.py query \"your question\"")
                return
                
            query = " ".join(sys.argv[2:])
            
            print("\n" + "="*80)
            print("PROCESSING QUERY")
            print("="*80)
            print(f"Query: {query}\n")
            
            response = system.process_query(query)
            
            print("\n" + "="*80)
            print("RESPONSE")
            print("="*80 + "\n")
            print(response['answer'])
            print("\n" + "="*80)
            print("METADATA")
            print("="*80)
            print(f"Model: {response.get('model', 'N/A')}")
            print(f"Model Type: {response.get('model_type', 'N/A')}")
            print(f"Cost: ${response.get('cost', 0):.4f}")
            
            if 'savings' in response:
                print(f"Savings: ${response['savings']:.4f}")
            
            print(f"Processing Time: {response.get('processing_time', 0):.2f}s")
            print(f"Retrieved Papers: {len(response.get('retrieved_papers', []))}")
            
            if response.get('retrieved_papers'):
                print("\n" + "="*80)
                print("TOP RETRIEVED PAPERS")
                print("="*80)
                for i, paper in enumerate(response['retrieved_papers'][:3], 1):
                    print(f"\n{i}. {paper['title']}")
                    print(f"   arXiv: {paper['arxiv_id']}")
                    print(f"   Relevance: {paper['relevance_score']:.3f}")
            
            print("\n" + "="*80 + "\n")
            
        elif command == "topics":
            topics = system.get_trending_topics(top_n=5)
            print("\n" + "="*80)
            print("TRENDING TOPICS")
            print("="*80)
            
            if not topics:
                print("No topics found. Update database first.")
            else:
                for i, topic in enumerate(topics, 1):
                    print(f"\n{i}. Topic ID: {topic['cluster_id']}")
                    print(f"   Papers: {topic['num_papers']}")
                    print(f"   Keywords: {', '.join(topic['keywords'][:5])}")
                    categories = [f"{cat}({count})" for cat, count in topic['top_categories'][:3]]
                    print(f"   Categories: {', '.join(categories)}")
            
            print("\n" + "="*80 + "\n")
            
        elif command == "stats":
            stats = system.get_system_stats()
            print("\n" + "="*80)
            print("SYSTEM STATISTICS")
            print("="*80)
            print(f"Papers Indexed: {stats['papers_indexed']}")
            print(f"Queries Processed: {stats['queries_processed']}")
            print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
            print(f"Cached Responses: {stats['cached_responses']}")
            print(f"Vector Store Size: {stats['vector_store_size']}")
            print(f"Total Costs Saved: ${stats['total_costs_saved']:.2f}")
            print(f"Last Update: {stats['last_update']}")
            print("="*80 + "\n")
            
        elif command == "daemon":
            print("\n" + "="*80)
            print("STARTING AUTO-UPDATE DAEMON")
            print("="*80)
            print(f"Update interval: {config.ARXIV_POLL_INTERVAL} seconds")
            print("Press Ctrl+C to stop")
            print("="*80 + "\n")
            
            update_count = 0
            
            try:
                while True:
                    update_count += 1
                    print(f"\n[UPDATE #{update_count}] {datetime.now()}")
                    
                    try:
                        system.update_paper_database()
                        print(f"✓ Update #{update_count} complete")
                    except Exception as e:
                        print(f"✗ Update #{update_count} failed: {e}")
                    
                    print(f"Sleeping for {config.ARXIV_POLL_INTERVAL} seconds...")
                    time.sleep(config.ARXIV_POLL_INTERVAL)
                    
            except KeyboardInterrupt:
                print(f"\n\nDaemon stopped. Total updates: {update_count}")
                system.save_state()
                
        else:
            print("\nUnknown command. Available commands:")
            print("  update - Fetch and index new papers")
            print("  query  - Process research query")
            print("  topics - Show trending topics")
            print("  stats  - Show system statistics")
            print("  daemon - Run auto-update service\n")
    else:
        print("\n" + "="*80)
        print("RESEARCH ASSISTANT SYSTEM")
        print("="*80)
        print("\nUsage:")
        print("  python src/main.py update")
        print("      Fetch and index new papers from arXiv")
        print()
        print("  python src/main.py query \"your question\"")
        print("      Process a research query")
        print()
        print("  python src/main.py topics")
        print("      Show trending topics")
        print()
        print("  python src/main.py stats")
        print("      Show system statistics")
        print()
        print("  python src/main.py daemon")
        print("      Run auto-update service (every 5 minutes)")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
