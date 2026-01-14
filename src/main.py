import logging
import time
from datetime import datetime
from typing import Dict, List
import schedule

from src.crawler.arxiv_crawler import ArxivCrawler
from src.vectorstore.embedder import PaperEmbedder
from src.vectorstore.vector_store import FAISSVectorStore
from src.retrieval.topic_extractor import TopicExtractor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.cache.semantic_cache import SemanticCache
from src.llm.llm_manager import LLMManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ResearchAssistantSystem:
    """
    Main system orchestrator
    Integrates all components for research topic discovery
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing Research Assistant System...")
        
        self.crawler = ArxivCrawler(poll_interval_seconds=300)
        self.embedder = PaperEmbedder(model_name='allenai/specter')
        self.vector_store = FAISSVectorStore(
            embedding_dim=768,  # SPECTER dimension
            max_size=10000
        )
        self.topic_extractor = TopicExtractor(
            min_cluster_size=5,
            min_samples=3
        )
        self.retrieval_pipeline = RetrievalPipeline(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=10,
            similarity_threshold=0.5
        )
        self.semantic_cache = SemanticCache(
            embedder=self.embedder,
            similarity_threshold=0.92,
            max_cache_size=1000
        )
        self.llm_manager = LLMManager(self.semantic_cache)
        
        self.current_topics = {}
        self.system_stats = {
            'papers_indexed': 0,
            'queries_processed': 0,
            'last_update': None
        }
        
        self.logger.info("System initialized successfully")
    
    def update_paper_database(self):
        """
        Background task: Fetch and index new papers
        Runs every 5 minutes
        """
        self.logger.info("Starting paper database update...")
        
        try:
            # Fetch recent papers
            papers = self.crawler.fetch_recent_papers(
                categories=['cs.AI', 'cs.CL', 'cs.LG', 'cs.CV'],
                days_back=1
            )
            
            if not papers:
                self.logger.info("No new papers found")
                return
            
            # Generate embeddings
            embeddings = self.embedder.embed_papers(papers)
            
            # Add to vector store
            self.vector_store.add_papers(papers, embeddings)
            
            # Update topics
            self.current_topics = self.topic_extractor.extract_topics(
                papers, embeddings
            )
            
            # Update stats
            self.system_stats['papers_indexed'] += len(papers)
            self.system_stats['last_update'] = datetime.now().isoformat()
            
            # Save state
            self.save_state()
            
            self.logger.info(
                f"Updated with {len(papers)} papers. "
                f"Total indexed: {self.system_stats['papers_indexed']}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating paper database: {e}")
    
    def process_query(self, query: str, use_cache: bool = True) -> Dict:
        """
        Process user query and generate response
        
        Args:
            query: User's research question
            use_cache: Whether to use semantic caching
            
        Returns:
            Response dict with answer and metadata
        """
        self.logger.info(f"Processing query: {query[:100]}...")
        
        start_time = time.time()
        
        try:
            # Retrieve relevant papers
            results = self.retrieval_pipeline.retrieve(query)
            
            # Format context for LLM
            context = self.retrieval_pipeline.format_results_for_llm(
                results, max_abstracts=5
            )
            
            # Generate response
            response = self.llm_manager.generate_response(
                query, context, use_cache=use_cache
            )
            
            # Add retrieval metadata
            response['retrieved_papers'] = [
                {
                    'title': paper['title'],
                    'arxiv_id': paper['arxiv_id'],
                    'relevance_score': score
                }
                for paper, score in results[:5]
            ]
            response['processing_time'] = time.time() - start_time
            
            # Update stats
            self.system_stats['queries_processed'] += 1
            
            self.logger.info(
                f"Query processed in {response['processing_time']:.2f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'error': True
            }
    
    def get_trending_topics(self, top_n: int = 5) -> List[Dict]:
        """Get current trending topics"""
        sorted_topics = sorted(
            self.current_topics.values(),
            key=lambda x: x['num_papers'],
            reverse=True
        )
        return sorted_topics[:top_n]
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        cache_stats = self.semantic_cache.get_stats()
        cost_report = self.llm_manager.get_cost_report()
        
        return {
            **self.system_stats,
            'cache_hit_rate': cache_stats['hit_rate'],
            'total_costs_saved': cost_report['total_costs_saved'],
            'vector_store_size': len(self.vector_store.metadata_store),
            'cached_responses': len(self.semantic_cache.cache_queries)
        }
    
    def save_state(self):
        """Save system state to disk"""
        self.vector_store.save('data/vector_store.pkl')
        self.semantic_cache.save('data/semantic_cache.json')
        self.logger.info("System state saved")
    
    def load_state(self):
        """Load system state from disk"""
        try:
            self.vector_store.load('data/vector_store.pkl')
            self.semantic_cache.load('data/semantic_cache.json')
            self.logger.info("System state loaded")
        except FileNotFoundError:
            self.logger.info("No saved state found, starting fresh")
    
    def start_background_tasks(self):
        """Start scheduled background tasks"""
        # Update papers every 5 minutes
        schedule.every(5).minutes.do(self.update_paper_database)
        
        # Reset daily index at midnight
        schedule.every().day.at("00:00").do(self.crawler.reset_daily_index)
        
        self.logger.info("Background tasks scheduled")
        
        # Run initial update
        self.update_paper_database()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)

# Command-line interface
if __name__ == "__main__":
    import sys
    
    system = ResearchAssistantSystem()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "update":
            # Manual update
            system.update_paper_database()
            
        elif sys.argv[1] == "query":
            # Process query
            query = " ".join(sys.argv[2:])
            response = system.process_query(query)
            
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print(f"{'='*80}\n")
            print(response['answer'])
            print(f"\n{'='*80}")
            print(f"Model: {response['model']}")
            print(f"Cost: ${response['cost']:.4f}")
            print(f"Retrieved Papers: {len(response['retrieved_papers'])}")
            print(f"{'='*80}\n")
            
        elif sys.argv[1] == "topics":
            # Show trending topics
            topics = system.get_trending_topics()
            print("\nTrending Research Topics:")
            for i, topic in enumerate(topics, 1):
                print(f"\n{i}. Topic (ID: {topic['cluster_id']})")
                print(f"   Papers: {topic['num_papers']}")
                print(f"   Keywords: {', '.join(topic['keywords'][:5])}")
                
        elif sys.argv[1] == "stats":
            # Show statistics
            stats = system.get_system_stats()
            print("\nSystem Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        elif sys.argv[1] == "daemon":
            # Run as background daemon
            system.start_background_tasks()
    else:
        print("Usage:")
        print("  python main.py update           - Fetch and index new papers")
        print("  python main.py query <text>     - Process research query")
        print("  python main.py topics           - Show trending topics")
        print("  python main.py stats            - Show system statistics")
        print("  python main.py daemon           - Run as background service")
      
