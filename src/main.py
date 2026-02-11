import logging
import time
from datetime import datetime
from typing import Dict, List
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from src.crawler.arxiv_crawler import ArxivCrawler
from src.vectorstore.embedder import PaperEmbedder
from src.vectorstore.vector_store import RedisVectorStore 
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.cache.semantic_cache import SemanticCache
from src.llm.llm_manager import LLMManager

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
    """Main system orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Research Assistant System...")
        
        # Create directories
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        # Initialize components
        try:
            self.crawler = ArxivCrawler()
            self.embedder = PaperEmbedder()
            self.vector_store = RedisVectorStore()
            self.retrieval_pipeline = RetrievalPipeline(self.vector_store, self.embedder)
            self.semantic_cache = SemanticCache(self.embedder)
            self.llm_manager = LLMManager(self.semantic_cache)
            
            self.system_stats = {
                'papers_indexed': 0,
                'queries_processed': 0,
                'last_update': None
            }
            
            self.logger.info("Attempting to load saved state...")
            self.load_state() 
            # ----------------------
            self.logger.info("System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise
    
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
                    'cost': 0.0,
                    'cached': False
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
            if use_cache:
                self.save_state()
            
            self.logger.info(f"Query processed in {response['processing_time']:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'error': True,
                'processing_time': time.time() - start_time,
                'cost': 0.0,
                'cached': False,
                'retrieved_papers': []
            }
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        cache_stats = self.semantic_cache.get_stats()
        cost_report = self.llm_manager.get_cost_report()
        
        # Check if vector store has a count method (Redis) or use len (FAISS fallback)
        if hasattr(self.vector_store, 'count'):
            vs_size = self.vector_store.count()
        elif hasattr(self.vector_store, 'metadata_store'):
            vs_size = len(self.vector_store.metadata_store)
        else:
            vs_size = 0

        return {
            **self.system_stats,
            'cache_hit_rate': cache_stats['hit_rate'],
            'total_costs_saved': cost_report['total_costs_saved'],
            'vector_store_size': vs_size,  # <--- FIXED
            'cached_responses': len(self.semantic_cache.cache_queries)
        }
    
    def save_state(self):
        """Save system state to disk"""
        try:
            self.vector_store.save()
            self.semantic_cache.save()
            self.logger.info("System state saved")
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load system state from disk"""
        try:
            self.vector_store.load()
            self.semantic_cache.load()
            self.logger.info("System state loaded")
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")

    


    def get_trending_topics(self, top_n: int = 5) -> List[Dict]:
        """Get trending topics from indexed papers"""
        from src.retrieval.topic_extractor import TopicExtractor

        try:
            # Get papers differently based on vector store type
            if isinstance(self.vector_store, RedisVectorStore):
                # For Redis: Read from papers file
                import json
                papers = []

                if not os.path.exists(config.PAPERS_FILE):
                    self.logger.warning("No papers file found")
                    return []

                with open(config.PAPERS_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            papers.append(json.loads(line))

                # Take only recent papers (last 100)
                papers = papers[-100:] if len(papers) > 100 else papers

            else:
                # For FAISS: Use metadata_store
                papers = list(self.vector_store.metadata_store.values())

            if len(papers) < 5:
                self.logger.warning(f"Not enough papers for topic extraction: {len(papers)}")
                return []

            # Get embeddings
            self.logger.info(f"Extracting topics from {len(papers)} papers...")
            embeddings = self.embedder.embed_papers(papers)

            # Extract topics
            min_cluster = max(3, len(papers) // 20)
            extractor = TopicExtractor(
            min_cluster_size=min_cluster,
            min_samples=2
            )

            topics = extractor.extract_topics(papers, embeddings)

            if not topics:
                self.logger.warning("No topics extracted")
                return []

            # Sort by number of papers
            sorted_topics = sorted(
            topics.values(),
            key=lambda x: x['num_papers'],
            reverse=True
            )

            self.logger.info(f"Extracted {len(sorted_topics)} topics")
            return sorted_topics[:top_n]

        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}", exc_info=True)
            return []

def main():
    """Command-line interface"""
    import sys
    
    system = ResearchAssistantSystem()
    
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
            print(f"Cost: ${response.get('cost', 0):.4f}")
            print(f"Cached: {response.get('cached', False)}")
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
                
        elif command == "load":
            system.load_state()
            print("✓ State loaded successfully\n")
        
        elif command == "topics":
            topics = system.get_trending_topics(top_n=5)
            print("\n" + "="*80)
            print("TRENDING TOPICS")
            print("="*80)
            
            if not topics:
                print ("no topics found. update database first.")
            
            else:
                for i, topic in enumerate(topics, 1):
                    print(f"\n{i}. Topic ID: {topic['cluster_id']}")
                    print(f"    Papers: {topic['num_papers']}")
                    print(f"    Keywords: {', '.join(topic['keywords'][:5])}")
                    print(f"    Categories: {', '.join([f'{cat}({count})' for cat, count in topic['top_categories']][:3])}")
            print("\n" + "="*80 + "\n")

            
        else:
            print("\nUnknown command. Available commands:")
            print("  update - Fetch and index new papers")
            print("  query  - Process research query")
            print("  stats  - Show system statistics")
            print("  load   - Load saved state\n")
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
        print("  python src/main.py stats")
        print("      Show system statistics")
        print()
        print("  python src/main.py load")
        print("      Load saved state from disk")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
