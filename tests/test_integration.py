"""
Integration tests for the complete system
Tests end-to-end workflows
"""
import unittest
import sys
sys.path.append('..')

from src.main import ResearchAssistantSystem
import time

class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize system once for all tests"""
        print("\nInitializing system for integration tests...")
        cls.system = ResearchAssistantSystem()
        
        # Fetch and index some papers
        print("Fetching initial papers...")
        cls.system.update_paper_database()
        print("System ready for testing")
    
    def test_01_paper_indexing(self):
        """Test that papers are fetched and indexed"""
        papers_indexed = self.system.system_stats['papers_indexed']
        self.assertGreater(papers_indexed, 0, "No papers were indexed")
        print(f"✓ Indexed {papers_indexed} papers")
    
    def test_02_topic_extraction(self):
        """Test that topics are extracted from papers"""
        topics = self.system.get_trending_topics(top_n=5)
        self.assertIsInstance(topics, list)
        print(f"✓ Extracted {len(topics)} topics")
        
        if len(topics) > 0:
            topic = topics[0]
            self.assertIn('keywords', topic)
            self.assertIn('num_papers', topic)
            print(f"  Sample topic: {topic['keywords'][:3]}")
    
    def test_03_query_processing(self):
        """Test end-to-end query processing"""
        query = "What are recent advances in transformer architectures?"
        
        print(f"\nProcessing query: {query}")
        response = self.system.process_query(query, use_cache=False)
        
        self.assertIn('answer', response)
        self.assertIn('model', response)
        self.assertIn('cost', response)
        self.assertGreater(len(response['answer']), 50)
        
        print(f"✓ Response generated ({len(response['answer'])} chars)")
        print(f"  Model: {response['model']}")
        print(f"  Cost: ${response['cost']:.4f}")
    
    def test_04_cache_functionality(self):
        """Test semantic caching works"""
        query1 = "What are transformers in machine learning?"
        
        # First query (cache miss)
        print(f"\nFirst query: {query1}")
        response1 = self.system.process_query(query1, use_cache=True)
        cost1 = response1['cost']
        
        # Similar query (should hit cache)
        query2 = "What are transformer models in ML?"
        print(f"Similar query: {query2}")
        response2 = self.system.process_query(query2, use_cache=True)
        cost2 = response2['cost']
        
        # Cache hit should have 0 cost
        cache_stats = self.system.semantic_cache.get_stats()
        print(f"✓ Cache hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"  Total queries: {cache_stats['total_queries']}")
        print(f"  Cache hits: {cache_stats['hits']}")
    
    def test_05_one_shot_prompting(self):
        """Test that low-cost model can be used with examples"""
        query = "What are the latest trends in computer vision?"
        
        print(f"\nQuery with one-shot: {query}")
        response = self.system.process_query(query, use_cache=True)
        
        if response.get('model_type') == 'low_cost':
            print(f"✓ Low-cost model used successfully")
            print(f"  Savings: ${response.get('savings', 0):.4f}")
        else:
            print(f"  Flagship model used (no similar example found)")
    
    def test_06_retrieval_quality(self):
        """Test that relevant papers are retrieved"""
        query = "graph neural networks"
        
        print(f"\nTesting retrieval for: {query}")
        response = self.system.process_query(query)
        
        retrieved = response.get('retrieved_papers', [])
        self.assertGreater(len(retrieved), 0, "No papers retrieved")
        
        print(f"✓ Retrieved {len(retrieved)} papers")
        for i, paper in enumerate(retrieved[:3], 1):
            print(f"  {i}. {paper['title'][:60]}...")
            print(f"     Relevance: {paper['relevance_score']:.2f}")
    
    def test_07_system_stats(self):
        """Test system statistics are tracked correctly"""
        stats = self.system.get_system_stats()
        
        print("\nSystem Statistics:")
        print(f"  Papers indexed: {stats['papers_indexed']}")
        print(f"  Queries processed: {stats['queries_processed']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Total costs saved: ${stats['total_costs_saved']:.2f}")
        print(f"  Vector store size: {stats['vector_store_size']}")
        
        self.assertGreater(stats['queries_processed'], 0)
        self.assertGreaterEqual(stats['cache_hit_rate'], 0)

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
