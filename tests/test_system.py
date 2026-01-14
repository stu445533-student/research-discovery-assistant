import unittest
import sys
import os
sys.path.append('..')

from src.crawler.arxiv_crawler import ArxivCrawler
from src.vectorstore.embedder import PaperEmbedder
from src.vectorstore.vector_store import FAISSVectorStore
from src.retrieval.topic_extractor import TopicExtractor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.cache.semantic_cache import SemanticCache

class TestArxivCrawler(unittest.TestCase):
    """Test arXiv crawler functionality"""
    
    def setUp(self):
        self.crawler = ArxivCrawler(max_results_per_query=10)
    
    def test_fetch_papers(self):
        """Test fetching papers from arXiv"""
        papers = self.crawler.fetch_recent_papers(
            categories=['cs.AI'],
            days_back=1
        )
        
        self.assertIsInstance(papers, list)
        if len(papers) > 0:
            paper = papers[0]
            self.assertIn('arxiv_id', paper)
            self.assertIn('title', paper)
            self.assertIn('abstract', paper)
            self.assertIn('authors', paper)
    
    def test_daily_index_increment(self):
        """Test batch index increments correctly"""
        initial_index = self.crawler.daily_index
        self.crawler.fetch_recent_papers(days_back=1)
        self.assertGreaterEqual(self.crawler.daily_index, initial_index)

class TestEmbedder(unittest.TestCase):
    """Test embedding generation"""
    
    def setUp(self):
        self.embedder = PaperEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.sample_papers = [
            {
                'title': 'Attention Is All You Need',
                'abstract': 'We propose a new simple network architecture, the Transformer...'
            },
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'abstract': 'We introduce BERT, a new language representation model...'
            }
        ]
    
    def test_embed_papers(self):
        """Test generating embeddings for papers"""
        embeddings = self.embedder.embed_papers(self.sample_papers)
        
        self.assertEqual(embeddings.shape[0], len(self.sample_papers))
        self.assertEqual(embeddings.shape[1], 384)  # MiniLM dimension
    
    def test_embed_query(self):
        """Test generating query embedding"""
        query = "transformer models for NLP"
        embedding = self.embedder.embed_query(query)
        
        self.assertEqual(embedding.shape[0], 384)

class TestVectorStore(unittest.TestCase):
    """Test FAISS vector store"""
    
    def setUp(self):
        self.embedder = PaperEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store = FAISSVectorStore(embedding_dim=384, max_size=100)
        
        self.sample_papers = [
            {
                'arxiv_id': '1706.03762',
                'title': 'Attention Is All You Need',
                'abstract': 'Transformer architecture for NLP tasks'
            },
            {
                'arxiv_id': '1810.04805',
                'title': 'BERT',
                'abstract': 'Bidirectional transformer pre-training'
            }
        ]
        
        embeddings = self.embedder.embed_papers(self.sample_papers)
        self.vector_store.add_papers(self.sample_papers, embeddings)
    
    def test_add_and_search(self):
        """Test adding papers and searching"""
        query = "transformer models"
        query_embedding = self.embedder.embed_query(query)
        
        results = self.vector_store.search(query_embedding, k=2)
        
        self.assertGreater(len(results), 0)
        self.assertEqual(len(results[0]), 2)  # (paper, score)
    
    def test_lru_eviction(self):
        """Test LRU eviction when max size reached"""
        small_store = FAISSVectorStore(embedding_dim=384, max_size=5)
        
        # Add 10 papers to store with max_size=5
        for i in range(10):
            paper = {'arxiv_id': f'test_{i}', 'title': f'Paper {i}', 'abstract': f'Abstract {i}'}
            embedding = self.embedder.embed_papers([paper])
            small_store.add_papers([paper], embedding)
        
        # Should only have 5 papers
        self.assertEqual(len(small_store.metadata_store), 5)

class TestTopicExtractor(unittest.TestCase):
    """Test topic extraction with clustering"""
    
    def setUp(self):
        self.embedder = PaperEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.extractor = TopicExtractor(min_cluster_size=2, min_samples=1)
        
        # Create papers with distinct topics
        self.papers = [
            {'title': 'Transformer for NLP', 'abstract': 'Natural language processing with transformers', 'categories': ['cs.CL']},
            {'title': 'BERT for NLP', 'abstract': 'Language models for NLP tasks', 'categories': ['cs.CL']},
            {'title': 'CNN for Vision', 'abstract': 'Computer vision with convolutional networks', 'categories': ['cs.CV']},
            {'title': 'ResNet for Vision', 'abstract': 'Image classification with residual networks', 'categories': ['cs.CV']},
        ]
    
    def test_extract_topics(self):
        """Test topic extraction from papers"""
        embeddings = self.embedder.embed_papers(self.papers)
        topics = self.extractor.extract_topics(self.papers, embeddings)
        
        self.assertIsInstance(topics, dict)
        # Should find at least 1 cluster
        self.assertGreaterEqual(len(topics), 1)
        
        for topic_id, info in topics.items():
            self.assertIn('num_papers', info)
            self.assertIn('keywords', info)
            self.assertIn('paper_ids', info)

class TestSemanticCache(unittest.TestCase):
    """Test semantic caching system"""
    
    def setUp(self):
        self.embedder = PaperEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.cache = SemanticCache(
            embedder=self.embedder,
            similarity_threshold=0.9,
            max_cache_size=10
        )
    
    def test_cache_miss(self):
        """Test cache miss on first query"""
        result = self.cache.get("What are transformers?")
        self.assertIsNone(result)
    
    def test_cache_hit(self):
        """Test cache hit on similar query"""
        # Store a response
        query1 = "What are transformer models?"
        response = {
            'answer': 'Transformers are neural network architectures...',
            'model': 'claude-sonnet-4',
            'token_count': 100
        }
        self.cache.set(query1, response, model_type='flagship')
        
        # Query with similar text
        query2 = "What are transformer architectures?"
        cached = self.cache.get(query2)
        
        self.assertIsNotNone(cached)
        self.assertEqual(cached['answer'], response['answer'])
    
    def test_cache_stats(self):
        """Test cache statistics tracking"""
        self.cache.get("query 1")
        self.cache.get("query 2")
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['total_queries'], 2)
        self.assertEqual(stats['misses'], 2)
        self.assertEqual(stats['hit_rate'], 0.0)

class TestRetrievalPipeline(unittest.TestCase):
    """Test end-to-end retrieval pipeline"""
    
    def setUp(self):
        self.embedder = PaperEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store = FAISSVectorStore(embedding_dim=384)
        
        # Add sample papers
        papers = [
            {
                'arxiv_id': '1706.03762',
                'title': 'Attention Is All You Need',
                'abstract': 'Transformer architecture for sequence modeling',
                'authors': ['Vaswani et al.'],
                'categories': ['cs.CL']
            }
        ]
        embeddings = self.embedder.embed_papers(papers)
        self.vector_store.add_papers(papers, embeddings)
        
        self.pipeline = RetrievalPipeline(
            self.vector_store,
            self.embedder,
            top_k=5,
            similarity_threshold=0.3
        )
    
    def test_retrieve(self):
        """Test retrieval with query"""
        results = self.pipeline.retrieve("transformer models")
        
        self.assertIsInstance(results, list)
        if len(results) > 0:
            paper, score = results[0]
            self.assertIn('title', paper)
            self.assertIsInstance(score, float)
    
    def test_format_for_llm(self):
        """Test formatting results for LLM context"""
        results = self.pipeline.retrieve("transformer models")
        context = self.pipeline.format_results_for_llm(results)
        
        self.assertIsInstance(context, str)
        self.assertIn('Retrieved Research Papers', context)

def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestArxivCrawler))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbedder))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorStore))
    suite.addTests(loader.loadTestsFromTestCase(TestTopicExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticCache))
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalPipeline))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == '__main__':
    result = run_all_tests()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
