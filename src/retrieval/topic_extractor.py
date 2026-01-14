import hdbscan
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import logging
from collections import Counter

class TopicExtractor:
    """
    Extracts trending topics using HDBSCAN clustering
    Identifies hot topics from recent papers
    """
    
    def __init__(self, min_cluster_size=5, min_samples=3):
        """
        Args:
            min_cluster_size: Minimum papers to form a cluster
            min_samples: Min samples in neighborhood for core point
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)
        
    def extract_topics(self, papers: List[Dict], 
                      embeddings: np.ndarray) -> Dict[int, Dict]:
        """
        Cluster papers and extract topic summaries
        
        Returns:
            Dict mapping cluster_id to topic information
        """
        # Perform clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Extract topics for each cluster
        topics = {}
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise cluster
                continue
                
            # Get papers in this cluster
            cluster_papers = [
                p for p, label in zip(papers, cluster_labels) 
                if label == cluster_id
            ]
            
            # Extract topic summary
            topic_info = self._summarize_cluster(cluster_papers, cluster_id)
            topics[cluster_id] = topic_info
            
        self.logger.info(f"Extracted {len(topics)} topics from {len(papers)} papers")
        return topics
    
    def _summarize_cluster(self, papers: List[Dict], 
                          cluster_id: int) -> Dict:
        """
        Summarize a cluster using TF-IDF keywords
        """
        # Combine abstracts
        texts = [p['abstract'] for p in papers]
        
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=10,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            keywords = vectorizer.get_feature_names_out()
        except:
            keywords = []
        
        # Get most common categories
        all_categories = []
        for p in papers:
            all_categories.extend(p.get('categories', []))
        category_counts = Counter(all_categories)
        
        return {
            'cluster_id': cluster_id,
            'num_papers': len(papers),
            'keywords': list(keywords),
            'top_categories': category_counts.most_common(3),
            'paper_ids': [p['arxiv_id'] for p in papers],
            'sample_titles': [p['title'] for p in papers[:3]]
        }

# Test topic extraction
if __name__ == "__main__":
    from embedder import PaperEmbedder
    import json
    
    # Load papers
    papers = []
    with open('data/papers.jsonl', 'r') as f:
        for line in f:
            papers.append(json.loads(line))
    
    # Generate embeddings
    embedder = PaperEmbedder()
    embeddings = embedder.embed_papers(papers)
    
    # Extract topics
    extractor = TopicExtractor(min_cluster_size=3)
    topics = extractor.extract_topics(papers, embeddings)
    
    print(f"\nDiscovered {len(topics)} topics:")
    for topic_id, info in topics.items():
        print(f"\nTopic {topic_id}:")
        print(f"  Papers: {info['num_papers']}")
        print(f"  Keywords: {', '.join(info['keywords'][:5])}")
        print(f"  Categories: {info['top_categories']}")
