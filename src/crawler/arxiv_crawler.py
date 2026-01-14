import arxiv
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import json

class ArxivCrawler:
    """
    Crawls arXiv API for computer science papers
    Implements 5-minute polling with rate limiting
    """
    
    def __init__(self, max_results_per_query=100, 
                 poll_interval_seconds=300):
        self.max_results = max_results_per_query
        self.poll_interval = poll_interval_seconds
        self.last_query_time = None
        self.daily_index = 0
        self.logger = logging.getLogger(__name__)
        
    def fetch_recent_papers(self, categories=['cs.AI', 'cs.CL', 'cs.LG'],
                           days_back=1) -> List[Dict]:
        """
        Fetch papers from specified CS categories
        
        Args:
            categories: List of arXiv category codes
            days_back: How many days to look back
            
        Returns:
            List of paper dictionaries with metadata
        """
        papers = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Construct query
        category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
        date_query = f'submittedDate:[{start_date.strftime("%Y%m%d")} TO {end_date.strftime("%Y%m%d")}]'
        
        search_query = f'({category_query}) AND {date_query}'
        
        try:
            # Query arXiv API
            search = arxiv.Search(
                query=search_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for result in search.results():
                paper = {
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': [author.name for author in result.authors],
                    'categories': result.categories,
                    'published_date': result.published.isoformat(),
                    'updated_date': result.updated.isoformat(),
                    'pdf_url': result.pdf_url,
                    'batch_index': self.daily_index
                }
                papers.append(paper)
                self.daily_index += 1
                
            self.logger.info(f"Fetched {len(papers)} papers")
            
        except Exception as e:
            self.logger.error(f"Error fetching papers: {e}")
            
        return papers
    
    def save_papers(self, papers: List[Dict], filepath='data/papers.jsonl'):
        """Save papers to JSONL file"""
        with open(filepath, 'a') as f:
            for paper in papers:
                f.write(json.dumps(paper) + '\n')
                
    def reset_daily_index(self):
        """Reset index at midnight"""
        self.daily_index = 0
        self.logger.info("Daily index reset")

# Test the crawler
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crawler = ArxivCrawler()
    papers = crawler.fetch_recent_papers(days_back=1)
    print(f"Retrieved {len(papers)} papers")
    crawler.save_papers(papers)
