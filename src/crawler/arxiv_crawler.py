import arxiv
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class ArxivCrawler:
    """Crawls arXiv API for computer science papers"""
    
    def __init__(self):
        self.max_results = config.ARXIV_MAX_RESULTS
        self.poll_interval = config.ARXIV_POLL_INTERVAL
        self.daily_index = 0
        self.logger = logging.getLogger(__name__)
        
    def fetch_recent_papers(self, categories=None, days_back=None) -> List[Dict]:
        """Fetch papers from specified CS categories"""
        if categories is None:
            categories = config.ARXIV_CATEGORIES
        if days_back is None:
            days_back = config.ARXIV_LOOKBACK_DAYS
            
        papers = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
        
        try:
            search = arxiv.Search(
                query=category_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for result in search.results():
                paper = {
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'abstract': result.summary.replace('\n', ' '),
                    'authors': [author.name for author in result.authors],
                    'categories': result.categories,
                    'published_date': result.published.isoformat(),
                    'updated_date': result.updated.isoformat(),
                    'pdf_url': result.pdf_url,
                    'batch_index': self.daily_index
                }
                papers.append(paper)
                self.daily_index += 1
                
            self.logger.info(f"Fetched {len(papers)} papers from arXiv")
            
        except Exception as e:
            self.logger.error(f"Error fetching papers: {e}")
            
        return papers
    
    def save_papers(self, papers: List[Dict], filepath=None):
        """Save papers to JSONL file"""
        if filepath is None:
            filepath = config.PAPERS_FILE
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
                
        self.logger.info(f"Saved {len(papers)} papers to {filepath}")
                
    def reset_daily_index(self):
        """Reset index at midnight"""
        self.daily_index = 0
        self.logger.info("Daily index reset")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crawler = ArxivCrawler()
    papers = crawler.fetch_recent_papers()
    print(f"Retrieved {len(papers)} papers")
    if papers:
        crawler.save_papers(papers)
        print(f"Sample: {papers[0]['title']}")

