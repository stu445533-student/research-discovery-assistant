import os
from typing import Dict, Optional, List
import logging
from anthropic import Anthropic
from openai import OpenAI
from src.cache.semantic_cache import SemanticCache

class LLMManager:
    """
    Manages LLM interactions with semantic caching and one-shot prompting
    Implements cost-efficient model selection
    """
    
    def __init__(self, semantic_cache: SemanticCache):
        self.cache = semantic_cache
        
        # Initialize clients
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Model configurations
        self.flagship_model = {
            'provider': 'anthropic',
            'model': 'claude-sonnet-4-20250514',
            'cost_per_1k_tokens': 0.015  # Example cost
        }
        
        self.low_cost_model = {
            'provider': 'anthropic',
            'model': 'claude-haiku-4-20250514',
            'cost_per_1k_tokens': 0.001  # Example cost
        }
        
        self.costs_saved = 0
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, query: str, 
                         context: str,
                         use_cache: bool = True) -> Dict:
        """
        Generate response with intelligent model selection
        
        Args:
            query: User query
            context: Retrieved paper context
            use_cache: Whether to use semantic cache
            
        Returns:
            Response dict with answer and metadata
        """
        # Check cache first
        if use_cache:
            cached_response = self.cache.get(query)
            if cached_response:
                return cached_response
        
        # Determine which model to use
        cached_example = self._get_similar_example(query)
        
        if cached_example:
            # Use low-cost model with one-shot prompting
            response = self._generate_with_oneshot(
                query, context, cached_example
            )
        else:
            # Use flagship model
            response = self._generate_flagship(query, context)
            
            # Cache flagship responses
            self.cache.set(query, response, model_type='flagship')
        
        return response
    
    def _get_similar_example(self, query: str, 
                           threshold: float = 0.85) -> Optional[Dict]:
        """
        Find a similar cached query for one-shot prompting
        Uses slightly lower threshold than cache retrieval
        """
        # Temporarily lower cache threshold
        original_threshold = self.cache.similarity_threshold
        self.cache.similarity_threshol​​​​​​​​​​​​​​​​
