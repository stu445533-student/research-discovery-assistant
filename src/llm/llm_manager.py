import os
from typing import Dict, Optional
import logging
from anthropic import Anthropic
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM interactions with semantic caching"""
    
    def __init__(self, semantic_cache):
        self.cache = semantic_cache
        
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
            
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        
        self.flagship_model = config.FLAGSHIP_MODEL
        self.flagship_cost = config.FLAGSHIP_COST_PER_1K
        self.low_cost_model = config.LOW_COST_MODEL
        self.low_cost_cost = config.LOW_COST_COST_PER_1K
        self.max_tokens = config.MAX_TOKENS
        
        self.costs_saved = 0
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, query: str, context: str, use_cache: bool = True) -> Dict:
        """Generate response with intelligent model selection"""
        
        # Check cache first
        if use_cache:
            cached_response = self.cache.get(query)
            if cached_response:
                cached_response['cached'] = True
                cached_response['cost'] = 0.0
                return cached_response
        
        # Generate with flagship model
        response = self._generate_flagship(query, context)
        
        # Cache flagship responses
        if use_cache and response.get('model_type') == 'flagship':
            self.cache.set(query, response, model_type='flagship')
        
        return response
    
    def _generate_flagship(self, query: str, context: str) -> Dict:
        """Generate response using flagship model"""
        prompt = self._construct_prompt(query, context)
        
        try:
            response = self.client.messages.create(
                model=self.flagship_model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            answer = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            cost = (tokens / 1000) * self.flagship_cost
            
            self.logger.info(f"Flagship model response: {tokens} tokens, ${cost:.4f}")
            
            return {
                'answer': answer,
                'model': self.flagship_model,
                'model_type': 'flagship',
                'token_count': tokens,
                'cost': cost,
                'cached': False
            }
            
        except Exception as e:
            self.logger.error(f"Flagship model error: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'model': 'error',
                'model_type': 'error',
                'token_count': 0,
                'cost': 0.0,
                'cached': False
            }
    
    def _construct_prompt(self, query: str, context: str) -> str:
        """Construct prompt for flagship model"""
        prompt = f"""You are an expert research assistant helping postgraduate computer science students discover promising research topics.

Based on the following recent papers from arXiv, provide a comprehensive, insightful response to the student's query.

{context}

Student Query: {query}

Instructions:
1. Synthesize information from the retrieved papers
2. Identify trending themes and research gaps
3. Suggest specific, actionable research directions
4. Explain why these topics are timely and impactful
5. Cite paper titles and arXiv IDs when relevant
6. Keep response focused and well-structured (3-4 paragraphs)

Response:"""
        
        return prompt
    
    def get_cost_report(self) -> Dict:
        """Get cost savings report"""
        cache_stats = self.cache.get_stats()
        
        return {
            'total_costs_saved': self.costs_saved,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_hits': cache_stats['hits'],
            'total_queries': cache_stats['total_queries']
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("LLM manager module loaded")
