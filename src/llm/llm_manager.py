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
        self.cache.similarity_threshold = threshold
        
        example = self.cache.get(query)
        
        # Restore original threshold
        self.cache.similarity_threshold = original_threshold
        
        return example
    
    def _generate_flagship(self, query: str, context: str) -> Dict:
        """Generate response using flagship model"""
        prompt = self._construct_prompt(query, context)
        
        try:
            if self.flagship_model['provider'] == 'anthropic':
                response = self.anthropic_client.messages.create(
                    model=self.flagship_model['model'],
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                answer = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
                
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model=self.flagship_model['model'],
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=1500
                )
                
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens
            
            cost = (tokens / 1000) * self.flagship_model['cost_per_1k_tokens']
            
            self.logger.info(f"Flagship model response: {tokens} tokens, ${cost:.4f}")
            
            return {
                'answer': answer,
                'model': self.flagship_model['model'],
                'model_type': 'flagship',
                'token_count': tokens,
                'cost': cost
            }
            
        except Exception as e:
            self.logger.error(f"Flagship model error: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'model': 'error',
                'model_type': 'error',
                'token_count': 0,
                'cost': 0
            }
    
    def _generate_with_oneshot(self, query: str, context: str, 
                              example: Dict) -> Dict:
        """Generate response using low-cost model with one-shot example"""
        prompt = self._construct_oneshot_prompt(query, context, example)
        
        try:
            if self.low_cost_model['provider'] == 'anthropic':
                response = self.anthropic_client.messages.create(
                    model=self.low_cost_model['model'],
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                answer = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
                
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model=self.low_cost_model['model'],
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=1500
                )
                
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens
            
            cost = (tokens / 1000) * self.low_cost_model['cost_per_1k_tokens']
            
            # Calculate savings
            flagship_cost = (tokens / 1000) * self.flagship_model['cost_per_1k_tokens']
            savings = flagship_cost - cost
            self.costs_saved += savings
            
            self.logger.info(
                f"Low-cost model response: {tokens} tokens, "
                f"${cost:.4f} (saved ${savings:.4f})"
            )
            
            return {
                'answer': answer,
                'model': self.low_cost_model['model'],
                'model_type': 'low_cost',
                'token_count': tokens,
                'cost': cost,
                'savings': savings,
                'example_used': example.get('cache_metadata', {}).get('cached_query', '')
            }
            
        except Exception as e:
            self.logger.error(f"Low-cost model error, falling back to flagship: {e}")
            return self._generate_flagship(query, context)
    
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
    
    def _construct_oneshot_prompt(self, query: str, context: str, 
                                  example: Dict) -> str:
        """Construct one-shot prompt for low-cost model"""
        
        example_query = example.get('cache_metadata', {}).get('cached_query', '')
        example_answer = example.get('answer', '')
        
        prompt = f"""You are an expert research assistant helping postgraduate computer science students discover promising research topics.

Here is an example of a high-quality response:

EXAMPLE QUERY: {example_query}

EXAMPLE RESPONSE: {example_answer}

---

Now, using the same style, depth, and structure as the example above, respond to this query:

{context}

Student Query: {query}

Provide a response with similar quality, structure, and actionable insights:"""
        
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

# Test LLM manager
if __name__ == "__main__":
    from embedder import PaperEmbedder
    from semantic_cache import SemanticCache
    
    embedder = PaperEmbedder()
    cache = SemanticCache(embedder)
    llm_manager = LLMManager(cache)
    
    # Test query with context
    context = """
    # Retrieved Research Papers
    
    ## Paper 1 (Relevance: 0.92)
    **Title:** Vision Transformers for Image Classification
    **Abstract:** We explore transformer architectures for computer vision tasks...
    """
    
    query = "What are emerging trends in transformer models for vision?"
    
    response = llm_manager.generate_response(query, context)
    print(f"\nQuery: {query}")
    print(f"\nResponse: {response['answer']}")
    print(f"\nModel: {response['model']}")
    print(f"Cost: ${response['cost']:.4f}")
​​​​​​​​​​​​​​​​
