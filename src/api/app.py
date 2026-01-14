from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
sys.path.append('..')

from src.main import ResearchAssistantSystem

app = FastAPI(title="Research Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
system = ResearchAssistantSystem()
system.load_state()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    use_cache: Optional[bool] = True
    
class QueryResponse(BaseModel):
    answer: str
    model: str
    model_type: str
    cost: float
    retrieved_papers: List[dict]
    processing_time: float
    cached: Optional[bool] = False

class TopicInfo(BaseModel):
    cluster_id: int
    num_papers: int
    keywords: List[str]
    top_categories: List[tuple]
    sample_titles: List[str]

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Research Assistant API",
        "version": "1.0.0",
        "endpoints": ["/query", "/topics", "/stats", "/update"]
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a research query"""
    try:
        response = system.process_query(
            request.query, 
            use_cache=request.use_cache
        )
        
        if response.get('error'):
            raise HTTPException(status_code=500, detail=response['answer'])
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics", response_model=List[TopicInfo])
async def get_topics(top_n: int = 10):
    """Get trending research topics"""
    try:
        topics = system.get_trending_topics(top_n=top_n)
        return topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = system.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def trigger_update():
    """Manually trigger paper database update"""
    try:
        system.update_paper_database()
        return {"message": "Update completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "papers_indexed": system.system_stats['papers_indexed'],
        "last_update": system.system_stats['last_update']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
