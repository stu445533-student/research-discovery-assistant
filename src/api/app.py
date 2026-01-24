from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.main import ResearchAssistantSystem

app = FastAPI(title="Research Assistant API")

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

class QueryRequest(BaseModel):
    query: str
    use_cache: Optional[bool] = True

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve simple HTML interface"""
    html_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return """
    <html>
        <body>
            <h1>Research Assistant API</h1>
            <p>API is running. Use POST /query to ask questions.</p>
        </body>
    </html>
    """

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a research query"""
    try:
        response = system.process_query(request.query, use_cache=request.use_cache)
        if response.get('error'):
            raise HTTPException(status_code=500, detail=response['answer'])
        return response
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
    import config
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
