# FastAPI Web Application

import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..pipeline import RAGPipeline
from ..config import get_config

logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str
    stream: bool = False


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    refused: bool
    refusal_reason: str
    original_query: str
    rewritten_query: str
    chapters_cited: list
    sources: list
    timing: dict


class IngestRequest(BaseModel):
    """Request model for ingestion."""
    novel_path: str
    force_reindex: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - init pipeline."""
    global pipeline
    pipeline = RAGPipeline()
    logger.info("RAG Pipeline initialized")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Novel RAG Chatbot",
        description="Ask questions about your novel",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Static files and templates
    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"
    
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Serve the main UI."""
        if templates:
            return templates.TemplateResponse("index.html", {"request": request})
        return HTMLResponse("<html><body><h1>Novel RAG Chatbot</h1><p>Templates not found</p></body></html>")
    
    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "pipeline_ready": pipeline.is_ready() if pipeline else False
        }
    
    @app.get("/api/stats")
    async def stats():
        """Get pipeline statistics."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        return pipeline.get_stats()
    
    @app.post("/api/ingest")
    async def ingest(request: IngestRequest):
        """Ingest a novel."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        novel_path = Path(request.novel_path)
        if not novel_path.exists():
            raise HTTPException(status_code=404, detail=f"Novel not found: {request.novel_path}")
        
        try:
            result = pipeline.ingest_novel(novel_path, request.force_reindex)
            return result
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Query the RAG system."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        if not pipeline.is_ready():
            raise HTTPException(status_code=400, detail="No novel indexed. Please ingest a novel first.")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        try:
            if request.stream:
                # Return streaming response
                return StreamingResponse(
                    pipeline.query_stream(request.query),
                    media_type="text/plain"
                )
            else:
                result = pipeline.query(request.query)
                return QueryResponse(
                    answer=result.answer,
                    refused=result.refused,
                    refusal_reason=result.refusal_reason,
                    original_query=result.original_query,
                    rewritten_query=result.rewritten_query,
                    chapters_cited=result.chapters_cited,
                    sources=result.sources,
                    timing=result.timing
                )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/query/stream")
    async def query_stream(request: QueryRequest):
        """Stream query response."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        if not pipeline.is_ready():
            raise HTTPException(status_code=400, detail="No novel indexed. Please ingest a novel first.")
        
        async def generate():
            for token in pipeline.query_stream(request.query):
                yield token
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    return app


# For running with uvicorn
app = create_app()
