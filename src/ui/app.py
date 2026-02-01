# FastAPI Web Application - Multi-Novel Support

import asyncio
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from queue import Queue
from threading import Thread

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..config import get_config

logger = logging.getLogger(__name__)

# Global pipeline instance - will be initialized based on config
pipeline = None


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


class NovelUploadResponse(BaseModel):
    """Response for novel upload."""
    id: str
    title: str
    author: str
    status: str
    chapters: int
    chunks: int


class SelectNovelRequest(BaseModel):
    """Request to select a novel."""
    novel_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - init pipeline."""
    global pipeline
    
    # Determine pipeline mode from config
    config = get_config()
    mode = getattr(config, 'pipeline_mode', 'simple')
    
    if mode == "simple":
        from ..simple_pipeline import SimpleRAGPipeline
        pipeline = SimpleRAGPipeline()
        logger.info("RAG Pipeline initialized (SIMPLE mode - dense retrieval only)")
    else:
        from ..pipeline import RAGPipeline
        pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized (ADVANCED mode - hybrid + reranking)")
    
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Novel RAG Chatbot",
        description="Chat with your novels - supports multiple books",
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
        return HTMLResponse("<html><body><h1>Novel RAG Chatbot</h1></body></html>")
    
    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "ready": pipeline.is_ready() if pipeline else False
        }
    
    @app.get("/api/stats")
    async def stats():
        """Get pipeline statistics."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        return pipeline.get_stats()
    
    # === Novel Management Endpoints ===
    
    @app.get("/api/novels")
    async def list_novels():
        """List all novels in the library."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        return {"novels": pipeline.list_novels()}
    
    @app.post("/api/novels")
    async def upload_novel(
        file: UploadFile = File(...),
        title: str = Form(None),
        author: str = Form("Unknown")
    ):
        """Upload and process a new novel."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        # Validate file type
        allowed_extensions = [".txt", ".pdf", ".epub"]
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file temporarily
        temp_path = Path("library/uploads") / file.filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            content = await file.read()
            temp_path.write_bytes(content)
            
            # Add and index novel
            result = pipeline.add_novel(
                temp_path,
                title=title or Path(file.filename).stem,
                author=author
            )
            
            # Clean up temp file (library creates its own copy)
            if temp_path.exists():
                temp_path.unlink()
            
            if result["indexing"]["status"] == "error":
                raise HTTPException(
                    status_code=500,
                    detail=result["indexing"]["error"]
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/novels/{novel_id}")
    async def get_novel(novel_id: str):
        """Get details for a specific novel."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        novels = pipeline.list_novels()
        novel = next((n for n in novels if n["id"] == novel_id), None)
        
        if not novel:
            raise HTTPException(status_code=404, detail="Novel not found")
        
        return novel
    
    @app.delete("/api/novels/{novel_id}")
    async def delete_novel(novel_id: str):
        """Delete a novel from the library."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        if pipeline.delete_novel(novel_id):
            return {"status": "deleted", "id": novel_id}
        
        raise HTTPException(status_code=404, detail="Novel not found")
    
    @app.post("/api/novels/{novel_id}/select")
    async def select_novel(novel_id: str):
        """Select a novel for querying."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        if pipeline.select_novel(novel_id):
            return {"status": "selected", "novel": pipeline.get_active_novel()}
        
        raise HTTPException(
            status_code=400, 
            detail="Could not select novel. Make sure it's fully indexed."
        )
    
    @app.post("/api/novels/{novel_id}/reindex")
    async def reindex_novel(novel_id: str):
        """Re-index a novel (for incremental updates)."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        result = pipeline.reindex_novel(novel_id)
        return result
    
    @app.get("/api/novels/active")
    async def get_active_novel():
        """Get the currently selected novel."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        active = pipeline.get_active_novel()
        if not active:
            return {"active": None}
        return {"active": active}
    
    # === Query Endpoints ===
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Query the active novel."""
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        try:
            if request.stream:
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
        
        async def generate():
            for token in pipeline.query_stream(request.query):
                yield token
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    return app


# For running with uvicorn
app = create_app()
