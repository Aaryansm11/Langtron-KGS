#!/usr/bin/env python3
"""
Main FastAPI application for Document Knowledge Graph Extraction
"""

import os
import asyncio
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Internal imports
from utils.parser import DocumentParser
from app.ner_model import NERPipeline
from app.relation_extractor import RelationExtractor
from app.kg_builder import KnowledgeGraphBuilder
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global variables
config = Config()
document_parser = None
ner_model = None
relation_extractor = None
kg_builder = None

# Processing status storage
processing_status = {}


class ProcessingStatus(BaseModel):
    """Processing status model."""
    task_id: str
    status: str  # 'pending', 'processing', 'completed', 'error'
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ExtractionRequest(BaseModel):
    """Request model for extraction."""
    text: str = Field(..., description="Text to extract entities and relations from")
    include_entities: bool = Field(True, description="Include entity extraction")
    include_relations: bool = Field(True, description="Include relation extraction")
    include_graph: bool = Field(True, description="Include knowledge graph building")
    confidence_threshold: float = Field(0.5, description="Minimum confidence threshold")


class ExtractionResponse(BaseModel):
    """Response model for extraction."""
    entities: List[Dict] = Field(default_factory=list)
    relations: List[Dict] = Field(default_factory=list)
    graph: Optional[Dict] = None
    metadata: Dict = Field(default_factory=dict)
    processing_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting up Document Knowledge Graph Extraction API")
    await initialize_models()
    yield
    # Shutdown
    logger.info("Shutting down Document Knowledge Graph Extraction API")
    await cleanup_resources()


async def initialize_models():
    """Initialize all models and components."""
    global document_parser, ner_model, relation_extractor, kg_builder
    
    try:
        logger.info("Initializing models...")
        
        # Initialize document parser
        document_parser = DocumentParser()
        
        # Initialize NER model
        ner_model = NERModel()
        await asyncio.to_thread(ner_model.load_model)
        
        # Initialize relation extractor
        relation_extractor = RelationExtractor()
        await asyncio.to_thread(relation_extractor.load_model)
        
        # Initialize knowledge graph builder
        kg_builder = KnowledgeGraphBuilder()
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise


async def cleanup_resources():
    """Cleanup resources on shutdown."""
    try:
        # Cleanup temporary files
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith("doc_kg_"):
                try:
                    os.remove(os.path.join(temp_dir, filename))
                except Exception:
                    pass
        
        logger.info("Resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error cleaning up resources: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Document Knowledge Graph Extraction API",
    description="Extract entities, relations, and build knowledge graphs from documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    if not config.API_KEY:
        return True  # No authentication required
    
    if credentials.credentials != config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Document Knowledge Graph Extraction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload",
            "extract": "/extract",
            "entities": "/entities",
            "relations": "/relations",
            "graph": "/graph",
            "status": "/status/{task_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if models are loaded
        models_status = {
            "document_parser": document_parser is not None,
            "ner_model": ner_model is not None and ner_model.model is not None,
            "relation_extractor": relation_extractor is not None and relation_extractor.model is not None,
            "kg_builder": kg_builder is not None
        }
        
        all_healthy = all(models_status.values())
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "models": models_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: bool = Depends(verify_token)
):
    """Upload and process document."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        if file.size and file.size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE} bytes"
            )
        
        # Check file type
        allowed_types = ['.pdf', '.docx', '.txt', '.md']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {allowed_types}"
            )
        
        # Generate task ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Initialize processing status
        processing_status[task_id] = ProcessingStatus(
            task_id=task_id,
            status="pending",
            message="File uploaded, processing queued"
        )
        
        # Save uploaded file
        temp_file_path = os.path.join(
            tempfile.gettempdir(),
            f"doc_kg_{task_id}{file_ext}"
        )
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add background task
        background_tasks.add_task(
            process_document_async,
            task_id,
            temp_file_path,
            file.filename
        )
        
        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "Document uploaded and queued for processing",
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Upload failed")


async def process_document_async(task_id: str, file_path: str, filename: str):
    """Process document asynchronously."""
    try:
        # Update status
        processing_status[task_id].status = "processing"
        processing_status[task_id].message = "Parsing document..."
        processing_status[task_id].progress = 0.1
        processing_status[task_id].updated_at = datetime.now()
        
        # Parse document
        text = await asyncio.to_thread(document_parser.parse_document, file_path)
        
        # Update status
        processing_status[task_id].message = "Extracting entities..."
        processing_status[task_id].progress = 0.3
        processing_status[task_id].updated_at = datetime.now()
        
        # Extract entities
        entities = await asyncio.to_thread(ner_model.extract_entities, text)
        
        # Update status
        processing_status[task_id].message = "Extracting relations..."
        processing_status[task_id].progress = 0.6
        processing_status[task_id].updated_at = datetime.now()
        
        # Extract relations
        relations = await asyncio.to_thread(relation_extractor.extract_relations, text, entities)
        
        # Update status
        processing_status[task_id].message = "Building knowledge graph..."
        processing_status[task_id].progress = 0.8
        processing_status[task_id].updated_at = datetime.now()
        
        # Build knowledge graph
        graph = kg_builder.build_knowledge_graph(entities, relations)
        enhanced_graph = kg_builder.enhance_graph(graph)
        optimized_graph = kg_builder.optimize_graph(enhanced_graph)
        
        # Update status - completed
        processing_status[task_id].status = "completed"
        processing_status[task_id].message = "Processing completed successfully"
        processing_status[task_id].progress = 1.0
        processing_status[task_id].updated_at = datetime.now()
        processing_status[task_id].result = {
            "entities": entities,
            "relations": relations,
            "graph": optimized_graph,
            "metadata": {
                "filename": filename,
                "total_entities": len(entities),
                "total_relations": len(relations),
                "graph_nodes": len(optimized_graph['nodes']),
                "graph_edges": len(optimized_graph['edges']),
                "processed_at": datetime.now().isoformat()
            }
        }
        
        # Cleanup temp file
        try:
            os.remove(file_path)
        except Exception:
            pass
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        processing_status[task_id].status = "error"
        processing_status[task_id].message = "Processing failed"
        processing_status[task_id].error = str(e)
        processing_status[task_id].updated_at = datetime.now()


@app.get("/status/{task_id}")
async def get_processing_status(task_id: str, _: bool = Depends(verify_token)):
    """Get processing status."""
    if task_id not in processing_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return processing_status[task_id]


@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_text(
    request: ExtractionRequest,
    _: bool = Depends(verify_token)
):
    """Extract entities and relations from text."""
    try:
        start_time = datetime.now()
        
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        if len(request.text) > config.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"Text too long. Maximum length: {config.MAX_TEXT_LENGTH} characters"
            )
        
        result = ExtractionResponse()
        
        # Extract entities
        if request.include_entities:
            entities = await asyncio.to_thread(ner_model.extract_entities, request.text)
            # Filter by confidence
            result.entities = [
                entity for entity in entities
                if entity.get('confidence', 0) >= request.confidence_threshold
            ]
        
        # Extract relations
        if request.include_relations:
            relations = await asyncio.to_thread(
                relation_extractor.extract_relations,
                request.text,
                result.entities
            )
            # Filter by confidence
            result.relations = [
                relation for relation in relations
                if relation.get('confidence', 0) >= request.confidence_threshold
            ]
        
        # Build knowledge graph
        if request.include_graph and result.entities:
            graph = kg_builder.build_knowledge_graph(result.entities, result.relations)
            result.graph = kg_builder.enhance_graph(graph)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result.processing_time = processing_time
        
        # Add metadata
        result.metadata = {
            "text_length": len(request.text),
            "entities_count": len(result.entities),
            "relations_count": len(result.relations),
            "graph_nodes": len(result.graph['nodes']) if result.graph else 0,
            "graph_edges": len(result.graph['edges']) if result.graph else 0,
            "confidence_threshold": request.confidence_threshold,
            "processing_time": processing_time,
            "processed_at": datetime.now().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Extraction failed")


@app.get("/entities")
async def get_entities(
    text: str = Query(..., description="Text to extract entities from"),
    confidence_threshold: float = Query(0.5, description="Minimum confidence threshold"),
    _: bool = Depends(verify_token)
):
    """Extract entities from text."""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        entities = await asyncio.to_thread(ner_model.extract_entities, text)
        
        # Filter by confidence
        filtered_entities = [
            entity for entity in entities
            if entity.get('confidence', 0) >= confidence_threshold
        ]
        
        return {
            "entities": filtered_entities,
            "total_count": len(filtered_entities),
            "confidence_threshold": confidence_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Entity extraction failed")


@app.get("/relations")
async def get_relations(
    text: str = Query(..., description="Text to extract relations from"),
    confidence_threshold: float = Query(0.5, description="Minimum confidence threshold"),
    _: bool = Depends(verify_token)
):
    """Extract relations from text."""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        # First extract entities
        entities = await asyncio.to_thread(ner_model.extract_entities, text)
        
        # Then extract relations
        relations = await asyncio.to_thread(
            relation_extractor.extract_relations,
            text,
            entities
        )
        
        # Filter by confidence
        filtered_relations = [
            relation for relation in relations
            if relation.get('confidence', 0) >= confidence_threshold
        ]
        
        return {
            "relations": filtered_relations,
            "total_count": len(filtered_relations),
            "confidence_threshold": confidence_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Relation extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Relation extraction failed")


@app.get("/graph")
async def get_knowledge_graph(
    text: str = Query(..., description="Text to build graph from"),
    confidence_threshold: float = Query(0.5, description="Minimum confidence threshold"),
    format: str = Query("json", description="Output format (json, gexf, graphml)"),
    _: bool = Depends(verify_token)
):
    """Build knowledge graph from text."""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Extract entities and relations
        entities = await asyncio.to_thread(ner_model.extract_entities, text)
        relations = await asyncio.to_thread(
            relation_extractor.extract_relations,
            text,
            entities
        )
        
        # Filter by confidence
        filtered_entities = [
            entity for entity in entities
            if entity.get('confidence', 0) >= confidence_threshold
        ]
        filtered_relations = [
            relation for relation in relations
            if relation.get('confidence', 0) >= confidence_threshold
        ]
        
        # Build graph
        graph = kg_builder.build_knowledge_graph(filtered_entities, filtered_relations)
        enhanced_graph = kg_builder.enhance_graph(graph)
        optimized_graph = kg_builder.optimize_graph(enhanced_graph)
        
        # Return in requested format
        if format.lower() == "json":
            return optimized_graph
        else:
            # Save to temporary file and return as download
            temp_file = os.path.join(
                tempfile.gettempdir(),
                f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            )
            
            success = kg_builder.save_graph(optimized_graph, temp_file, format)
            if success:
                return FileResponse(
                    temp_file,
                    media_type="application/octet-stream",
                    filename=f"knowledge_graph.{format}"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to generate graph file")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph building error: {str(e)}")
        raise HTTPException(status_code=500, detail="Graph building failed")


@app.get("/graph/statistics")
async def get_graph_statistics(
    text: str = Query(..., description="Text to analyze"),
    confidence_threshold: float = Query(0.5, description="Minimum confidence threshold"),
    _: bool = Depends(verify_token)
):
    """Get knowledge graph statistics."""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Extract entities and relations
        entities = await asyncio.to_thread(ner_model.extract_entities, text)
        relations = await asyncio.to_thread(
            relation_extractor.extract_relations,
            text,
            entities
        )
        
        # Filter by confidence
        filtered_entities = [
            entity for entity in entities
            if entity.get('confidence', 0) >= confidence_threshold
        ]
        filtered_relations = [
            relation for relation in relations
            if relation.get('confidence', 0) >= confidence_threshold
        ]
        
        # Build graph
        graph = kg_builder.build_knowledge_graph(filtered_entities, filtered_relations)
        enhanced_graph = kg_builder.enhance_graph(graph)
        
        # Get statistics
        stats = kg_builder.get_graph_statistics(enhanced_graph)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Graph statistics failed")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )