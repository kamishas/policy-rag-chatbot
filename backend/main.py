"""
Main FastAPI Application

This is the central API server for the Air Force RAG chatbot.
It provides endpoints for processing PDFs, searching documents, and chatting.

Key Concepts:
- FastAPI: Modern Python web framework for building APIs
- ASGI: Asynchronous Server Gateway Interface (handles concurrent requests)
- Pydantic: Data validation and serialization
- CORS: Cross-Origin Resource Sharing (allows React frontend to connect)
- RESTful API: Standard web API design patterns
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Import our custom modules (after environment is loaded)
from embeddings import embedding_service
from pdf_processor import pdf_processor
from vector_store import vector_store
from llm_service import llm_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Air Force RAG API",
    description="API for Air Force Roles & Responsibilities chatbot using RAG",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"  # ReDoc at http://localhost:8000/redoc
)

# Configure CORS for React frontend
# This allows your React app to call this API from a different port
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# ================================
# Pydantic Models (Data Validation)
# ================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's question or message", min_length=1)
    history: List[Dict[str, str]] = Field(default=[], description="Previous chat history")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "What are the responsibilities of SAF/AQ?",
                "history": [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"}
                ]
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI-generated response")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents used")
    processing_time: Optional[float] = Field(None, description="Time taken to process (seconds)")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "The Assistant Secretary of the Air Force for Acquisition (SAF/AQ) maintains responsibility for...",
                "sources": [
                    {
                        "document": "afi10-2402.pdf",
                        "score": 0.89,
                        "doc_type": "AFI"
                    }
                ],
                "processing_time": 1.23
            }
        }


class ProcessPDFRequest(BaseModel):
    """Request model for PDF processing."""
    pdf_urls: List[str] = Field(..., description="List of PDF URLs to process")
    
    class Config:
        schema_extra = {
            "example": {
                "pdf_urls": [
                    "https://static.e-publishing.af.mil/production/1/af_a3/publication/afi10-2402/afi10-2402.pdf"
                ]
            }
        }


class ProcessPDFResponse(BaseModel):
    """Response model for PDF processing."""
    total_pdfs: int = Field(..., description="Number of PDFs processed")
    total_chunks: int = Field(..., description="Total document chunks created")
    successful: int = Field(..., description="Successfully processed PDFs")
    failed: int = Field(..., description="Failed PDF processing attempts")
    processing_time: float = Field(..., description="Total processing time (seconds)")


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=50)
    doc_types: Optional[List[str]] = Field(default=None, description="Filter by document types")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "acquisition responsibilities",
                "top_k": 5,
                "doc_types": ["AFI", "AFMAN"]
            }
        }


class SearchResponse(BaseModel):
    """Response model for document search."""
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")


# ================================
# API Endpoints
# ================================

@app.get("/")
async def root():
    """
    Root endpoint - API health check.
    
    This is the simplest endpoint that confirms the API is running.
    """
    return {
        "message": "🚀 Air Force RAG API is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint.
    
    This endpoint checks if all services are working properly.
    """
    try:
        # Check embedding service
        embedding_info = embedding_service.get_model_info()
        
        # Check vector store
        vector_stats = await vector_store.get_index_stats()
        
        # Check PDF processor
        processor_info = pdf_processor.get_processor_stats()
        
        return {
            "status": "healthy",
            "services": {
                "embedding_service": {
                    "status": "ok",
                    "model": embedding_info["model_name"],
                    "dimension": embedding_info["dimension"]
                },
                "vector_store": {
                    "status": "ok",
                    "total_documents": vector_stats.get("total_vectors", 0),
                    "index_name": vector_store.index_name
                },
                "pdf_processor": {
                    "status": "ok",
                    "chunk_size": processor_info["chunk_size"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for the Air Force RAG chatbot.
    
    This endpoint:
    1. Takes a user question
    2. Converts it to an embedding
    3. Searches for relevant Air Force documents
    4. Builds a response based on found documents
    
    Args:
        request: Chat request with user message and optional history
        
    Returns:
        AI response with source citations
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"💬 Processing chat request: {request.message[:100]}...")
        
        # Step 1: Convert user question to embedding
        query_embedding = await embedding_service.get_embedding(request.message)
        
        # Step 2: Search for relevant documents
        search_results = await vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=10  # Get more candidates for better results
        )
        
        # Step 3: Filter results by relevance threshold
        # Only include results with high similarity scores
        relevant_results = [
            result for result in search_results 
            if result['score'] >= 0.6  # Minimum similarity threshold
        ]
        
        # Step 4: Prepare context chunks for LLM
        sources = []
        context_chunks = []
        
        if relevant_results:
            # Take top 5 most relevant results for better context
            top_results = relevant_results[:5]
            
            for result in top_results:
                # Prepare context chunk for LLM
                context_chunks.append({
                    'content': result['text'],
                    'metadata': {
                        'source': result['source'],
                        'doc_type': result['doc_type'],
                        'chunk_id': result['chunk_id'],
                        'page': result.get('page', 'Unknown')
                    }
                })
                
                # Collect source information for response
                sources.append({
                    "document": result['source'],
                    "doc_type": result['doc_type'],
                    "chunk_id": result['chunk_id'],
                    "similarity_score": round(result['score'], 3),
                    "page": result.get('page', 'Unknown')
                })
        
        # Step 5: Generate response using LLM
        try:
            # Check if Ollama is available
            if await llm_service.is_available():
                logger.info("🦙 Using LLM for response generation")
                response_text = await llm_service.generate_response(
                    user_question=request.message,
                    context_chunks=context_chunks
                )
            else:
                logger.warning("⚠️ LLM unavailable, using fallback response")
                response_text = llm_service._create_fallback_response(
                    user_question=request.message,
                    context_chunks=context_chunks
                )
        except Exception as llm_error:
            logger.error(f"❌ LLM generation failed: {str(llm_error)}")
            # Fallback to simple response
            if context_chunks:
                context_parts = []
                for i, chunk in enumerate(context_chunks[:3], 1):
                    source = chunk['metadata']['source']
                    doc_type = chunk['metadata']['doc_type']
                    content = chunk['content']
                    context_parts.append(f"**Source {i}** ({doc_type} - {source}):\n{content}")
                
                response_text = (
                    f"Based on Air Force documentation, here's what I found regarding: **{request.message}**\n\n"
                    + "\n\n---\n\n".join(context_parts) +
                    "\n\n*Note: LLM processing failed - showing raw document excerpts.*"
                )
            else:
                response_text = (
                    f"I couldn't find specific information about '{request.message}' "
                    "in the Air Force roles and responsibilities documentation. "
                    "Try rephrasing your question or asking about specific positions, "
                    "commands, or organizational responsibilities."
                )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Chat response generated in {processing_time:.2f}s with {len(sources)} sources")
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")


@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search endpoint for finding relevant Air Force documents.
    
    This endpoint allows direct searching without generating a chat response.
    Useful for exploring available documents or debugging.
    
    Args:
        request: Search request with query and optional filters
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        logger.info(f"🔍 Processing search request: {request.query}")
        
        # Convert search query to embedding
        query_embedding = await embedding_service.get_embedding(request.query)
        
        # Perform search with optional filters
        if request.doc_types:
            search_results = await vector_store.search_with_text_filter(
                query_embedding=query_embedding,
                doc_types=request.doc_types,
                top_k=request.top_k
            )
        else:
            search_results = await vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=request.top_k
            )
        
        logger.info(f"✅ Found {len(search_results)} documents for search query")
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"❌ Search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/process-pdfs", response_model=ProcessPDFResponse)
async def process_pdfs(request: ProcessPDFRequest, background_tasks: BackgroundTasks):
    """
    Process Air Force PDF documents and store them in the vector database.
    
    This endpoint:
    1. Downloads PDFs from provided URLs
    2. Extracts roles & responsibilities sections
    3. Creates embeddings for text chunks
    4. Stores everything in Pinecone
    
    Args:
        request: List of PDF URLs to process
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Processing summary with statistics
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"📄 Starting PDF processing for {len(request.pdf_urls)} PDFs")
        
        total_chunks = 0
        successful = 0
        failed = 0
        
        for i, pdf_url in enumerate(request.pdf_urls):
            try:
                logger.info(f"📄 Processing PDF {i+1}/{len(request.pdf_urls)}: {pdf_url}")
                
                # Step 1: Extract and chunk PDF content
                documents = await pdf_processor.process_pdf_from_url(pdf_url)
                
                if not documents:
                    logger.warning(f"⚠️ No content extracted from {pdf_url}")
                    failed += 1
                    continue
                
                # Step 2: Create embeddings for all chunks
                texts = [doc["text"] for doc in documents]
                embeddings = await embedding_service.get_embeddings_batch(texts)
                
                # Step 3: Store in vector database
                for doc, embedding in zip(documents, embeddings):
                    await vector_store.upsert_document(
                        text=doc["text"],
                        embedding=embedding,
                        metadata={
                            "source": doc["source"],
                            "chunk_id": doc["chunk_id"],
                            "doc_type": doc["doc_type"],
                            "section": doc["section"],
                            "total_chunks": doc["total_chunks"]
                        }
                    )
                
                total_chunks += len(documents)
                successful += 1
                
                logger.info(f"✅ Successfully processed {pdf_url} - {len(documents)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_url}: {str(e)}")
                failed += 1
                continue
        
        processing_time = time.time() - start_time
        
        logger.info(f"🎉 PDF processing complete! {successful}/{len(request.pdf_urls)} successful")
        
        return ProcessPDFResponse(
            total_pdfs=len(request.pdf_urls),
            total_chunks=total_chunks,
            successful=successful,
            failed=failed,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ PDF processing endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.post("/api/batch-process")
async def batch_process_all_pdfs():
    """
    Process all 1000+ Air Force PDFs.
    
    This is a convenience endpoint that processes your predefined list of
    Air Force PDF URLs. Use this to populate the database initially.
    
    ⚠️ Warning: This will take a long time (several hours) to complete!
    """
    # Your list of Air Force PDF URLs - Focused on Personnel Employment & Related Roles
    air_force_pdfs = [
        # Core Personnel Employment (PE-3/PE-4) Documents
        'https://static.e-publishing.af.mil/production/1/af_a3/publication/afi10-2402/afi10-2402.pdf',
        'https://static.e-publishing.af.mil/production/1/saf_ig/publication/afi71-101v3/afi71-101v3.pdf'
        ]
    
    # Use the existing process_pdfs endpoint
    request = ProcessPDFRequest(pdf_urls=air_force_pdfs)
    return await process_pdfs(request, BackgroundTasks())


@app.get("/api/stats")
async def get_system_stats():
    """
    Get system statistics and information.
    
    Returns:
        Current system status and database statistics
    """
    try:
        # Get vector database stats
        vector_stats = await vector_store.get_index_stats()
        
        # Get embedding service info
        embedding_info = embedding_service.get_model_info()
        
        # Get PDF processor info
        processor_info = pdf_processor.get_processor_stats()
        
        return {
            "database": {
                "total_documents": vector_stats.get("total_vectors", 0),
                "index_fullness": vector_stats.get("index_fullness", 0),
                "dimension": vector_stats.get("dimension", 0)
            },
            "embedding_service": embedding_info,
            "pdf_processor": processor_info,
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"❌ Stats endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ================================
# Startup and Shutdown Events
# ================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize services when the API starts.
    
    This runs once when the server starts up.
    """
    logger.info("🚀 Starting Air Force RAG API...")
    logger.info("✅ All services initialized successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup when the API shuts down.
    
    This runs when the server is stopped.
    """
    logger.info("🛑 Shutting down Air Force RAG API...")
    logger.info("✅ Cleanup completed!")


@app.delete("/api/clear-index")
async def clear_index():
    """
    Clear all vectors from the Pinecone index.
    
    ⚠️ WARNING: This will delete ALL documents from the vector database!
    Use this when you want to start fresh with new PDF processing.
    
    Returns:
        Confirmation of deletion
    """
    import time
    start_time = time.time()
    
    try:
        logger.info("🗑️ Clear index request received")
        
        # Get current stats
        stats = await vector_store.get_index_stats()
        initial_count = stats.get('total_vectors', 0)
        
        logger.info(f"📊 Current vectors in index: {initial_count}")
        
        if initial_count == 0:
            return {
                "message": "Index is already empty",
                "vectors_deleted": 0,
                "processing_time": round(time.time() - start_time, 2)
            }
        
        # Clear the index
        success = await vector_store.clear_all_vectors()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear index")
        
        # Check final stats
        final_stats = await vector_store.get_index_stats()
        final_count = final_stats.get('total_vectors', 0)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Index cleared in {processing_time:.2f}s")
        logger.info(f"📊 Vectors deleted: {initial_count - final_count}")
        
        return {
            "message": "Index successfully cleared",
            "initial_vectors": initial_count,
            "final_vectors": final_count,
            "vectors_deleted": initial_count - final_count,
            "processing_time": round(processing_time, 2),
            "note": "It may take a few moments for all vectors to be completely removed from Pinecone"
        }
        
    except Exception as e:
        logger.error(f"❌ Clear index error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")


# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    # Run the server
    logger.info(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",  # Import path to the FastAPI app
        host=host,
        port=port,
        reload=debug,  # Auto-reload on code changes in debug mode
        log_level="info"
    )

"""
API Usage Examples:

1. Health Check:
   GET http://localhost:8000/health

2. Chat with the bot:
   POST http://localhost:8000/api/chat
   Body: {"message": "What does SAF/AQ do?"}

3. Search documents:
   POST http://localhost:8000/api/search  
   Body: {"query": "acquisition responsibilities", "top_k": 5}

4. Process PDFs:
   POST http://localhost:8000/api/process-pdfs
   Body: {"pdf_urls": ["https://example.com/document.pdf"]}

5. Get system stats:
   GET http://localhost:8000/api/stats

6. API Documentation:
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)
"""