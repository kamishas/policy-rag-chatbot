"""
Local LLM Service Module

This module handles interactions with Ollama (local LLM server) to generate
meaningful responses from retrieved Air Force document chunks.

Key Concepts:
- Ollama: Local LLM server that runs models like Llama 3.1 8B
- RAG Pattern: Retrieval + LLM Generation = Better answers
- Prompt Engineering: Crafting prompts to get good Air Force-specific responses
- Streaming: Real-time response generation for better UX
"""

import aiohttp
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaService:
    """
    Service for generating responses using Ollama (local LLM server).
    
    This class handles:
    - Connecting to local Ollama server
    - Prompt engineering for Air Force context
    - Generating responses from retrieved document chunks
    - Error handling and fallbacks
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize Ollama service.
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model name to use (default: llama3.2:3b)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = None
        logger.info(f"🦙 Initializing Ollama service...")
        logger.info(f"🔗 Server: {self.base_url}")
        logger.info(f"🤖 Model: {self.model}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def is_available(self) -> bool:
        """
        Check if Ollama server is running and model is available.
        
        Returns:
            True if Ollama is ready, False otherwise
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Check if Ollama server is running
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    models = await response.json()
                    model_names = [model['name'] for model in models.get('models', [])]
                    
                    if self.model in model_names:
                        logger.info(f"✅ Ollama server ready with model: {self.model}")
                        return True
                    else:
                        logger.warning(f"⚠️ Model {self.model} not found. Available models: {model_names}")
                        return False
                else:
                    logger.error(f"❌ Ollama server responded with status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {str(e)}")
            return False
    
    def _create_air_force_prompt(self, user_question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a specialized prompt for Air Force questions.
        
        This prompt engineering is crucial for getting good responses.
        
        Args:
            user_question: The user's question
            context_chunks: Retrieved document chunks with metadata
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get('metadata', {}).get('source', 'Unknown source')
            doc_type = chunk.get('metadata', {}).get('doc_type', 'Document')
            content = chunk.get('content', '')
            
            context_parts.append(f"**Source {i}** ({doc_type} - {source}):\n{content}\n")
        
        context_text = "\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Check if this is a personnel role question to format as checklist
        is_personnel_question = any(
            pattern in user_question.lower() 
            for pattern in ['role of', 'responsibilities of', 'duties of', 'what does', 'af/a3', 'deputy chief']
        )
        
        if is_personnel_question:
            # Format as checklist for personnel roles
            prompt = f"""You are an Air Force expert. Answer this question about personnel roles using ONLY the provided documentation.

QUESTION: {user_question}

DOCUMENTATION:
{context_text}

INSTRUCTIONS: 
- Format the response as a checklist of responsibilities
- Use ✓ checkmarks for each responsibility 
- Include the personnel's title at the top
- Be specific and cite the source document
- List ALL responsibilities found in the documentation

ANSWER FORMAT:
**[Personnel Title]**

✓ Responsibility 1
✓ Responsibility 2
✓ Responsibility 3
[continue for all responsibilities found]

Source: [Document citation]

ANSWER:"""
        else:
            # Standard format for non-personnel questions
            prompt = f"""You are an Air Force expert. Answer this question using ONLY the provided documentation.

QUESTION: {user_question}

DOCUMENTATION:
{context_text}

INSTRUCTIONS: Be concise, specific, and cite sources. Answer in 2-3 paragraphs maximum.

ANSWER:"""
        
        return prompt
    
    async def generate_response(
        self, 
        user_question: str, 
        context_chunks: List[Dict[str, Any]],
        stream: bool = False
    ) -> str:
        """
        Generate a response using the local LLM.
        
        Args:
            user_question: User's question
            context_chunks: Retrieved document chunks
            stream: Whether to stream the response (for future use)
            
        Returns:
            Generated response text
        """
        start_time = time.time()
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Create the prompt
            prompt = self._create_air_force_prompt(user_question, context_chunks)
            
            # Prepare the request
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,  # We'll handle streaming later
                "options": {
                    "temperature": 0.3,  # Lower temperature for more factual responses
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048,  # Reduced context window for faster processing
                    "num_predict": 512,  # Limit response length for speed
                }
            }
            
            logger.info(f"🦙 Generating response for: {user_question[:50]}...")
            
            # Make the request to Ollama
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=120)  # 2 minute timeout
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                
                result = await response.json()
                generated_text = result.get('response', '').strip()
                
                generation_time = time.time() - start_time
                logger.info(f"✅ LLM response generated in {generation_time:.2f}s")
                logger.info(f"📝 Response length: {len(generated_text)} characters")
                
                return generated_text
                
        except asyncio.TimeoutError:
            logger.error("⏰ LLM generation timed out")
            return self._create_fallback_response(user_question, context_chunks)
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {str(e)}")
            return self._create_fallback_response(user_question, context_chunks)
    
    def _create_fallback_response(self, user_question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a fallback response when LLM is unavailable.
        
        This ensures the system still works even if Ollama is down.
        """
        if not context_chunks:
            return f"I couldn't find specific information about '{user_question}' in the Air Force documentation. Please try rephrasing your question or asking about specific positions, commands, or organizational responsibilities."
        
        # Create a simple response from the context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get('metadata', {}).get('source', 'Unknown source')
            doc_type = chunk.get('metadata', {}).get('doc_type', 'Document')
            content = chunk.get('content', '')[:500]  # Limit length
            
            context_parts.append(f"**Source {i}** ({doc_type} - {source}):\n{content}...")
        
        response = f"Based on Air Force documentation, here's what I found regarding: **{user_question}**\n\n" + "\n\n---\n\n".join(context_parts) + "\n\n*Note: LLM processing unavailable - showing raw document excerpts.*"
        
        return response


# Global instance
llm_service = OllamaService()


async def test_ollama_connection():
    """Test function to check Ollama connectivity."""
    async with OllamaService() as service:
        is_ready = await service.is_available()
        if is_ready:
            print("✅ Ollama is ready!")
            
            # Test generation
            test_chunks = [{
                'content': 'The AFOSI Commander is responsible for directing all special investigations.',
                'metadata': {'source': 'AFI-71-101', 'doc_type': 'AFI'}
            }]
            
            response = await service.generate_response(
                "What does the AFOSI commander do?",
                test_chunks
            )
            print(f"🦙 Test response: {response[:200]}...")
        else:
            print("❌ Ollama not ready")


if __name__ == "__main__":
    # Test the service
    asyncio.run(test_ollama_connection())