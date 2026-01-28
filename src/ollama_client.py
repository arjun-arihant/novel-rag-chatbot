# Ollama HTTP Client - Direct API calls without LangChain

import json
import logging
import httpx
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API."""
    content: str
    model: str
    done: bool
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """Direct HTTP client for Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=120.0)
        
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        format: Optional[str] = None,
        stream: bool = False
    ) -> OllamaResponse:
        """
        Generate completion from Ollama.
        
        Args:
            model: Model name (e.g., "llama3.1:8b")
            prompt: The prompt text
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            format: Output format ("json" for JSON mode)
            stream: Whether to stream response
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
            
        if stop:
            payload["options"]["stop"] = list(stop)
            
        if format == "json":
            payload["format"] = "json"
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return OllamaResponse(
                content=data.get("response", "").strip(),
                model=data.get("model", model),
                done=data.get("done", True),
                total_duration=data.get("total_duration"),
                eval_count=data.get("eval_count")
            )
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise
            
    def generate_stream(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Iterator[str]:
        """Stream generation token by token."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
            
        if stop:
            payload["options"]["stop"] = list(stop)
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    
    def embed(self, model: str, text: str) -> List[float]:
        """Get embedding for text."""
        payload = {
            "model": model,
            "input": text
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/embed",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle both single embedding and batch format
            embeddings = data.get("embeddings", [])
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            return embeddings
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama embed error: {e}")
            raise
            
    def embed_batch(self, model: str, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        payload = {
            "model": model,
            "input": texts
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/embed",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get("embeddings", [])
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama batch embed error: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as e:
            logger.error(f"Failed to list models: {e}")
            return []
            
    def is_healthy(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
            
    def close(self):
        """Close the HTTP client."""
        self.client.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.close()


# Convenience singleton
_client: Optional[OllamaClient] = None


def get_client(base_url: str = "http://localhost:11434") -> OllamaClient:
    """Get or create global client instance."""
    global _client
    if _client is None:
        _client = OllamaClient(base_url)
    return _client
