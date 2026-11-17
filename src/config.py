import os
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# NVIDIA NIM / LLM Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME", "nvidia/nemotron-3-ultra-550b-a55b")

# Embedding Configuration
# Options: "local" (fast ONNX CPU), "nvidia" (NVIDIA NeMo Retriever), "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")

# Search API
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Storage & Document Paths
POLICY_PDF_PATH = os.getenv("POLICY_PDF_PATH", str(BASE_DIR / "data" / "optima-secure-revision-pw.pdf"))
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "chroma_db"))

def get_llm(temperature: float = 0.0, max_tokens: int = 2048, streaming: bool = False):
    """
    Factory function for Chat Model via NVIDIA NIM (or OpenAI compatible endpoint).
    Supports token streaming for near-instant Time-To-First-Token (TTFT).
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("NVIDIA_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("API key not configured! Please set NVIDIA_API_KEY in .env or the UI.")

    return ChatOpenAI(
        model=os.getenv("NVIDIA_MODEL_NAME", NVIDIA_MODEL_NAME),
        openai_api_key=api_key,
        openai_api_base=os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL),
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )

def get_embeddings(provider: Optional[str] = None):
    """
    Factory function supporting multiple embedding techniques:
      1. 'local': Chroma's built-in ONNX all-MiniLM-L6-v2 (Zero cost, offline, low latency)
      2. 'nvidia': NVIDIA NIM Retriever (nvidia/nv-embedqa-e5-v5 / baai/bge-m3)
      3. 'openai': text-embedding-3-small
    """
    prov = provider or os.getenv("EMBEDDING_PROVIDER", "local").lower()

    if prov == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY required for NVIDIA embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=os.getenv("NVIDIA_EMBEDDING_MODEL", NVIDIA_EMBEDDING_MODEL),
            openai_api_base=os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL),
            openai_api_key=api_key
        )
    elif prov == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    else:
        # Default local embedding function
        return None  # Triggers ChromaDB default ONNX MiniLM
