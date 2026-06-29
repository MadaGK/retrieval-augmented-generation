"""
Embedding Model
===============
Students must choose and configure the embedding model.
"""

from typing import List, Optional
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder(
    provider: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
):
    """
    Get an embedding model.

    Students: Modify this:
    - Try different embedding models (e.g., BAAI/bge-small-en)
    - Test Ollama embeddings for local inference

    Args:
        provider: "huggingface", "openai", or "ollama"
        model_name: Name of the embedding model
    """
    if provider == "huggingface":
        # Open-source, free to use
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    elif provider == "openai":
        # Requires OPENAI_API_KEY
        # Explore the following link for free usable api keys:
        # https://console.groq.com/keys
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)
    elif provider == "ollama":
        # Local inference via Ollama
        return OllamaEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def embed_documents(embedder, documents, batch_size= 64):
    """
    Embed a list of documents.
    """
    texts = [doc.page_content for doc in documents]
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings


def embed_query(embedder, query: str) -> List[float]:
    """
    Embed a query string.
    """
    return embedder.embed_query(query)


if __name__ == "__main__":
    embedder = get_embedder(provider="huggingface")
    test_text = "What is Retrieval-Augmented Generation?"
    embedding = embed_query(embedder, test_text)
    print(f"Embedding dimension: {len(embedding)}")
