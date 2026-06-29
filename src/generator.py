"""
LLM Generator Module
====================
Students: customize the prompt and LLM configuration.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)


class SimpleFallbackLLM(LLM):
    """Simple fallback LLM that works without API keys."""
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "simple_fallback"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Simple text processing without API calls."""
        lines = prompt.split('\n')
        question = None
        context = ""
        
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif not line.startswith(("Context:", "Answer:")) and line.strip():
                context += line + " "
        
        if not question or not context.strip():
            return "I don't have enough context to answer this question."
        
        question_words = set(question.lower().split())
        context_sentences = context.split('.')
        
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > 1:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = " ".join(relevant_sentences[:2])
        else:
            answer = f"Based on the provided context: {context[:200]}..."
        
        return answer if answer else "I cannot provide a specific answer based on available context."


def get_llm(
    provider: str = "ollama",
    model_name: str = "phi3",
    temperature: float = 0.7,
    **kwargs
):
    """
    Get an LLM for generation with automatic fallback.

    If a cloud provider is requested but API key is missing, 
    automatically falls back to SimpleFallbackLLM.

    modify this:
    - Choose appropriate model (phi3, llama3, mistral, etc.)
    - Adjust temperature for creativity vs accuracy
    - Add API key configurations

    Recommended local models:
    - phi3 (small, fast)
    - llama3 (better quality)
    - mistral (balanced)
    """
    if provider == "ollama":
        try:
            llm = Ollama(model=model_name, temperature=temperature)
            # Test connection by attempting a simple invoke
            llm.invoke("test")
            return llm
        except Exception as e:
            logger.warning(f"Ollama not available ({type(e).__name__}), using simple fallback")
            return SimpleFallbackLLM(temperature=temperature)
    elif provider == "huggingface":
        logger.warning("HuggingFace requires API token, using simple fallback")
        return SimpleFallbackLLM(temperature=temperature)
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set, using simple fallback")
            return SimpleFallbackLLM(temperature=temperature)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            **kwargs
        )
    elif provider == "google":
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            logger.warning("Google API key not set, using simple fallback")
            return SimpleFallbackLLM(temperature=temperature)
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_genai_api_key=kwargs.get("google_genai_api_key") or os.getenv("GOOGLE_API_KEY"),
            **kwargs
        )
    else:
        logger.warning(f"Unknown provider {provider}, using simple fallback")
        return SimpleFallbackLLM(temperature=temperature)


def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None
) -> PromptTemplate:
    """
    Create a RAG prompt template.

    modify:
    - Customize the system message for their scenario
    - Add citation/instruction formatting
    - Include few-shot examples
    """
    if system_message is None:
        system_message = """You are a helpful AI Lab and software installation assistant.
                            Use the retrieved context to answer the user's question.
                            If you don't know the answer, say so clearly.
                            Always cite your sources when possible."""

    if template is None:
        template = """Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return prompt


def create_qa_chain(llm, retriever, prompt: Optional[PromptTemplate] = None):
    """
    Create a simple QA chain combining retriever and LLM.

    Students MUST modify:
    - Implement custom context formatting
    - Add return_source_documents=True
    - Implement custom output parsing
    """
    if prompt is None:
        prompt = create_rag_prompt()
    
    # Simple wrapper to combine retriever and LLM
    class SimpleQAChain:
        def __init__(self, llm, retriever, prompt):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
        
        def invoke(self, inputs):
            query = inputs.get("query", inputs)
            # Retrieve relevant documents
            documents = self.retriever.invoke(query)
            
            # Format context from documents
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # Format prompt
            formatted_prompt = self.prompt.format(context=context, question=query)
            
            # Generate response
            response_text = self.llm.invoke(formatted_prompt)
            
            return {
                "result": str(response_text),
                "source_documents": documents
            }
    
    return SimpleQAChain(llm, retriever, prompt)


def generate_response(
    qa_chain,
    query: str,
    return_sources: bool = True
) -> Dict:
    """
    Generate a response using the RAG pipeline.

    Returns:
        Dict with 'answer' and optionally 'source_documents'
    """
    result = qa_chain.invoke({"query": query})

    response = {
        "answer": result["result"]
    }

    if return_sources and "source_documents" in result:
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
        response["sources"] = sources

    return response


if __name__ == "__main__":
    # Test prompt creation
    prompt = create_rag_prompt()
    print("Default prompt template:")
    print(prompt.template)
