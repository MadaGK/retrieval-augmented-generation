#!/usr/bin/env python3
"""Quick test of the RAG pipeline."""

import sys
sys.path.insert(0, '.')

print("Step 1: Testing imports...")
try:
    from src.pipeline import RAGPipeline
    print("  ✓ Pipeline imported")
except Exception as e:
    print(f"  ✗ Pipeline import failed: {e}")
    sys.exit(1)

print("\nStep 2: Creating pipeline...")
try:
    pipeline = RAGPipeline(
        data_dir="data/",
        embedder_provider="huggingface",
        llm_provider="ollama",  # Will fallback to SimpleFallbackLLM if unavailable
    )
    print("  ✓ Pipeline created")
except Exception as e:
    print(f"  ✗ Pipeline creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Loading and indexing documents...")
try:
    pipeline.load_and_index()
    print("  ✓ Documents loaded and indexed")
except Exception as e:
    print(f"  ✗ Document loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Running demo queries...")
try:
    questions = [
        "What is the main topic?",
        "What are the key points?",
    ]
    for q in questions:
        print(f"\n  Q: {q}")
        response = pipeline.query(q, return_sources=True)
        print(f"  A: {response['answer'][:100]}...")
    print("  ✓ Demo queries completed")
except Exception as e:
    print(f"  ✗ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
