"""
Example usage of the specialization module.

This script demonstrates how to use the specialized pipelines
and components for enhanced NLP processing.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from specialization.pipelines import EmbeddingsPipeline, KnowledgeGraphPipeline, SQLitePipeline
from specialization.config.config import *

def example_embeddings_pipeline():
    """Example of using the embeddings pipeline."""
    print("🔢 Running Embeddings Pipeline Example")
    
    pipeline = EmbeddingsPipeline()
    
    # Example input/output paths
    input_dir = "specialization/data/raw/documents"
    output_dir = "specialization/data/embeddings"
    
    try:
        results = pipeline.run(input_dir, output_dir)
        print(f"✅ Processed {results['total_documents']} documents")
        print(f"✅ Created {results['total_chunks']} chunks")
        print(f"✅ Output saved to: {results['output_files']}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

def example_knowledge_graph_pipeline():
    """Example of using the knowledge graph pipeline."""
    print("🕸️ Running Knowledge Graph Pipeline Example")
    
    pipeline = KnowledgeGraphPipeline()
    
    input_dir = "specialization/data/raw/documents"
    output_dir = "specialization/data/knowledge_graphs"
    
    try:
        results = pipeline.run(input_dir, output_dir)
        print(f"✅ Processed {results['total_documents']} documents")
        print(f"✅ Extracted {results['total_entities']} entities")
        print(f"✅ Found {results['total_relations']} relations")
        print(f"✅ Output saved to: {results['output_files']}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

def example_sqlite_pipeline():
    """Example of using the SQLite pipeline."""
    print("🗄️ Running SQLite Pipeline Example")
    
    pipeline = SQLitePipeline()
    
    input_dir = "specialization/data/raw/documents"
    output_dir = "specialization/data/sqlite"
    
    try:
        results = pipeline.run(input_dir, output_dir)
        print(f"✅ Processed {results['total_documents']} documents")
        print(f"✅ Created {results['total_chunks']} chunks")
        print(f"✅ Database saved to: {results['output_files']}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

def example_hybrid_workflow():
    """Example of a complete hybrid workflow."""
    print("🔄 Running Complete Hybrid Workflow Example")
    
    # Step 1: Create embeddings
    print("Step 1: Creating embeddings...")
    embeddings_pipeline = EmbeddingsPipeline()
    embeddings_results = embeddings_pipeline.run(
        "specialization/data/raw/documents",
        "specialization/data/embeddings"
    )
    
    # Step 2: Build knowledge graph
    print("Step 2: Building knowledge graph...")
    kg_pipeline = KnowledgeGraphPipeline()
    kg_results = kg_pipeline.run(
        "specialization/data/raw/documents",
        "specialization/data/knowledge_graphs"
    )
    
    # Step 3: Create SQLite database
    print("Step 3: Creating SQLite database...")
    sqlite_pipeline = SQLitePipeline()
    sqlite_results = sqlite_pipeline.run(
        "specialization/data/raw/documents",
        "specialization/data/sqlite"
    )
    
    print("🎉 Hybrid workflow completed!")
    print(f"📊 Total documents: {embeddings_results['total_documents']}")
    print(f"🔢 Total embeddings: {embeddings_results['total_chunks']}")
    print(f"🕸️ Total entities: {kg_results['total_entities']}")
    print(f"🗄️ Database records: {sqlite_results['total_chunks']}")

if __name__ == "__main__":
    print("🚀 Specialization Module Examples")
    print("=" * 50)
    
    # Make sure data directories exist
    os.makedirs("specialization/data/raw/documents", exist_ok=True)
    os.makedirs("specialization/data/embeddings", exist_ok=True)
    os.makedirs("specialization/data/knowledge_graphs", exist_ok=True)
    os.makedirs("specialization/data/sqlite", exist_ok=True)
    
    # Run examples
    example_embeddings_pipeline()
    print()
    
    example_knowledge_graph_pipeline()
    print()
    
    example_sqlite_pipeline()
    print()
    
    example_hybrid_workflow()
    print()
    
    print("✨ All examples completed!")
    print()
    print("📚 Next steps:")
    print("1. Add your documents to specialization/data/raw/documents/")
    print("2. Implement the TODO items in the pipeline files")
    print("3. Run: python specialization/main.py --pipeline all")
    print("4. Evaluate results and iterate!")
