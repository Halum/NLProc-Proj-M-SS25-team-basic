"""
Main entry point for the specialization pipeline.

This module provides the command-line interface and orchestration
for specialized NLP processing workflows.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specialization.config.config import *
from specialization.pipelines.raw_to_embeddings import EmbeddingsPipeline
from specialization.pipelines.raw_to_knowledge_graph import KnowledgeGraphPipeline
from specialization.pipelines.raw_to_sqlite import SQLitePipeline

def main():
    """
    Main function to execute specialized processing pipelines.
    """
    parser = argparse.ArgumentParser(description="Specialized NLP Pipeline")
    parser.add_argument(
        "--pipeline",
        choices=["embeddings", "knowledge_graph", "sqlite", "hybrid", "all"],
        default="all",
        help="Which pipeline to run"
    )
    parser.add_argument(
        "--input-dir",
        default=DOCUMENT_FOLDER_PATH,
        help="Input directory containing raw documents"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (pipeline-specific default used if not provided)"
    )
    parser.add_argument(
        "--config",
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after processing"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Specialized NLP Pipeline")
    print(f"📋 Pipeline: {args.pipeline}")
    print(f"📁 Input Directory: {args.input_dir}")
    
    try:
        if args.pipeline in ["embeddings", "all"]:
            print("🔢 Running Embeddings Pipeline...")
            pipeline = EmbeddingsPipeline()
            pipeline.run(args.input_dir, args.output_dir or EMBEDDINGS_OUTPUT_PATH)
            
        if args.pipeline in ["knowledge_graph", "all"]:
            print("🕸️ Running Knowledge Graph Pipeline...")
            pipeline = KnowledgeGraphPipeline()
            pipeline.run(args.input_dir, args.output_dir or KG_OUTPUT_PATH)
            
        if args.pipeline in ["sqlite", "all"]:
            print("🗄️ Running SQLite Pipeline...")
            pipeline = SQLitePipeline()
            pipeline.run(args.input_dir, args.output_dir or SQLITE_OUTPUT_PATH)
            
        if args.pipeline == "hybrid":
            print("🔄 Running Hybrid Pipeline...")
            # TODO: Implement hybrid pipeline
            pass
            
        if args.evaluate:
            print("📊 Running Evaluation...")
            # TODO: Implement evaluation
            pass
            
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
