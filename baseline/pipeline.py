import os
import sys
from baseline.retriever import retreiver
# from baseline.preprocessor.document_reader import load_document
from preprocessor.chunking_service import (
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
    SentenceBasedChunkingStrategy,
    ParagraphBasedChunkingStrategy,
    SemanticChunkingStrategy,
    MarkdownHeaderChunkingStrategy
)

# Set the path to the config directory
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(config_path)

from config.config import DOCUMENT_FOLDER_PATH


def main():
    """
    Main function to execute the document processing pipeline.
    """
    chunking_stratigies = [
        FixedSizeChunkingStrategy(chunk_size=1000),
        SlidingWindowChunkingStrategy(chunk_size=100, overlap=20),
        SentenceBasedChunkingStrategy(chunk_size=1000),
        ParagraphBasedChunkingStrategy(chunk_size=500),
        SemanticChunkingStrategy(chunk_size=1000, chunk_overlap=200),
        MarkdownHeaderChunkingStrategy()
    ]
    
    retrievers = []
    for strategy in chunking_stratigies:
        retriever = Retriever(strategy)
        retrievers.append(retriever)
        
    for retriever in retrievers:
        retriever.add_document(DOCUMENT_FOLDER_PATH, is_directory=True)
    
    


if __name__ == "__main__":
    main()
