import os
import sys
from preprocessor.document_loader import load_document
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


def add_documents(folder_path: str) -> None:
    """
    Processes all readable documents in the specified folder.

    Args:
        folder_path (str): Path to the directory containing document files.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                content = load_document(file_path)
                # print(
                #     f"\n--- {filename} ---\n{content[:500]}\n"
                # )  # Print first 500 characters
                # chunker = FixedSizeChunkingStrategy(chunk_size=500)
                # chunker = SlidingWindowChunkingStrategy(chunk_size=100, overlap=20)
                # chunker = SentenceBasedChunkingStrategy(chunk_size=1000)
                # chunker = ParagraphBasedChunkingStrategy(chunk_size=500)
                # chunker = SemanticChunkingStrategy(chunk_size=1000, chunk_overlap=200)
                chunker = MarkdownHeaderChunkingStrategy()
                chunks = chunker.chunk(content)

                for i, chunk in enumerate(chunks):
                    print(f"\n--- Chunk {i+1} ---\n{chunk[:100]}")
                    if i == 1:
                        break
            except Exception as e:
                print(f"[ERROR] Failed to process '{filename}': {e}")


def main():
    """
    Main function to execute the document processing pipeline.
    """
    add_documents(DOCUMENT_FOLDER_PATH)


if __name__ == "__main__":
    main()
