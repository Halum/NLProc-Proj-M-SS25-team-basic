from preprocessor.document_reader import DocumentReader
from preprocessor.chunking_service import (
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
    SentenceBasedChunkingStrategy,
    ParagraphBasedChunkingStrategy,
    SemanticChunkingStrategy,
    MarkdownHeaderChunkingStrategy
)


class Retriever:
    def __init__(self, chunking_strategy):
        self.documents = []
        self.chunking_strategy = chunking_strategy
        
    
    def add_document(self, document_path, is_directory=False):        
        if is_directory:
            # Load all documents in the directory
            self.documents +=  DocumentReader.read_documents_in_dir(document_path)
        else:
            # Load a single document
            self.documents.append(DocumentReader.read_document(document_path))
    
    def query(self, query):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass