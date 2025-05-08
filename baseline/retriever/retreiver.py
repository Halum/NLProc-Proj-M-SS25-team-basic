from preprocessor.document_reader import DocumentReader
from preprocessor.chunking_service import (
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
    SentenceBasedChunkingStrategy,
    ParagraphBasedChunkingStrategy,
    SemanticChunkingStrategy,
    MarkdownHeaderChunkingStrategy
)
from generator.llm import LLM

class Retriever:
    def __init__(self, chunking_strategy):
        self.__documents = []
        self.chunking_strategy = chunking_strategy
        self.__chunks = []
        
    def __reset__(self):
        self.__chunks = []        
    
    def add_document(self, document_path, is_directory=False):        
        if is_directory:
            # Load all __documents in the directory
            self.__documents.extend(DocumentReader.read_documents_in_dir(document_path))
        else:
            # Load a single document
            self.__documents.append(DocumentReader.read_document(document_path))
            
        self.__reset__()
        
        return self.__documents
    
    def query(self, query):
        pass
    
    def save(self):
        embeddings = LLM.generate_embedding(self.__chunks)
        
        return embeddings
    
    def load(self, path):
        for document in self.__documents:
            chunks = self.chunking_strategy.chunk(document)
            self.__chunks.extend(chunks)
            
        return self.__chunks