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
from retriever.vector_store import VectorStoreFaiss

class Retriever:
    def __init__(self, chunking_strategy):
        self.__documents = []
        self.chunking_strategy = chunking_strategy
        self.__chunks = []
        self.__vector_store = VectorStoreFaiss(LLM.embedding_dimensions())
        
    def __reset__(self):
        self.__chunks = []
        self.__vector_store.cleanup()
        
    def __get_relevant_chunks(self, indices):
        relevant_chunks = []
        
        for i in indices[0]:
            relevant_chunks.append(self.__chunks[i])
        return relevant_chunks
    
    def add_document(self, document_path, is_directory=False):        
        if is_directory:
            # Load all __documents in the directory
            self.__documents.extend(DocumentReader.read_documents_in_dir(document_path))
        else:
            # Load a single document
            self.__documents.append(DocumentReader.read_document(document_path))
            
        self.__reset__()
        
        return self.__documents
    
    def save(self):
        embeddings = LLM.generate_embedding(self.__chunks)
        self.__vector_store.add(embeddings)
        
        return embeddings
    
    def preprocess(self):
        for document in self.__documents:
            chunks = self.chunking_strategy.chunk(document)
            self.__chunks.extend(chunks)
            
        return self.__chunks
    
    def load(self, query):
        query_embedding = LLM.generate_embedding([query])
        distances, indices = self.__vector_store.search(query_embedding)
        retrieved_chunks = self.__get_relevant_chunks(indices)
        
        return retrieved_chunks
    
    def query(self, query, relevant_chunks):
        answering_prompt = LLM.generate_answering_prompt(query, relevant_chunks)
        answer = LLM.invoke_llm(answering_prompt)
        
        return answer