from preprocessor.document_reader import DocumentReader

from generator.llm import LLM
from retriever.vector_store import VectorStoreFaiss

class Retriever:
    """
    Class responsible for document retrieval using vector embeddings.
    Handles document processing, chunking, embedding, storage, and querying.
    """
    
    def __init__(self, chunking_strategy):
        """
        Initialize the Retriever with a specified chunking strategy.
        
        Args:
            chunking_strategy: Strategy object for dividing documents into chunks.
        """
        self.__documents = []
        self.chunking_strategy = chunking_strategy
        self.__chunks = []
        self.__vector_store = VectorStoreFaiss(LLM.embedding_dimensions())
        
    def __reset__(self):
        """
        Reset the retriever by clearing chunks and vector store.
        """
        self.__chunks = []
        self.__vector_store.cleanup()
        
    def __get_relevant_chunks(self, indices):
        """
        Get chunks corresponding to the provided indices.
        
        Args:
            indices (list): List of indices of relevant chunks.
            
        Returns:
            list: List of relevant text chunks.
        """
        relevant_chunks = []
        
        for i in indices[0]:
            relevant_chunks.append(self.__chunks[i])
        return relevant_chunks
    
    def add_document(self, document_path, is_directory=False):
        """
        Add documents from a file or directory to the retriever.
        
        Args:
            document_path (str): Path to the document file or directory.
            is_directory (bool, optional): Whether the path is a directory. Defaults to False.
            
        Returns:
            list: List of document contents.
        """
        if is_directory:
            # Load all __documents in the directory
            self.__documents.extend(DocumentReader.read_documents_in_dir(document_path))
        else:
            # Load a single document
            self.__documents.append(DocumentReader.read_document(document_path))
            
        self.__reset__()
        
        return self.__documents
    
    def save(self):
        """
        Generate embeddings for chunks and save them to vector store.
        
        Returns:
            list: Generated embeddings.
        """
        embeddings = LLM.generate_embedding(self.__chunks)
        self.__vector_store.add(embeddings)
        
        return embeddings
    
    def preprocess(self):
        """
        Process documents to generate chunks using the specified chunking strategy.
        
        Returns:
            list: List of document chunks.
        """
        for document in self.__documents:
            chunks = self.chunking_strategy.chunk(document)
            self.__chunks.extend(chunks)
            
        return self.__chunks
    
    def load(self, query):
        """
        Retrieve chunks relevant to the query using vector similarity search.
        
        Args:
            query (str): The query to search for.
            
        Returns:
            list: List of relevant chunks.
        """
        query_embedding = LLM.generate_embedding([query])
        distances, indices = self.__vector_store.search(query_embedding)
        retrieved_chunks = self.__get_relevant_chunks(indices)
        
        return retrieved_chunks, distances
    
    def query(self, query, relevant_chunks):
        """
        Generate an answer to the query using the relevant chunks as context.
        
        Args:
            query (str): The query to answer.
            relevant_chunks (list): List of text chunks to use as context.
            
        Returns:
            str: Generated answer to the query.
        """
        answering_prompt = LLM.generate_answering_prompt(query, relevant_chunks)
        answer = LLM.invoke_llm(answering_prompt)
        
        return answer