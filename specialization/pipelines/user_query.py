#!/usr/bin/env python3
"""
User Query Pipeline for Specialization Track

This pipeline provides an interactive user query interface that:
1. Loads existing embeddings from ChromaDB
2. Takes user queries in a loop
3. Retrieves relevant information using enhanced retriever with metadata filtering
4. Performs RAG (Retrieval-Augmented Generation) using OpenAI chat models
5. Supports LangChain integrations for advanced RAG operations
6. Provides metadata-based filtering capabilities

Features:
- Interactive query loop with exit options
- Enhanced retriever with ChromaDB backend
- LangChain compatibility for advanced RAG chains
- Metadata filtering for specialized movie queries
- OpenAI chat model integration for response generation
- Comprehensive error handling and logging
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.prompts import PromptTemplate

from baseline.preprocessor.chunking_service import FixedSizeChunkingStrategy
from specialization.retriever.enhanced_retriever import EnhancedRetriever
from specialization.generator.enhanced_llm import EnhancedLLM
from specialization.generator.query_parser import QueryParser
from specialization.config.config import (
    CHUNK_SIZE,
    LOG_LEVEL
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UserQueryPipeline:
    """
    Interactive user query pipeline using enhanced retrieval and RAG operations.
    Supports both direct retrieval and LangChain-based RAG chains.
    """
    
    def __init__(self, chunk_size: int = None, use_existing_db: bool = True):
        """
        Initialize the user query pipeline.
        
        Args:
            chunk_size (int): Size for text chunking
            use_existing_db (bool): Whether to use existing database or recreate
        """        
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunking_strategy = FixedSizeChunkingStrategy(chunk_size=self.chunk_size)
        
        # Initialize enhanced retriever (use existing DB by default)
        self.retriever = EnhancedRetriever(
            self.chunking_strategy, 
            fresh_db=not use_existing_db
        )
        
        # Initialize LLM
        self.llm = EnhancedLLM.chat_model()
        
        # Initialize query parser for metadata extraction
        self.query_parser = QueryParser()
        
        # Create custom RAG prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful movie expert assistant. Use the following context to answer the question about movies. 
            Guidelines:
            - Mention the all the movies relevant or partially relevant to the question.
            - Use precise and concise language.
            - If the question is not answerable with the provided context, say "No Data Found".

            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        logger.info("User Query Pipeline initialized successfully")
        logger.info(f"Using chunk size: {self.chunk_size}")

    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of retrieved documents with content and metadata
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # Create a nice format with metadata
            title = metadata.get('title', 'Unknown Movie')
            context_part = f"Movie: {title}\nContent: {content}"
            context_parts.append(context_part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def query_basic_rag(self, query: str, k: int = 5, 
                       filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform basic RAG operation with automatic metadata parsing and filtering.
        
        Args:
            query (str): User query
            k (int): Number of documents to retrieve
            filter_dict (Optional[Dict]): Manual metadata filter (overrides parsing)
            
        Returns:
            Dict: Response with answer, context, and metadata
        """        
        # Retrieve relevant documents with metadata filtering
        retrieved_docs = self.retriever.query(query, k=k, filter_dict=filter_dict)
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'context': [],
                'query': query,
                'filter_applied': filter_dict,
                'num_retrieved': 0
            }
        
        # Format context
        context = self._format_context(retrieved_docs)
        
        # Generate answer using LLM with the original question for context
        prompt = self.rag_prompt.format(context=context, question=query)
        answer = self.llm.invoke(prompt).content
        
        return {
            'answer': answer,
            'context': retrieved_docs,
            'query': query,
            'filter_applied': filter_dict,
            'num_retrieved': len(retrieved_docs)
        }

    def _display_results(self, result: Dict[str, Any], show_context: bool = False):
        """
        Display query results in a formatted way.
        
        Args:
            result (Dict): Query result
            show_context (bool): Whether to show retrieved context
        """
        print("\n" + "="*80)
        print(f"QUERY: {result['query']}")
        print("="*80)
        
        if result.get('filter_applied'):
            print(f"FILTERS APPLIED: {result['filter_applied']}")
            print("-" * 40)
        
        print(f"ANSWER:\n{result['answer']}")
        print("-" * 40)
        print(f"Retrieved {result['num_retrieved']} documents")
        
        if show_context and result.get('context'):
            print("\nRETRIEVED CONTEXT:")
            print("-" * 40)
            for i, doc in enumerate(result['context']):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Unknown').title()
                score = doc.get('score', 'N/A')  # Score is at document level, not in metadata
                # Format score to 4 decimal places if it's a number
                if isinstance(score, (int, float)):
                    score = f"{score:.3f}"
                print(f"\n[{i+1}] Movie: {title} (Score: {score})")
                print(f"Content: {doc['content'][:150]}...")
        
        print("="*80)
    
    def run_interactive_loop(self):
        """
        Run the interactive user query loop.
        """
        
        print("\n🎬 Movie RAG Query System")
        print("="*50)
        print("Ask questions about movies in the database!")
        print("Commands:")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'info' to see database information")
        print("  - 'help' for more commands")
        print("="*50)
        
        # Show database info
        collection_info = self.retriever.get_collection_info()
        print(f"Database contains {collection_info.get('count', 0)} chunks")
        print()
        
        while True:
            try:
                # Get user query
                query = input("\n🎯 Enter your movie question: ").strip()
                
                if not query:
                    continue
                    
                # Handle special commands
                if query.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                elif query.lower() == 'info':
                    print(f"\nDatabase Info: {collection_info}")
                    continue
                    
                elif query.lower() == 'help':
                    print("\n📖 Help:")
                    print("- Ask any question about movies")
                    print("- Use filters like: genre=Comedy, revenue>1000000")
                    print("- Type 'basic' or 'langchain' to choose RAG method")
                    print("- Type 'context' to toggle context display")
                    continue
                
                show_context = True
                
                parsed_query, parsed_filters = self.query_parser.parse_with_chroma_filters(query)
                result = self.run_single_query(parsed_query, filter_dict=parsed_filters, show_context=show_context)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ Error: {e}")
    
    def run_single_query(self, query: str, filter_dict: Optional[Dict[str, Any]] = None,
                        show_context: bool = True) -> Dict[str, Any]:
        """
        Run a single query for programmatic use.
        
        Args:
            query (str): User query
            filter_dict (Optional[Dict]): Metadata filter
            show_context (bool): Whether to display context
            
        Returns:
            Dict: Query result
        """
        result = self.query_basic_rag(query, k=5, filter_dict=filter_dict)
        
        if show_context:
            self._display_results(result, show_context=True)
        
        return result


def main():
    """Main function to run the user query pipeline."""
    try:
        # Fix SSL certificate issue on Windows
        if "SSL_CERT_FILE" in os.environ:
            logger.info(f"Removing problematic SSL_CERT_FILE environment variable: {os.environ['SSL_CERT_FILE']}")
            del os.environ["SSL_CERT_FILE"]
            
        # Initialize pipeline
        pipeline = UserQueryPipeline(use_existing_db=True)
        
        # Run interactive loop
        pipeline.run_interactive_loop()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")


if __name__ == "__main__":
    main()