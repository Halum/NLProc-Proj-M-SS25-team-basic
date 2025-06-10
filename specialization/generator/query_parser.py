#!/usr/bin/env python3
"""
Query Parser Module for User Query Pipeline

This module provides intelligent parsing of user queries to extract:
1. Filterable metadata (revenue, cast, title)
2. Clean question text for semantic search
3. Structured filters for vector store metadata filtering

The parser uses LangChain tools and OpenAI's structured output to identify
movie-specific metadata filters from natural language queries.
"""

import logging
from typing import Dict, Any, Optional, TypedDict
from pydantic import BaseModel

from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from specialization.generator.enhanced_llm import EnhancedLLM

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FiltersInput(TypedDict, total=False):
    """Type definition for extracted metadata filters."""
    min_revenue: Optional[float]
    max_revenue: Optional[float] 
    title_contains: Optional[str]
    question: str


class ParsedQuery(BaseModel):
    """Structured representation of a parsed user query."""
    filters: Dict[str, Any]
    question: str
    original_query: str


@tool
def extract_metadata_filters(
    min_revenue: Optional[float] = None,
    max_revenue: Optional[float] = None,
    title_contains: Optional[str] = None,
    question: str = "",
) -> FiltersInput:
    """Extract movie metadata filters and question from user query.
    
    Args:
        min_revenue: Minimum revenue filter (e.g., 1000000 for $1M)
        max_revenue: Maximum revenue filter 
        title_contains: Text that should be in the movie title
        question: The core question about movies after removing filter criteria
        
    Returns:
        FiltersInput: Extracted filters and clean question
    """
    return {
        "min_revenue": min_revenue,
        "max_revenue": max_revenue,
        "title_contains": title_contains,
        "question": question,
    }


class QueryParser:
    """
    Intelligent query parser that extracts metadata filters from natural language.
    
    Uses OpenAI's function calling capabilities to identify filterable movie metadata
    like revenue, and titles from user queries while preserving the core question.
    """

    def __init__(self):
        """Initialize the query parser with OpenAI chat model."""
            
        # Initialize LLM with tool binding
        llm = EnhancedLLM.chat_model()
        self.llm = llm.bind_tools([extract_metadata_filters])

        # Create prompt template for query parsing
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that extracts filterable metadata from movie queries.

            Your task is to:
            1. Identify any filterable movie metadata (revenue, title)
            2. Extract the core question about movies
            3. Preserve the user's intent while separating filters from the question

            Guidelines:
            - Revenue should be in dollars (e.g., "5 million" = 5000000)
            - Title contains should be specific title text mentioned
            - The question should be the core information need about movies

            Examples:
            "Movies with Tom Hanks that made over 100 million" → 
            - min_revenue: 100000000
            - question: "Movies"

            "Action movies from 2020 starring Brad Pitt with revenue under 50M" →
            - question: "Action movies from 2020"
            - max_revenue: 50000000
            - question: "Action movies from 2020"
            """),
            ("user", "{query}")
        ])
        
        # Build the parsing chain
        self.chain: Runnable = self.prompt | self.llm
        
        logger.info("Query parser initialized successfully")
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a user query to extract metadata filters and clean question.
        
        Args:
            query (str): The original user query
            
        Returns:
            ParsedQuery: Structured parsing results with filters and question
        """
        try:
            logger.info(f"Parsing query: {query[:100]}...")
            
            # Run the parsing chain
            output = self.chain.invoke({"query": query})
            
            # Extract tool results
            if output.tool_calls and len(output.tool_calls) > 0:
                tool_result = output.tool_calls[0]['args']
                
                # Create filter dictionary (excluding None values and question)
                filters = {}
                for key, value in tool_result.items():
                    if key != 'question' and value is not None:
                        filters[key] = value
                
                # Get the cleaned question
                question = tool_result.get('question', query)
                if not question.strip():
                    question = query
                
                logger.info(f"Extracted filters: {filters}")
                logger.info(f"Cleaned question: {question}")
                
                return ParsedQuery(
                    filters=filters,
                    question=question,
                    original_query=query
                )
            else:
                # No tool calls made, return original query as question
                logger.info("No metadata filters detected")
                return ParsedQuery(
                    filters={},
                    question=query,
                    original_query=query
                )
                
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            # Return original query as fallback
            return ParsedQuery(
                filters={},
                question=query,
                original_query=query
            )
    
    def _convert_filters_to_chroma_format(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parsed filters to ChromaDB-compatible format.
        
        ChromaDB supports different filter operators:
        - String fields: Use direct matching or contains
        - Numeric fields: Use comparison operators like $gte, $lte
        
        Args:
            filters (Dict[str, Any]): Parsed filters from query
            
        Returns:
            Dict[str, Any]: ChromaDB-compatible filter dictionary
        """
        chroma_filters = {}
        
        for key, value in filters.items():
            if key == 'title_contains' and value:
                # Title filtering - Use exact equality for ChromaDB
                # Convert to lowercase to match metadata preprocessing
                chroma_filters['title'] = {"$eq": value.lower()}
                
            elif key == 'min_revenue' and value is not None:
                # Revenue filtering - ChromaDB supports comparison operators
                if 'revenue' in chroma_filters:
                    # Combine with existing revenue filter
                    if isinstance(chroma_filters['revenue'], dict):
                        chroma_filters['revenue']['$gte'] = float(value)
                    else:
                        chroma_filters['revenue'] = {"$gte": float(value)}
                else:
                    chroma_filters['revenue'] = {"$gte": float(value)}
                    
            elif key == 'max_revenue' and value is not None:
                # Max revenue filtering
                if 'revenue' in chroma_filters:
                    # Combine with existing revenue filter
                    if isinstance(chroma_filters['revenue'], dict):
                        chroma_filters['revenue']['$lte'] = float(value)
                    else:
                        chroma_filters['revenue'] = {"$lte": float(value)}
                else:
                    chroma_filters['revenue'] = {"$lte": float(value)}
        
        # If we have multiple top-level conditions, wrap them in $and for ChromaDB
        if len(chroma_filters) > 1:
            return {"$and": [{key: value} for key, value in chroma_filters.items()]}
        else:
            return chroma_filters
    
    def parse_with_chroma_filters(self, query: str) -> tuple[str, Dict[str, Any]]:
        """
        Parse query and return both cleaned question and ChromaDB-compatible filters.
        
        Args:
            query (str): Original user query
            
        Returns:
            tuple[str, Dict[str, Any]]: (cleaned_question, chroma_filters)
        """
        parsed = self.parse(query)
        logger.info(f"Parsed query: {parsed}")
        chroma_filters = self._convert_filters_to_chroma_format(parsed.filters)
        logger.info(f"ChromaDB filters: {chroma_filters}")
        
        return parsed.question, chroma_filters


# Example usage and testing
if __name__ == "__main__":
    # Test the parser
    parser = QueryParser()
    
    test_queries = [
        "Movies with Tom Hanks that made over 100 million dollars",
        "Action movies from 2020 with revenue under 50 million",
        "Films starring Brad Pitt and Leonardo DiCaprio",
        "What are the best comedy movies?",
        "Movies with titles containing 'Dark' that made more than 200M",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        parsed = parser.parse(query)
        print(f"Filters: {parsed.filters}")
        print(f"Question: {parsed.question}")
        
        question, chroma_filters = parser.parse_with_chroma_filters(query)
        print(f"ChromaDB filters: {chroma_filters}")
