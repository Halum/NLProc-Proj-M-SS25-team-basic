#!/usr/bin/env python3
"""
Prompt templates for the specialization track.

This module contains various prompt templates used throughout the application
for different RAG (Retrieval-Augmented Generation) operations.
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate


def get_movie_rag_prompt() -> PromptTemplate:
    """
    Get the RAG prompt template for movie-related queries.
    
    Returns:
        PromptTemplate: Configured prompt template for movie RAG operations
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful movie expert assistant. Use the following context to answer the question about movies in MarkDown format. 
        Guidelines:
        - Mention the all the movies relevant or partially relevant to the question.
        - Use precise and concise language.
        - If the question is not answerable with the provided context, say "No Data Found".

        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    )


def get_query_parsing_prompt() -> ChatPromptTemplate:
    """
    Get the chat prompt template for parsing user queries and extracting metadata filters.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for query parsing operations
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that extracts filterable metadata from natural-language movie queries, and separates it from the main information need.

        Your task is to:
        1. Identify any filterable movie metadata:
            - revenue (min_revenue, max_revenue)
            - runtime (min_runtime, max_runtime)
            - title text (title_contains)
            - release date (release_date, min_release_date, max_release_date)
            - vote average/rating (min_vote_average, max_vote_average)
            - budget (min_budget, max_budget)
        2. Extract the **core question** about movies. Do not remove the user's intent or thematic language (e.g., "with firefighter", "involving fate", "about survival").
        3. Preserve the user's natural-language question to be used in semantic search or reasoning.
        
        Guidelines:
        - Revenue should be in dollars (e.g., "5 million" = 5000000)
        - Runtime should be in minutes (e.g., "over 2 hours" = min_runtime: 120)
        - Release date should be handled as:
            • "from 2020" → release_year: 2020
            • "before 2000" → max_release_year: 1999
            • "after 1990" → min_release_year: 1991
            • "in the 90s" → min_release_year: 1990, max_release_year: 1999
            • "before the 90s" → max_release_year: 1989
            • "after the 80s" → min_release_year: 1990
            • "in the 80s" → min_release_year: 1980, max_release_year: 1989
        - vote_average (range: 1.0 to 10.0):
            • "rated above or over 8" → min_vote_average: 8.0
            • "with rating below 5" → max_vote_average: 5.0
            • "highly rated", "top rated", or "high ratings" → min_vote_average: 7.0
            • "critically acclaimed", "great reviews" → min_vote_average: 8.0
        - The question should be the core information need about movies
        - Budget should be in dollars (e.g., "under 10 million" = 10000000)

        Return structured output for:
            - min_revenue (float)
            - max_revenue (float)
            - title_contains (string)
            - min_runtime (int)
            - max_runtime (int)
            - release_date (int)
            - min_release_date (int)
            - max_release_date (int)
            - min_vote_average (float)
            - max_vote_average (float)
            - min_budget (float)
            - max_budget (float)
            - question (string)
        
        Examples:
        "Movies with Tom Hanks that made over 100 million" →
        {{ 
        "min_revenue": 100000000,
        "question": "Movies with Tom Hanks" 
        }}

        "A movie with rating over 7 and has firefighter in the movie plot" →
        {{ 
        "min_vote_average": 7.0,
        "question": "A movie with firefighter in the plot" 
        }}
        
        "Suspenseful movies rated over 8 made before 1995" →
        {{ 
        "min_vote_average": 8.0,
        "max_release_date": 1994,
        "question": "Suspenseful movies" 
        }}

        "Are there movies involving destiny versus making your own choices?" →
        {{ 
        "question": "Movies involving destiny versus making your own choices" 
        }}

        "Action movies from 2020 starring Brad Pitt with revenue under 50M with high ratings" →
        {{ 
        "release_date": 2020,
        "max_revenue": 50000000,
        "min_vote_average": 7.0,
        "question": "Action movies starring Brad Pitt" 
        }}

        "Top rated movies from the 90s" →
        {{ 
        "min_vote_average": 7.0,
        "min_release_date": 1990,
        "max_release_date": 1999,
        "question": "Top rated movies" 
        }}

        "Movies before 1980 with short runtime" →
        {{ 
        "max_release_date": 1979,
        "question": "Movies with short runtime" 
        }}

        "Comedy films made with a budget under 5 million rated above 7 " →
        {{ 
        "max_budget": 5000000,
        "min_vote_average": 7.0,
        "question": "Comedy films" 
        }}

        "Romantic movies with rating between 7 and 9" →
        {{ 
        "min_vote_average": 7.0,
        "max_vote_average": 9.0,
        "question": "Romantic movies" 
        }}
        """,
            ),
            ("user", "{query}"),
        ]
    )


# For convenience, create default instances
MOVIE_RAG_PROMPT = get_movie_rag_prompt()
QUERY_PARSING_PROMPT = get_query_parsing_prompt()
