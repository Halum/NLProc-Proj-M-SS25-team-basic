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
        template="""
        You are a helpful and precise movie assistant.

        Use **only** the information from the context below to answer the user's question about movies. Format your answer in **Markdown**.

        ---

        **Instructions**:
        - If the answer can be fully or partially answered from the context, list all relevant movies with a brief explanation.
        - If **no relevant information is found**, respond with **exactly** this text:  
        **No Data Found**
        - Do **not** change the wording, do **not** paraphrase it.
        - Do **not** include both an answer and "No Data Found".
        - Do **not** guess or use any information not present in the context.
        ---
        
        **Examples**:

        If relevant movies found:
        - **Inception:** Mentioned in context as a dream-based thriller.

        If no answer is possible:
        **No Data Found**
        
        ---

        **Context**:
        {context}

        **Question**:
        {question}

        **Answer**:
        """
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
            - release date (release_date, min_release_date, max_release_date)
            - vote average/rating (min_vote_average, max_vote_average)
            - budget (min_budget, max_budget)
        2. Create a **normalized question** that standardizes the query for better retrieval:
            - Convert to lowercase
            - Standardize movie terms (films/flicks/motion pictures → movies)
            - Remove unnecessary words (articles, question phrases)
            - Focus on core search terms
            - Keep essential movie attributes, actors, genres, themes
        
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
        - Budget should be in dollars (e.g., "under 10 million" = 10000000)
        
        Query Normalization Guidelines:
        - For the normalized_question, standardize the following:
          • Keep all movie-related terms as they are (don't automatically convert "films" to "movies")
          • Remove introductory phrases like "Can you tell me about", "Show me", "What are", "Could you suggest"
          • Remove articles and determiners (a, an, the, some) unless they're part of a title
          • Make text lowercase for consistent matching
          • CRITICAL: Preserve genre specifications exactly as stated (e.g., "Mystery genre", "Action genre")
          • IMPORTANT: Preserve question intent - keep words that indicate specific questions about events, outcomes, or relationships (e.g., "did", "was", "how", "why", "when")
          • For questions about events or outcomes (e.g., "Did X die?", "Was X in Y?"), maintain the question structure
          • Keep essential terms related to genres, actors, directors, themes, plots, and events
          • Maintain negations and qualifiers that affect meaning
          • Preserve specificity in the query - don't oversimplify and lose important context

        Return structured output for:
            - min_revenue (float)
            - max_revenue (float)
            - min_runtime (int)
            - max_runtime (int)
            - release_date (int)
            - min_release_date (int)
            - max_release_date (int)
            - min_vote_average (float)
            - max_vote_average (float)
            - min_budget (float)
            - max_budget (float)
            - normalized_question (string)
        
        Examples:
        "Movies with Tom Hanks that made over 100 million" →
        {{ 
        "min_revenue": 100000000,
        "normalized_question": "tom hanks movies" 
        }}

        "A movie with rating over 7 and has firefighter in the movie plot" →
        {{ 
        "min_vote_average": 7.0,
        "normalized_question": "firefighter movies plot" 
        }}
        
        "Suspenseful movies rated over 8 made before 1995" →
        {{ 
        "min_vote_average": 8.0,
        "max_release_date": 1994,
        "normalized_question": "suspenseful movies before 1995" 
        }}

        "Are there movies involving destiny versus making your own choices?" →
        {{ 
        "normalized_question": "movies involving destiny versus making own choices" 
        }}

        "Action movies from 2020 starring Brad Pitt with revenue under 50M with high ratings" →
        {{ 
        "release_date": 2020,
        "max_revenue": 50000000,
        "min_vote_average": 7.0,
        "normalized_question": "action movies brad pitt" 
        }}

        "Top rated movies from the 90s" →
        {{ 
        "min_vote_average": 7.0,
        "min_release_date": 1990,
        "max_release_date": 1999,
        "normalized_question": "top rated movies from 90s" 
        }}

        "Movies before 1980 with short runtime" →
        {{ 
        "max_release_date": 1979,
        "normalized_question": "short runtime movies" 
        }}

        "Comedy films made with a budget under 5 million rated above 7 " →
        {{ 
        "max_budget": 5000000,
        "min_vote_average": 7.0,
        "normalized_question": "comedy movies" 
        }}

        "Romantic movies with rating between 7 and 9" →
        {{ 
        "min_vote_average": 7.0,
        "max_vote_average": 9.0,
        "normalized_question": "romantic movies" 
        }}

        "Did Jack Dawson survive in Titanic?" →
        {{ 
        "normalized_question": "did jack dawson survive in titanic" 
        }}

        "Was Ryan Reynolds in Deadpool 2?" →
        {{ 
        "normalized_question": "was ryan reynolds in deadpool 2" 
        }}

        "How did the Joker become a villain?" →
        {{ 
        "normalized_question": "how did joker become villain" 
        }}

        "Could you suggest me some high rated films in 2000s Mystery genre?" →
        {{ 
        "min_vote_average": 7.0,
        "min_release_date": 2000,
        "max_release_date": 2009,
        "normalized_question": "high rated films 2000s mystery genre" 
        }}

        "What are some good Science Fiction films from the 80s?" →
        {{ 
        "min_vote_average": 7.0,
        "min_release_date": 1980,
        "max_release_date": 1989,
        "normalized_question": "good science fiction films 80s" 
        }}
        """,
            ),
            ("user", "{query}"),
        ]
    )


# For convenience, create default instances
MOVIE_RAG_PROMPT = get_movie_rag_prompt()
QUERY_PARSING_PROMPT = get_query_parsing_prompt()
