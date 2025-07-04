"""
Specialized configuration module.

This module provides configuration settings that extend and override
the baseline configuration for specialized NLP processing tasks.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

RAW_DOCUMENT_DIR_PATH = "specialization/data/raw/"
PROCESSED_DOCUMENT_DIR_PATH = "specialization/data/processed/"

# Processing pipeline configurations
DATA_SAMPLE_SIZE = 1000  # Number of rows to sample from each CSV file for processing
RAW_DATA_FILES = [
    "movies_metadata.csv",
    "credits.csv",
    "keywords.csv",
]
TARGET_GENRES = ["Family", "Mystery", "Western"]
PROCESSED_DOCUMENT_NAME = "processed_movies_data_sample.json"

EXCLUDED_COLUMNS = [
    "crew",
    "belongs_to_collection",
    "homepage",
    "imdb_id",
    "original_language",
    "poster_path",
    "video",
]

FLATTEN_COLUMNS = [
    "cast",
    "spoken_languages",
    "production_countries",
    "keywords",
    "production_companies",
    "genres",
]

# Embedding pipeline configurations
DATA_COLUMNS_TO_KEEP = [
    "overview",
    "title",
    "revenue",
    "cast",
    "runtime",
    "release_date",
    "vote_average",
    "genres",
    "keywords",
    "production_companies",
    "budget",
    "spoken_languages",
    "production_countries"
]
DATA_COLUMNS_TYPE_MAPPING = (
    [  # accepted types : int | float | str | bool | year (special)
        {"column": "revenue", "type": "float"},
        {"column": "runtime", "type": "int"},
        {"column": "release_date", "type": "year"},
        {"column": "vote_average", "type": "float"},
        {"column": "budget", "type": "float"},
    ]
)
METADATA_COLUMNS = ["title", "revenue", "runtime", "release_date", "vote_average", "budget"]
ADD_TO_CHUNKING_COLUMN = [
    {"column": "cast", "prefix": "Starring with "},
    {"column": "genres", "prefix": "The movie belongs to genres like "},
    {"column": "keywords", "prefix": "The movie has keywords like "},
    {"column": "production_companies", "prefix": "Produced by "},
    {"column": "spoken_languages", "prefix": "The movie is available in languages like "},
    {"column": "production_countries", "prefix": "Produced in countries like "},
]
CHUNKING_COLUMN = "overview"

CHUNK_SIZE = 1000
VECTOR_STORE_TYPE = "chromadb"  # Using ChromaDB for this pipeline
VECTOR_COLLECTION_NAME = "movie_embeddings"
VECTOR_PERSIST_DIRECTORY = "specialization/data/db/chroma_db"

# Batch processing settings
EMBEDDING_BATCH_SIZE = 1500  # Max batch size for ChromaDB operations

DB_INDEX_PATH = "specialization/data/db/"
KG_OUTPUT_PATH = "specialization/data/knowledge_graphs/"
EMBEDDINGS_OUTPUT_PATH = "specialization/data/embeddings/"
SQLITE_OUTPUT_PATH = "specialization/data/sqlite/"
TEST_QUESTIONS_PATH = "specialization/data/tests/test_input.json"

# Specialized model configurations - OpenAI only
# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's embedding model
OPENAI_CHAT_MODEL = "gpt-4.1-nano"  # OpenAI's chat model

# Knowledge Graph settings
KG_EXTRACTION_MODEL = "spacy"  # or 'transformers'
ENTITY_LINKING_THRESHOLD = 0.8
RELATION_CONFIDENCE_THRESHOLD = 0.7

# Hybrid retrieval settings
VECTOR_WEIGHT = 0.7
GRAPH_WEIGHT = 0.3
TOP_K_DOCUMENTS = 10
TOP_K_ENTITIES = 5

# Database settings
SQLITE_DB_NAME = "specialization_documents.db"
VECTOR_STORE_TYPE = "faiss"  # or 'chromadb'

# Evaluation settings
EVALUATION_METRICS = ["precision", "recall", "f1", "bleu", "rouge"]
GOLD_INPUT_PATH = "specialization/data/tests/gold_input_v2.json"
# Base path for insights - actual files will be saved with timestamps (format: evaluation_insights_YYYYMMDD_HHMMSS.json)
EVALUATION_INSIGHTS_PATH = str(Path(__file__).parent.parent / "data" / "insight" / "evaluation_insights.json")
APPEND_INSIGHTS = (
    False  # When False, replaces existing insights file; when True, appends to it
)

# Cache settings for visualizations
CACHE_DIR_PATH = "specialization/data/processed/cache/"

# Logging
LOG_LEVEL = "INFO"
ENABLE_DETAILED_LOGGING = True
