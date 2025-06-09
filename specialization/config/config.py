"""
Specialized configuration module.

This module provides configuration settings that extend and override
the baseline configuration for specialized NLP processing tasks.
"""
# Document loader configs
RAW_DOCUMENT_DIR_PATH = 'specialization/data/raw/'

PROCESSED_DOCUMENT_DIR_PATH = 'specialization/data/processed/'
RAW_DATA_FILES = [
    'movies_metadata.csv',
    'credits.csv',
    'keywords.csv',
]
PROCESSED_DOCUMENT_NAME = 'processed_movies_data.json'
TARGET_GENRES = ['Family', 'Mystery', 'Western']

# Data processing configurations
EXCLUDED_COLUMNS = [
    'crew',
    'belongs_to_collection', 
    'homepage',
    'imdb_id',
    'original_language',
    'poster_path',
    'video'
]

FLATTEN_COLUMNS = [
    'cast',
    'spoken_languages',
    'production_countries',
    'keywords',
    'production_companies',
    'genres',
]
DB_INDEX_PATH = 'specialization/data/db/'
KG_OUTPUT_PATH = 'specialization/data/knowledge_graphs/'
EMBEDDINGS_OUTPUT_PATH = 'specialization/data/embeddings/'
SQLITE_OUTPUT_PATH = 'specialization/data/sqlite/'
TEST_QUESTIONS_PATH = 'specialization/data/tests/test_input.json'

# Specialized model configurations
ENHANCED_EMBEDDING_MODEL = 'all-mpnet-base-v2'  # Better for semantic similarity
GRAPH_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L12-v2'

# Knowledge Graph settings
KG_EXTRACTION_MODEL = 'spacy'  # or 'transformers'
ENTITY_LINKING_THRESHOLD = 0.8
RELATION_CONFIDENCE_THRESHOLD = 0.7

# Hybrid retrieval settings
VECTOR_WEIGHT = 0.7
GRAPH_WEIGHT = 0.3
TOP_K_DOCUMENTS = 10
TOP_K_ENTITIES = 5

# Database settings
SQLITE_DB_NAME = 'specialization_documents.db'
VECTOR_STORE_TYPE = 'faiss'  # or 'chromadb'

# Evaluation settings
EVALUATION_METRICS = ['precision', 'recall', 'f1', 'bleu', 'rouge']

# Logging
LOG_LEVEL = 'INFO'
ENABLE_DETAILED_LOGGING = True
