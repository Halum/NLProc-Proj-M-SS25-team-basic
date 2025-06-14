"""
Configuration module for the NLP processing pipeline.
This file provides centralized configuration for:
1. File paths for data, insights, and database storage
2. Model selection for embeddings and language model components
3. System-wide parameters used across different modules
4. Environment and deployment identification settings
"""

# Configuration management
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_FOLDER_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw')
DOCUMENT_FOLDER_PATH = os.path.abspath(DOCUMENT_FOLDER_PATH)
INSIGHT_FOLDER_PATH='baseline/data/insight/'
LOG_FILE_NAME='chunking_strategy_insights'
DB_INDEX_PATH='baseline/data/db/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_QUESTIONS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "tests", "test_input.json"))
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# LLM_MODEL = 'declare-lab/flan-alpaca-base'
LLM_MODEL = 'google/flan-t5-large'
GROUP_ID = 'Team Basic'
VECTOR_STORE_TYPE = 'faiss'