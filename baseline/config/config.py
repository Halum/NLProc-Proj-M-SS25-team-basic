"""
Configuration module for the NLP processing pipeline.
This file provides centralized configuration for:
1. File paths for data, insights, and database storage
2. Model selection for embeddings and language model components
3. System-wide parameters used across different modules
4. Environment and deployment identification settings
"""

# Configuration management
DOCUMENT_FOLDER_PATH='baseline/data/raw/'
INSIGHT_FOLDER_PATH='baseline/data/insight/'
DB_INDEX_PATH='baseline/data/db/'
TEST_QUESTIONS_PATH='baseline/data/tests/test_input.json'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'declare-lab/flan-alpaca-base'
GROUP_ID = 'Team Basic'