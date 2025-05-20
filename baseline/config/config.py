"""
Configuration file for the NLP processing pipeline.
Contains paths, model names and other configuration parameters
used throughout the application.
"""

# Configuration management
DOCUMENT_FOLDER_PATH='baseline/data/raw/'
INSIGHT_FOLDER_PATH='baseline/data/insight/'
DB_INDEX_PATH='baseline/data/db/'
TEST_QUESTIONS_PATH='baseline/data/tests/test_questions.json'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'declare-lab/flan-alpaca-base'
GROUP_ID = 'Team Basic'