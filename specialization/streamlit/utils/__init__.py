"""
RAG Evaluation Dashboard utilities.

This package provides utilities for transforming and analyzing RAG system
evaluation data for visualization in the Streamlit dashboard.
"""

from .basic_metrics import (
    calculate_dynamic_chart_height,
    extract_nested_score,
    prepare_correctness_data,
    prepare_correctness_by_groups_data,
    prepare_similarity_distribution_data,
    prepare_bert_score_data,
    prepare_rouge_score_data,
    calculate_overall_metrics
)

from .context_metrics import (
    analyze_context_retrieval,
    prepare_gold_context_presence_data
)

from .historical_metrics import (
    load_all_insight_files,
    calculate_average_metrics,
    prepare_historical_metrics,
    extract_score_distributions
)

__all__ = [
    'calculate_dynamic_chart_height',
    'extract_nested_score',
    'prepare_correctness_data',
    'prepare_correctness_by_groups_data',
    'prepare_similarity_distribution_data',
    'prepare_bert_score_data',
    'prepare_rouge_score_data',
    'calculate_overall_metrics',
    'analyze_context_retrieval',
    'prepare_gold_context_presence_data',
    'load_all_insight_files',
    'calculate_average_metrics',
    'prepare_historical_metrics',
    'extract_score_distributions'
]
