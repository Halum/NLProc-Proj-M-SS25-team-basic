"""
Main pipeline module for the NLP processing system.
This file orchestrates the end-to-end pipeline by:
1. Coordinating components from preprocessing to answer generation
2. Providing command-line interface for system operation
3. Implementing evaluation and verification workflows
4. Supporting different chunking strategies and retrieval methods
"""

import argparse

from retriever.retreiver import Retriever
from preprocessor.chunking_service import (
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
    SentenceBasedChunkingStrategy,
    ParagraphBasedChunkingStrategy,
    SemanticChunkingStrategy,
    MarkdownHeaderChunkingStrategy
)

from config.config import DOCUMENT_FOLDER_PATH
from evaluation.answer_verifier import AnswerVerifier
from evaluation.insight_generator import InsightGenerator
from generator.generator import Generator

def print_chunks(chunks):
    """
    Print the chunks in a formatted manner.
    
    Args:
        chunks (list): List of chunks to print.
    """
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 50)

def main():
    """
    Main function to execute the document processing pipeline.
    """
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument(
        "--withindex",
        default=False,
        action="store_true",
        help="With existing vector indexes",
    )
    args = parser.parse_args()

    chunking_strategies = [
        FixedSizeChunkingStrategy(chunk_size=1000),
        SlidingWindowChunkingStrategy(chunk_size=1000, overlap=100),
        SentenceBasedChunkingStrategy(chunk_size=1000),
        ParagraphBasedChunkingStrategy(chunk_size=1000),
        SemanticChunkingStrategy(),
        MarkdownHeaderChunkingStrategy()
    ]
    insight_generator = InsightGenerator()
    
    retrievers = []
    for strategy in chunking_strategies:
        print(f"Using strategy: {strategy.__class__.__name__}")
        retriever = Retriever(strategy)
        retrievers.append(retriever)

        print(f"Processing documents with strategy: {retriever.chunking_strategy.__class__.__name__}")
        retriever.add_document(DOCUMENT_FOLDER_PATH, is_directory=True)

        chunks = retriever.preprocess()
        print(f"Number of chunks generated: {len(chunks)}")
        
        if args.withindex:
            print("Loading existing vector indexes...")
            retriever.load()
        else:
            print("Saving vector indexes...")
            retriever.save()
        
    for retriever in retrievers:
        print(f"Using retriever with strategy: {retriever.chunking_strategy.__class__.__name__}")
        
        labeled_date_list = AnswerVerifier.get_sample_labeled_data()
        
        
        for labeled_date in labeled_date_list:
            retrieved_chunks, distances  = retriever.query(labeled_date["query"])
            answering_prompt = Generator.build_answering_prompt(labeled_date["query"], retrieved_chunks)
            generated_answer = Generator.generate_answer(answering_prompt)

            print(f"Query: {labeled_date['query']}")
            print(f"Expected answer: {labeled_date['answer']}")
            print(f"Generated answer: {generated_answer}")
            print(f"Context: {labeled_date['context']}")
            
            expected_chunk_index, expected_chunk = AnswerVerifier.find_chunk_containing_context(retrieved_chunks, labeled_date["context"])
            if expected_chunk_index != -1:
                print(f"Context found in chunk: {expected_chunk_index}")
            else:
                print("Expected chunk not found in retrieved chunks.")
                
            feedback = InsightGenerator.human_feedback(labeled_date["answer"], generated_answer)
            
            insight_generator.update_insight(
                question=labeled_date["query"],
                retrieved_chunks=retrieved_chunks,
                prompt=answering_prompt,
                generated_answer=generated_answer,
                chunk_strategy=retriever.chunking_strategy.__class__.__name__,
                number_of_chunks=len(chunks),
                retrieved_chunk_rank=expected_chunk_index,
                is_correct_answer=feedback,
                similarity_scores=distances[0],
                similarity_mean=distances[0].mean()
            )
            
            print("-"*50)
            print("Retrieved chunks:")
            print_chunks(retrieved_chunks)
            
            # break
        
    insight_generator.save_insight('chunking_strategy_insights')
    print("Pipeline completed successfully.")
    # for embedding visualization
    # analyze_embeddings()
    # analyze_heatmap()


if __name__ == "__main__":
    main()
