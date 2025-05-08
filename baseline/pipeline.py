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
        
def human_feedback(expected_answer, generated_answer):
    """
    Function to simulate human feedback on the generated answer.
    
    Args:
        expected_answer (str): The expected answer.
        generated_answer (str): The generated answer.
        
    Returns:
        bool: True if the generated answer is correct, False otherwise.
    """
    print("Is the answer correct? (y/n)")
    feedback = input().strip().lower()
    
    return feedback == 'y' 
    

def main():
    """
    Main function to execute the document processing pipeline.
    """
    chunking_stratigies = [
        FixedSizeChunkingStrategy(chunk_size=1000),
        # SlidingWindowChunkingStrategy(chunk_size=1000, overlap=100),
        # SentenceBasedChunkingStrategy(chunk_size=1000),
        # ParagraphBasedChunkingStrategy(chunk_size=1000),
        # SemanticChunkingStrategy(),
        # MarkdownHeaderChunkingStrategy()
    ]
    insight_generator = InsightGenerator()
    
    retrievers = []
    for strategy in chunking_stratigies:
        print(f"Using strategy: {strategy.__class__.__name__}")
        retriever = Retriever(strategy)
        retrievers.append(retriever)
        
    for retriever in retrievers:
        print(f"Processing documents with strategy: {retriever.chunking_strategy.__class__.__name__}")
        retriever.add_document(DOCUMENT_FOLDER_PATH, is_directory=True)
        
        chunks = retriever.preprocess()
        print(f"Number of chunks generated: {len(chunks)}")
        retriever.save()
        
        labeled_date = AnswerVerifier.get_sample_labeled_data()
        
        retrieved_chunks = retriever.load(labeled_date["query"])
        generated_answer = retriever.query(labeled_date["query"], retrieved_chunks)
        
        
        print(f"Query: {labeled_date['query']}")
        print(f"Expected answer: {labeled_date['answer']}")
        print(f"Generated answer: {generated_answer}")
        print(f"Context: {labeled_date['context']}")
        
        expected_chunk_index, expected_chunk = AnswerVerifier.find_chunk_containing_context(retrieved_chunks, labeled_date["context"])
        if expected_chunk_index != -1:
            print(f"Context found in chunk: {expected_chunk_index}")
        else:
            print("Expected chunk not found in retrieved chunks.")
            
        feedback = human_feedback(labeled_date["answer"], generated_answer)
        insight_generator.update_insight(
            chunk_strategy=retriever.chunking_strategy.__class__.__name__,
            number_of_chunks=len(chunks),
            retrieved_chunk_rank=expected_chunk_index,
            correct_answer=feedback
        )
        
        print("-"*50)
        # print(f"Retrieved chunks:")
        # print_chunks(retrieved_chunks)
        
    insight_generator.save_insight('chunking_strategy_insights.csv')
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
