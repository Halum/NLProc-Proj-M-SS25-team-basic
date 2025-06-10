"""
Main module for text embedding and visualization.
This script processes text files, generates embeddings, and visualizes them in different ways.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.metrics.pairwise import cosine_similarity

from read_documents.document_reader import read_csv, chunk_text
from analysis.visualize_embedding import (
    visualize_using_pca,
    visualize_using_tsne,
    visualize_cosine_similarity
)
from utils.llm import invoke_llm
from utils.prompts import generate_answering_prompt, generate_answer_evaluation_prompt
from utils.retriever import store_embeddings, retrieve_similar_embeddings, get_retrieved_chunks, generate_document_embeddings
from utils.evaluation import rank_chunks_by_ratings, run_with_timer
from config import (
    CHUNK_SIZE,
    COSINE_VIS_TITLE,
    TSNE_VIS_TITLE,
    PCA_VIS_TITLE,
    INPUT_FILE_PATH,
    TOP_K
)

def visualize_topic_embeddings(topic_contents, labels):
    """Visualize embeddings using different techniques."""
    embeddings = generate_document_embeddings(topic_contents)
    # Calculate and visualize cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    visualize_cosine_similarity(similarity_matrix, labels, graph_title=COSINE_VIS_TITLE)
    
    # Visualize using dimensionality reduction techniques
    visualize_using_pca(embeddings, labels, graph_title=PCA_VIS_TITLE)
    visualize_using_tsne(embeddings, labels, graph_title=TSNE_VIS_TITLE)  

def get_qna_tuples(df):
    question_columns = ['q1', 'q2', 'q3', 'q4', 'q5']
    answer_columns = ['a1', 'a2', 'a3', 'a4', 'a5']
    
    qna_tuples = []
    for _, row in df.iterrows():
        row_qna = []
        for q_col, a_col in zip(question_columns, answer_columns):
            row_qna.append((row[q_col], row[a_col]))
        qna_tuples.append(row_qna)
        
    return qna_tuples

def print_chunks(file_chunks):
    """Print the chunks of text."""
    for i, chunk in enumerate(file_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 50)

def run_and_evaluate_answer(db, file_chunks, qna_tuples):
    """Run the LLM and evaluate the answers."""
    all_execution_times = []
    all_chunk_ratings = []
    all_evaluation_Scores = []
    
    for topic_wise_qna in qna_tuples:
        topic_chunk_ratings = {}
        topic_execution_times = []
        topic_answer_scores = []
        
        
        for q, a in topic_wise_qna:
            print(f"Question: {q}")
            distances, indices = retrieve_similar_embeddings(db, q, k=TOP_K)
            retrieved_chunks = get_retrieved_chunks(indices, file_chunks)
            answering_prompt = generate_answering_prompt(retrieved_chunks, q)
            
            print_chunks(retrieved_chunks)
            
            rank_chunks_by_ratings(retrieved_chunks, topic_chunk_ratings)
            
            execution_time, answer = run_with_timer(invoke_llm, answering_prompt)
            topic_execution_times.append(execution_time)
            # print(f"LLM execution time: {execution_time:.2f} seconds")
            # print(f"Question: {q}\n")
            # print(f"AI Answer: {answer}\n")
            # print(f"Expected Answer: {a}\n")
            feedback_prompt = generate_answer_evaluation_prompt(q, answer, a)
            accuracy_score = invoke_llm(feedback_prompt)
            topic_answer_scores.append(accuracy_score)
            # print(f"Accuracy Score: {accuracy_score}\n")
            # print(f"Explaination: {explaination}\n")
            # print("-" * 50)
        
        all_chunk_ratings.append(list(topic_chunk_ratings.values()))
        all_execution_times.append(topic_execution_times)
        all_evaluation_Scores.append(topic_answer_scores)
        
    return all_execution_times, all_chunk_ratings, all_evaluation_Scores

def main():
    """Main function to orchestrate the document processing and visualization workflow."""
    # Process documents
    df = read_csv(INPUT_FILE_PATH)
    file_contents = df['text']
    file_names = df['name']
    file_chunks = [chunk for content in file_contents for chunk in chunk_text(content, CHUNK_SIZE)]
    
    # Visualize results
    visualize_topic_embeddings(file_contents, file_names)
 
    
    # Store embeddings in a database, this needs to be a single dimensional array
    embeddings = generate_document_embeddings(file_chunks)
    db = store_embeddings(embeddings)
    
    qna_tuples = get_qna_tuples(df)
    
    total_execution_time, (execution_times, chunk_ratings, evaluation_Scores) = run_with_timer(run_and_evaluate_answer, db, file_chunks, qna_tuples)
    
    
    print(f"Total execution time: {total_execution_time:.2f} seconds")    
    print("Execution times for each topic:", execution_times)
    print("Chunk ratings for each topic:", chunk_ratings)
    print("Evaluation scores for each topic:", evaluation_Scores)


if __name__ == "__main__":
    main()