"""
Main module for text embedding and visualization.
This script processes text files, generates embeddings, and visualizes them in different ways.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

from text_embedding.load_model import load_transformer_model
from read_documents.read_txt_files import get_text_chunks_from_directory
from text_embedding.generate_embedding import generate_embedding
from analysis.visualize_embedding import (
    visualize_using_pca,
    visualize_using_tsne,
    visualize_cosine_similarity
)
from config import (
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INPUT_DIRECTORY,
    COSINE_VIS_TITLE,
    TSNE_VIS_TITLE,
    PCA_VIS_TITLE,
    LLM_MODEL
)


def process_documents():
    """Process documents and return their contents and names."""
    txt_files_directory = INPUT_DIRECTORY
    return get_text_chunks_from_directory(txt_files_directory, chunk_size=CHUNK_SIZE)


def generate_document_embeddings(file_contents):
    """Generate embeddings from file contents using the configured model."""
    model = load_transformer_model(EMBEDDING_MODEL)
    embeddings = generate_embedding(file_contents, model)
    
    del model
    
    return embeddings


def visualize_embeddings(embeddings, sample_names):
    """Visualize embeddings using different techniques."""
    # Calculate and visualize cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    visualize_cosine_similarity(similarity_matrix, sample_names, graph_title=COSINE_VIS_TITLE)
    
    # Visualize using dimensionality reduction techniques
    visualize_using_pca(embeddings, sample_names, graph_title=PCA_VIS_TITLE)
    visualize_using_tsne(embeddings, sample_names, graph_title=TSNE_VIS_TITLE)
    
def store_embeddings(embeddings):
    db = faiss.IndexFlatL2(embeddings[0].shape[0])
    db.add(np.array(embeddings))
    
    return db

def retrieve_similar_embeddings(db, query, k=5):
    """Retrieve the top k most similar embeddings from the database."""
    query_embedding = generate_document_embeddings([query])
    distances, indices = db.search(np.array(query_embedding), k=k)
    return distances, indices

def get_retrieved_chunks(indices, file_contents):
    """Retrieve the chunks corresponding to the indices."""
    retrieved_chunks = []
    for i in indices[0]:
        retrieved_chunks.append(file_contents[i])
    return retrieved_chunks

def generate_prompt(retrieved_chunks, query):
    context  = "\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:"""
    
    return prompt
    
def generate_answer(prompt):
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    inputs = tokenizer("Can yo answer this?", return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
        
    # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = 'I do not know the answer to that question.'
    
    del tokenizer
    del model
    torch.cuda.empty_cache()
    
    return answer

def main():
    """Main function to orchestrate the document processing and visualization workflow."""
    # Process documents
    file_contents, sample_names = process_documents()
    
    # Generate embeddings
    embeddings = generate_document_embeddings(file_contents)
    
    # # Visualize results
    visualize_embeddings(embeddings, sample_names)
    
    # db = store_embeddings(embeddings)
    # query = 'What are some common reasons university students in Germany don’t always eat healthily?' 
    # distances, indices = retrieve_similar_embeddings(db, query, 5)
    # retrieved_chunks = get_retrieved_chunks(indices, file_contents)
    # prompt = generate_prompt(retrieved_chunks, query)
    
    
    # answer = generate_answer(prompt)
    # print(f"Answer: {answer}")


if __name__ == "__main__":
    main()