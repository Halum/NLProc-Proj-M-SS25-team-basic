def generate_embedding(text_chunk, model):
    # Generate the embedding
    embedding = model.encode(text_chunk)
    
    return embedding