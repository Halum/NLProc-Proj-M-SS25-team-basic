import os

def get_text_chunks_from_directory(directory, chunk_size=200):
    file_chunks = []
    sample_names = []
    
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        
        if filename.endswith('.txt') and os.path.isfile(full_path):
            with open(full_path, 'r', encoding='utf-8') as file:
                text = file.read()
        
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            file_chunks.append(chunks)
            sample_names.append(filename)
    
    return file_chunks, sample_names
