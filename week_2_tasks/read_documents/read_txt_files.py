import os

def get_text_chunks_from_directory(directory):
    file_chunks = []
    sample_names = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if filename.endswith('.txt') and os.path.isfile(full_path):
            with open(full_path, 'r', encoding='utf-8') as file:
                text = file.read()
            chunks = [text[i:i+200] for i in range(0, len(text), 200)]
            file_chunks.append(chunks)
            sample_names.append(filename)
    
    return file_chunks, sample_names
