from read_documents.read_txt_files import get_text_chunks_from_directory
from text_embedding.load_model import load_transformer_model
from text_embedding.generate_embedding import generate_embedding

txt_files_directory = 'documents'

file_contents, sample_names = get_text_chunks_from_directory(txt_files_directory)
print("Sample names:", sample_names)

model = load_transformer_model('all-MiniLM-L6-v2')
embeddings = generate_embedding(file_contents, model)
