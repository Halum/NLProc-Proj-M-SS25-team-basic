from read_documents.read_txt_files import get_text_chunks_from_directory
from text_embedding.load_model import load_transformer_model
from text_embedding.generate_embedding import generate_embedding
from analysis.visualize_embedding import visualize_using_pca, visualize_using_tsne
from analysis.cosine_similarity import compute_cosine_similarity, visualize_cosine_similarity

txt_files_directory = 'documents'

file_contents, sample_names = get_text_chunks_from_directory(txt_files_directory)

model = load_transformer_model('all-MiniLM-L6-v2')
embeddings = generate_embedding(file_contents, model)

# calculate cosine similarity and visualize it
cosine_similarity = compute_cosine_similarity(embeddings)
visualize_cosine_similarity(cosine_similarity, sample_names)


# visualize embeddings using PCA 
visualize_using_pca(embeddings, sample_names)
# visualize embeddings using t-SNE
visualize_using_tsne(embeddings, sample_names)