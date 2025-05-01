from sklearn.metrics.pairwise import cosine_similarity
from text_embedding.load_model import load_transformer_model

from read_documents.read_txt_files import get_text_chunks_from_directory
from text_embedding.generate_embedding import generate_embedding
from analysis.visualize_embedding import visualize_using_pca, visualize_using_tsne, visualize_cosine_similarity

from config import CHUNK_SIZE, EMBEDDING_MODEL, INPUT_DIRECTORY, COSINE_VIS_TITLE, TSNE_VIS_TITLE, PCA_VIS_TITLE

txt_files_directory = INPUT_DIRECTORY

file_contents, sample_names = get_text_chunks_from_directory(txt_files_directory, chunk_size=CHUNK_SIZE)

model = load_transformer_model(EMBEDDING_MODEL)
embeddings = generate_embedding(file_contents, model)

# calculate cosine similarity and visualize it
cosine_similarity = cosine_similarity(embeddings)
visualize_cosine_similarity(cosine_similarity, sample_names, graph_title=COSINE_VIS_TITLE)


# visualize embeddings using PCA 
visualize_using_pca(embeddings, sample_names, graph_title=PCA_VIS_TITLE)
# visualize embeddings using t-SNE
visualize_using_tsne(embeddings, sample_names, graph_title=TSNE_VIS_TITLE)