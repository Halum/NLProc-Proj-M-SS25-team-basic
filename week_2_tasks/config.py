from tkinter import TOP


CHUNK_SIZE = 200
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INPUT_FILE_PATH = 'data/sample_texts.csv'
COSINE_VIS_TITLE = 'Cosine Similarity Between Text Samples'
TSNE_VIS_TITLE = 'Sentence Embeddings Visualized with t-SNE'
PCA_VIS_TITLE = 'Sentence Embeddings Visualized with PCA'
# this model worked but the other one has bigger parameters
# LLM_MODEL = 'google/flan-t5-small'
# this model is not working in Mac M1 and giving segmentation fault error
# LLM_MODEL = 'google/flan-t5-base'
LLM_MODEL = 'declare-lab/flan-alpaca-base'
TOP_K = 5