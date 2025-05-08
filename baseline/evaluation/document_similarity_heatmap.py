import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from config.config import DOCUMENT_FOLDER_PATH
from preprocessor.document_reader import DocumentReader
from generator.llm import LLM

def get_document_embeddings(doc_folder_path):
    file_names = []
    texts = []

    for file in os.listdir(doc_folder_path):
        path = os.path.join(doc_folder_path, file)
        if os.path.isfile(path):
            try:
                content = DocumentReader.read_document(path)
                texts.append(content)
                file_names.append(file)
            except Exception as e:
                print(f"❌ Could not read {file}: {e}")

    embeddings = LLM.generate_embedding(texts)
    return file_names, np.array(embeddings)

def plot_document_similarity(file_names, embeddings):
    similarity_matrix = cosine_similarity(embeddings)

    plt.figure(figsize=(10, 7))
    sns.heatmap(similarity_matrix, xticklabels=file_names, yticklabels=file_names,
                cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("Cosine Similarity Between Text Documents")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_heatmap():
    file_names, embeddings = get_document_embeddings(DOCUMENT_FOLDER_PATH)
    plot_document_similarity(file_names, embeddings)
    print("\n✅ Document-level similarity heatmap complete.")


