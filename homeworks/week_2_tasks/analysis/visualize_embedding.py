from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_using_pca(embeddings, sample_names, graph_title="PCA Visualization"):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, txt in enumerate(sample_names):
        plt.scatter(pca_result[i, 0], pca_result[i, 1])
        plt.annotate(f"{sample_names[i]}", (pca_result[i, 0], pca_result[i, 1]))
    plt.title(graph_title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()


def visualize_using_tsne(embeddings, sample_names, graph_title="t-SNE Visualization"):
    n_samples = len(embeddings)
    perplexity = min(30, max(1, n_samples - 1)) 
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, txt in enumerate(sample_names):
        plt.scatter(tsne_result[i, 0], tsne_result[i, 1])
        plt.annotate(f"{sample_names[i]}", (tsne_result[i, 0], tsne_result[i, 1]))
    plt.title(graph_title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.show()
    
def visualize_cosine_similarity(similarity_matrix, sample_names, graph_title='Cosine Visualization'):
    df = pd.DataFrame(similarity_matrix, index=sample_names, columns=sample_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap='coolwarm')
    plt.title(graph_title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()