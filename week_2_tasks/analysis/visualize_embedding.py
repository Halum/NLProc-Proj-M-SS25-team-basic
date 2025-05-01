from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_using_pca(embeddings, sample_names):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, txt in enumerate(sample_names):
        plt.scatter(pca_result[i, 0], pca_result[i, 1])
        plt.annotate(f"{sample_names[i]}", (pca_result[i, 0], pca_result[i, 1]))
    plt.title("Sentence Embeddings Visualized with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()


def visualize_using_tsne(embeddings, sample_names):
    n_samples = len(embeddings)
    perplexity = min(30, max(1, n_samples - 1)) 
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, txt in enumerate(sample_names):
        plt.scatter(tsne_result[i, 0], tsne_result[i, 1])
        plt.annotate(f"{sample_names[i]}", (tsne_result[i, 0], tsne_result[i, 1]))
    plt.title("Sentence Embeddings Visualized with t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.show()