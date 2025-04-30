from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_cosine_similarity(embeddings):
    return cosine_similarity(embeddings)

def visualize_cosine_similarity(similarity_matrix, sample_names):
    df = pd.DataFrame(similarity_matrix, index=sample_names, columns=sample_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap='coolwarm')
    plt.title("Cosine Similarity Between Sentences")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()