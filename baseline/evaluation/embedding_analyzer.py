import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from retriever.retreiver import Retriever
from preprocessor.chunking_service import (
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
    SentenceBasedChunkingStrategy,
    ParagraphBasedChunkingStrategy,
    SemanticChunkingStrategy,
    MarkdownHeaderChunkingStrategy,
)
from config.config import DOCUMENT_FOLDER_PATH

from generator.llm import LLM

from evaluation.answer_verifier import labeled_data


def embed_labeled_queries(labeled_data):
    queries = [item["query"] for item in labeled_data]
    return LLM.generate_embedding(queries), queries


def plot_embeddings(embeddings_dict, method="pca", labeled_data=None):
    import itertools

    # Define distinct markers and colors
    markers = itertools.cycle(["o", "s", "^", "D", "v", "*", "P", "X"])
    colors = itertools.cycle(
        ["red", "blue", "green", "orange", "purple", "brown", "teal", "magenta"]
    )

    plt.figure(figsize=(10, 6))

    # Add labeled queries if provided
    if labeled_data:
        query_embeddings, queries = embed_labeled_queries(labeled_data)
        embeddings_dict["Labeled Queries"] = np.array(query_embeddings)

    for name, embs in embeddings_dict.items():
        n = len(embs)
        if n < 2:
            print(f"⚠️ Skipping {name}: too few embeddings ({n})")
            continue

        try:
            if method == "pca":
                reducer = PCA(n_components=2)
            elif method == "tsne":
                perplexity = min(30, max(3, n // 3))  # auto-adjust
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            else:
                raise ValueError("Unsupported method: use 'pca' or 'tsne'")

            reduced = reducer.fit_transform(embs)
            x, y = reduced[:, 0], reduced[:, 1]

            # Special formatting for labeled queries
            if name == "Labeled Queries":
                plt.scatter(
                    x,
                    y,
                    label=name,
                    marker="*",
                    color="black",
                    s=450,
                    alpha=1.0,
                    edgecolors="white",
                )
            else:
                marker = next(markers)
                color = next(colors)
                plt.scatter(
                    x, y, label=name, marker=marker, color=color, s=30, alpha=0.7
                )
        except Exception as e:
            print(f"❌ Failed to reduce {name} ({n} samples): {e}")

    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_embeddings():
    strategies = [
        FixedSizeChunkingStrategy(chunk_size=1000),
        SlidingWindowChunkingStrategy(chunk_size=1000, overlap=100),
        SentenceBasedChunkingStrategy(chunk_size=1000),
        ParagraphBasedChunkingStrategy(chunk_size=1000),
        SemanticChunkingStrategy(),
    ]

    embeddings_by_strategy = {}

    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        print(f"\n🔍 Processing Strategy: {strategy_name}")

        retriever = Retriever(strategy)
        retriever.add_document(DOCUMENT_FOLDER_PATH, is_directory=True)
        retriever.preprocess()
        embeddings = retriever.save()
        embeddings_by_strategy[strategy_name] = np.array(embeddings)
        print(f"✔️ {strategy_name}: {len(embeddings)} embeddings extracted")

    plot_embeddings(embeddings_by_strategy, method="pca", labeled_data=labeled_data)
    plot_embeddings(embeddings_by_strategy, method="tsne", labeled_data=labeled_data)

    print("\n✅ Embedding analysis complete.")
