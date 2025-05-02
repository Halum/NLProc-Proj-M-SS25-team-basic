# Textual explanation of embeddings and retrieval

## Index
- [Embeddings](#embeddings)
- [Retrieval](#retrieval)
- [The Workflow of Query Retrieval](#the-workflow-of-query-retrieval)
  - [Explanation of the Workflow](#explanation-of-the-workflow)
- [How Vector Search Works](#how-vector-search-works)

## Embeddings

Embeddings are numerical representations of data—typically text—that capture the semantic meaning of words, phrases, or entire documents in a way that computers can process. Instead of relying on exact word matching, embeddings convert text into high-dimensional vectors (lists of numbers), where similar meanings are positioned close together in this vector space. For example, the words "king" and "queen" will have embeddings that are closer together than "king" and "banana." These vector representations are generated using models like Word2Vec, GloVe, or more advanced ones like BERT or OpenAI's embedding models. The key advantage of embeddings is their ability to represent similarity based on context and meaning rather than just exact text matches.

## Retrieval

Retrieval refers to the process of finding the most relevant pieces of information from a large collection based on a query. In embedding-based retrieval systems, both the query and the documents are converted into embeddings. The system then calculates the similarity (usually using cosine similarity or dot product) between the query vector and document vectors to find the closest matches. This method, called vector search, is at the core of Retrieval-Augmented Generation (RAG) systems, where relevant documents are first retrieved and then passed to a language model to generate an informed and context-aware response. This combination makes it possible to build AI systems that can provide accurate answers using up-to-date or domain-specific information not seen during model training.

## The Workflow of Query Retrieval

```sql
+----------------+        +----------------+        +----------------+        +----------------+
|                |        |                |        |                |        |                |
|  User Query    +------->+  Query Encoder +------->+  Similarity     +------->+  Retrieved     |
|  (e.g., "What  |        |  (Embeddings)  |        |  Search (e.g.,  |        |  Documents     |
|  is quantum    |        |                |        |  cosine sim.)  |        |  or Chunks     |
|  entanglement?")|       |                |        |                |        |                |
+----------------+        +----------------+        +----------------+        +----------------+
                                                                                      |
                                                                                      v
                                                                              +----------------+
                                                                              |                |
                                                                              |  Large Language|
                                                                              |  Model (LLM)   |
                                                                              |  (e.g., GPT)   |
                                                                              |                |
                                                                              +----------------+
                                                                                      |
                                                                                      v
                                                                              +----------------+
                                                                              |                |
                                                                              |  Final Answer  |
                                                                              |  Generation    |
                                                                              |                |
                                                                              +----------------+

```

### Explanation of the Workflow:

**User Query:** The process begins when a user inputs a question or prompt, such as "What is quantum entanglement?"


**Query Encoder:** This query is transformed into a numerical vector (embedding) that captures its semantic meaning using a pre-trained embedding model.


**Similarity Search:** The system compares the query embedding against a database of pre-encoded document embeddings to find the most relevant pieces of information. This is typically done using similarity metrics like cosine similarity.


**Retrieved Documents:** The top-matching documents or text chunks are retrieved based on their similarity to the query.


**Large Language Model (LLM):** These retrieved documents are then provided as context to a large language model, which uses them to generate a comprehensive and accurate response to the original query.


**Final Answer Generation:** The LLM outputs the final answer, which is presented to the user.

## How Vector Search Works

Vector search is a method for finding information based on semantic similarity rather than exact keyword matching. It starts by converting all pieces of text (such as documents, sentences, or user queries) into embeddings—dense numerical vectors that represent meaning. These vectors exist in a high-dimensional space (typically 100s or 1000s of dimensions). The closer two vectors are in this space, the more similar their corresponding texts are in meaning.

When you submit a query, it is converted into a vector using the same embedding model used for the documents. The system then compares this query vector to all document vectors in the database to find the ones that are closest. This comparison is done using similarity metrics—most commonly cosine similarity (which measures the angle between two vectors) or Euclidean distance. The top results (i.e., vectors that are closest to the query vector) are returned as the most relevant documents.