# Homework Week 3 Observation

## Test Environment Description
To explore the behavior and performance of a Retrieval-Augmented Generation (RAG) system, the following test setup was used:

-  **Model:** [declare-lab/flan-alpaca-base]

-  **Use Case:** Question-answering over small domain-specific texts

-  **Dataset:**

   -  5 topic-specific sample texts (each on a different subject)

   -  Each text was paired with 5 semantically similar but phrased-differently questions and their corresponding reference answers

-  **Retrieval Method:**

   -  Texts chunked into smaller passages (paragraph or sentence-level)

   -  Embedded and indexed using FAISS

   -  Top-k (k=5) most similar chunks retrieved per question

- **LLM Task:** Generate an answer given a question and top-k retrieved chunks

-  **Evaluation Strategy:** Human feedback was used to evaluate the generated answers.

-  **Metrics Recorded:**

   -  Chunking strategy used

   -  Number of chunks generated

   -  Retrieved chunk rank

   -  Correctness of the retrieved chunk

   -  Similarity scores for the top-k retrieved chunks

## Observations

### Chunking Strategy Comparison Table

| Chunking Strategy | Total Chunks | Correct Answers | Incorrect Answers | Accuracy | Avg Similarity Score (Correct) | Avg Similarity Score (Incorrect) |
|-------------------|--------------|-----------------|-------------------|----------|--------------------------------|----------------------------------|
| Fixed Size        | 188          | 2               | 3                 | 40%      | 0.94                           | 0.70                             |
| Sliding Window    | 267          | 1               | 4                 | 20%      | 0.60                           | 0.82                             |
| Sentence Based    | 206          | 3               | 2                 | 60%      | 0.84                           | 0.79                             |
| Paragraph Based   | 121          | 3               | 2                 | 60%      | 0.67                           | 1.09                             |
| Semantic          | 83           | 0               | 5                 | 0%       | N/A                            | 1.04                             |

*Note: Average similarity scores are calculated from the mean of the first value in the similarity scores array for each entry.*

### Chunking Strategy Performance
- **FixedSizeChunkingStrategy:** Generated 188 chunks. It showed mixed performance, with some correct answers at higher ranks but also many incorrect answers.
- **SlidingWindowChunkingStrategy:** Generated 267 chunks. It retrieved correct answers occasionally but had a higher number of incorrect answers compared to other strategies.
- **SentenceBasedChunkingStrategy:** Generated 206 chunks. It performed relatively well, with several correct answers at higher ranks.
- **ParagraphBasedChunkingStrategy:** Generated 121 chunks. It had a balanced performance, with correct answers often retrieved at lower ranks.
- **SemanticChunkingStrategy:** Generated 83 chunks. It retrieved fewer correct answers overall, possibly due to the smaller number of chunks.

### Similarity Scores
- Higher similarity scores were generally associated with correct answers, but there were exceptions where high scores did not lead to correct answers.
- SemanticChunkingStrategy had the highest similarity scores but did not consistently yield correct answers.

### Impact of Chunking Granularity
- Strategies with finer granularity (e.g., SentenceBased) tended to retrieve more relevant chunks compared to coarser strategies (e.g., ParagraphBased).

### Human Feedback
- Human evaluation highlighted discrepancies between similarity scores and actual relevance, suggesting that similarity alone is not a reliable metric for correctness.

### Retriever Performance on Expected Chunks
- The retriever's success varies significantly across chunking strategies, as evidenced by the "Retrieved Chunk Ranks" data
- Only **40%** of queries across all strategies resulted in the expected chunk being retrieved in the top 3 positions (ranks 0, 1, or 2)
- In **60%** of cases (15 out of 25 total evaluations), the expected chunk containing the correct context was either retrieved at a lower rank or not retrieved at all (-1)
- The most successful strategies were:
  - **Sentence Based Chunking:** Retrieved the expected chunk in top positions for 2 out of 5 queries (40%)
  - **Fixed Size Chunking:** Retrieved the expected chunk in a top position for 1 out of 5 queries (20%)
  - **Paragraph Based Chunking:** Retrieved the expected chunk in a top position for 1 out of 5 queries (20%)
- **Key Finding:** There is a significant gap between similarity score rankings and the presence of expected chunks, suggesting that current embedding and similarity metrics may not always prioritize the most relevant context

## Possible Improvements

- **Chunking Strategies**
  - Experiment with hybrid strategies combining semantic and sentence-based chunking to balance granularity and relevance.
  - Optimize the number of chunks to avoid redundancy while maintaining coverage.

- **Evaluation Metrics**
  - Incorporate additional metrics, such as precision and recall, to better assess retrieval performance.
  - Use human feedback to fine-tune similarity thresholds for determining relevance.

- **Pipeline Enhancements**
  - Implement a feedback loop to adjust retrieval strategies based on human evaluation results.
  - Visualize similarity scores and retrieval performance to identify patterns and anomalies.