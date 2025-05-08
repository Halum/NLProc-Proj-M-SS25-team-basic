# Homework Week 2 Observation

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

-  **Evaluation Strategy:** The same LLM (flan-alpaca-base) was prompted to compare its own generated answer with the reference answer and return a relevance score from 1 to 5

-  **Metrics Recorded:**

   -  Execution time per question

   -  Retrieved chunks for each question

   -  Evaluation score from the model

## Observations

### Execution Time Varies Across Similar Questions
Even though the 5 questions are semantically similar and derived from the same topic, the execution time varies significantly for each.

#### Possible Explanations:
- **Retrieval Overhead:** Slight differences in phrasing can cause different embeddings, leading to varied retrieval results. These, in turn, may affect search latency and total execution time.

- **LLM Load and Decoding Time:** Variations in the retrieved context length or complexity can impact the LLM's generation time.

### Different Chunks Retrieved Despite Similar Questions
Similar but differently phrased questions often retrieve different top-k chunks, though some chunks overlap.

#### Possible Explanations:
- **Embedding Sensitivity:** Even small changes in wording affect sentence embeddings significantly, leading to divergent nearest neighbor search results.

- **Retrieval Vector Granularity:** Depending on how the text was chunked (e.g., sentence vs. paragraph), the retrieval model may favor different matches based on slight shifts in semantic focus.

- **Lack of Query Normalization:** No paraphrase normalization or semantic rephrasing causes the system to treat them as distinct queries.

### Evaluation by the Same LLM Yields Unreliable Scores
The same LLM (`flan-alpaca-base`) used for answer generation is also used for evaluating answers against expected outputs.

It either returns a perfect score (5) consistently or echoes part of the generated answer instead of scoring.

#### Possible Explanations:
- **Lack of Evaluation Prompt Calibration:** The scoring prompt might not be well-structured or explicit enough to force a numeric response.

- **Model Limitations:** flan-alpaca-base, being a fine-tuned version of FLAN-T5, is not specialized for rubric-based or comparative scoring. It may lack instruction-following precision for scoring tasks.

- **Bias Toward Generation:** The model is likely biased toward generating content rather than making structured evaluations unless explicitly guided to do so.

## Possible Improvements
- **Execution Consistency**

  - Normalize questions using paraphrase models before embedding.

- **Retrieval Evaluation**

  - Log embedding distances for retrieved chunks to understand variance.

  - Experiment with sentence-level vs. paragraph-level chunking to find a better balance.

  - Use t-SNE or cosine similarity clustering to visualize how semantically similar teh question embeddings are.
  
  - Record retrieved chunks, embeddings, and scores for comparison.

- **Answer Evaluation**

  - Use a different, more capable model (e.g., GPT-3.5/4, or a reward model fine-tuned for evaluation) to assess answer quality.

  - Design a clearer evaluation prompt, only return the score as a number.

Observation regarding cosine similarity comparison with visualization can be found in [EMBEDDING_VISUALIZATION.md](../../week_2_tasks/EMBEDDING_VISUALIZATION.md)