# RAG Project – Summer Semester 2025

## Table of Contents
- [RAG Project – Summer Semester 2025](#rag-project--summer-semester-2025)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Structure](#structure)
  - [Environment Setup with Conda](#environment-setup-with-conda)
  - [🛠 Development Guidelines](#-development-guidelines)
    - [Task Flow](#task-flow)
    - [Branching Strategy](#branching-strategy)
      - [Branch Naming Conventions](#branch-naming-conventions)
      - [Commit Message Conventions](#commit-message-conventions)
    - [Coding Guideline](#coding-guideline)
  - [Homework Week 2 Observation](#homework-week-2-observation)
    - [Test Environment Description](#test-environment-description)
    - [Observations](#observations)
      - [Execution Time Varies Across Similar Questions](#execution-time-varies-across-similar-questions)
        - [Possible Explanations:](#possible-explanations)
      - [Different Chunks Retrieved Despite Similar Questions](#different-chunks-retrieved-despite-similar-questions)
        - [Possible Explanations:](#possible-explanations-1)
      - [Evaluation by the Same LLM Yields Unreliable Scores](#evaluation-by-the-same-llm-yields-unreliable-scores)
        - [Possible Explanations:](#possible-explanations-2)
    - [Possible Improvements](#possible-improvements)

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Structure
- `baseline/`: Common starter system (retriever + generator)
- `evaluation/`: Common tools for comparing results
- `utils/`: Helper functions shared across code
- `week_2_tasks/`: Week 2 assignment materials
  - [README.md](week_2_tasks/README.md): Textual explanations of embeddings and retrieval concepts
  - [EMBEDDING_VISUALIZATION.md](week_2_tasks/EMBEDDING_VISUALIZATION.md): Analysis of embedding visualization techniques

## Environment Setup with Conda

For a more controlled environment, we recommend using Conda:

1. **Install Miniconda**
   - Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

2. **Create a Conda Environment**
   ```bash
   conda create -n rag-project python=3.13
   conda activate rag-project
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Management**
   - Activate environment: `conda activate rag-project`
   - Deactivate environment: `conda deactivate`
   - List environments: `conda env list`

5. **Running the Application**
   - Navigate to the week_2_tasks folder:
     ```bash
     cd week_2_tasks
     ```
   - Run the main script:
     ```bash
     python main.py
     ```
   - This will:
     - Process sample texts from CSV files
     - Generate embeddings for text chunks
     - Visualize embeddings using PCA, t-SNE, and cosine similarity
     - Run question-answering tests using the RAG system
     - Evaluate the answers and report execution times

## 🛠 Development Guidelines

To maintain a clean and collaborative development workflow, please follow the guidelines below.

### Task Flow

1. **Create an Issue**  
   Open an issue **before starting any new task** to track progress and discussion.

2. **Create a Pull Request**  
   Submit all work as a pull request (PR) targeting the `develop` branch.  

---

### Branching Strategy

- **Target Branch for Marge:** `develop`

Important: **Do not merge directly into the `main` branch.**

#### Branch Naming Conventions

- **Feature Branches**  
  Format: `feature/<issue_number>-<short_description>`

  Example:  `feature/1-add-10-sample-texts`

- **Bugfix Branches**  
Format: `bugfix/<issue_number>-<short_description>`

    Example: `bugfix/42-fix-translation-error`

#### Commit Message Conventions

- Format: `#-<issue_number> <Message in present tense>`
   
   Example: `#3 - Update .gitignore to ignore vscode configs`
- Try to do incremental and modular commit. Do not dump all changes into a single commit.

---

### Coding Guideline

   - Do not use hardcoded values in code. Put them in `config`
   - Ensure decoupled function or pure functions so that most of the configuration are passed as argument and it becomes reusable.
   - Focus more on readable code than efficiency.
  
---

## Homework Week 2 Observation

Observation regarding cosine similarity comparison with visualization can be found in [EMBEDDING_VISUALIZATION.md](week_2_tasks/EMBEDDING_VISUALIZATION.md)

### Test Environment Description
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

### Observations

#### Execution Time Varies Across Similar Questions
Even though the 5 questions are semantically similar and derived from the same topic, the execution time varies significantly for each.

##### Possible Explanations:
- **Retrieval Overhead:** Slight differences in phrasing can cause different embeddings, leading to varied retrieval results. These, in turn, may affect search latency and total execution time.

- **LLM Load and Decoding Time:** Variations in the retrieved context length or complexity can impact the LLM’s generation time.

#### Different Chunks Retrieved Despite Similar Questions
Similar but differently phrased questions often retrieve different top-k chunks, though some chunks overlap.

##### Possible Explanations:
- **Embedding Sensitivity:** Even small changes in wording affect sentence embeddings significantly, leading to divergent nearest neighbor search results.

- **Retrieval Vector Granularity:** Depending on how the text was chunked (e.g., sentence vs. paragraph), the retrieval model may favor different matches based on slight shifts in semantic focus.

- **Lack of Query Normalization:** No paraphrase normalization or semantic rephrasing causes the system to treat them as distinct queries.

#### Evaluation by the Same LLM Yields Unreliable Scores
The same LLM (`flan-alpaca-base`) used for answer generation is also used for evaluating answers against expected outputs.

It either returns a perfect score (5) consistently or echoes part of the generated answer instead of scoring.

##### Possible Explanations:
- **Lack of Evaluation Prompt Calibration:** The scoring prompt might not be well-structured or explicit enough to force a numeric response.

- **Model Limitations:** flan-alpaca-base, being a fine-tuned version of FLAN-T5, is not specialized for rubric-based or comparative scoring. It may lack instruction-following precision for scoring tasks.

- **Bias Toward Generation:** The model is likely biased toward generating content rather than making structured evaluations unless explicitly guided to do so.

### Possible Improvements
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
