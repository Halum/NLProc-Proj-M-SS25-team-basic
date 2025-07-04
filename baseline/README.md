# RAG Project – Summer Semester 2025

## Table of Contents
- [RAG Project – Summer Semester 2025](#rag-project--summer-semester-2025)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Overall Goal](#overall-goal)
  - [System Architecture](#system-architecture)
    - [Data Processing Pipeline](#data-processing-pipeline)
    - [Suggestion Generation Pipeline](#suggestion-generation-pipeline)
  - [Project Status: Working Components and Areas for Improvement](#project-status-working-components-and-areas-for-improvement)
    - [Areas for Improvement](#areas-for-improvement)
    - [Next Steps](#next-steps)
  - [Structure](#structure)
  - [Environment Setup with Conda](#environment-setup-with-conda)
  - [How to Use](#how-to-use)
    - [Using the Streamlit UI](#using-the-streamlit-ui)
    - [Using the Command Line Interface](#using-the-command-line-interface)
  - [Homework Observations](#homework-observations)

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Overall Goal
The primary goal of this project is to develop and analyze various Retrieval-Augmented Generation (RAG) systems to understand how different chunking strategies and retrieval methods impact the performance and accuracy of AI assistants. By comparing approaches such as fixed-size chunking, sliding window, semantic chunking, and others, this project aims to identify optimal configurations for various document types and query scenarios.

In later phase, extend the baseline RAG application to a certain specialization.

## System Architecture

The RAG system is designed with a modular architecture consisting of two main processing workflows:

### Data Processing Pipeline
![Data Processing Architecture](Design%20Documents/Data%20Processing.png)

The data processing pipeline handles the ingestion and preparation of documents for retrieval. This includes document loading, chunking strategies, embedding generation, and vector store indexing.

### Suggestion Generation Pipeline
![Suggestion Generation Architecture](Design%20Documents/Suggestion%20Generation.png)

The suggestion generation pipeline manages the retrieval and generation process, taking user queries through the retrieval phase and generating contextually relevant responses using the configured language model.

## Project Status: Working Components and Areas for Improvement

This section provides an overview of the current state of the RAG project, highlighting both the strengths and the areas that need further development.


### Areas for Improvement

| Component | Status | Description |
|-----------|--------|-------------|
| **Context Matching** | ⚠️ | The current method for matching labeled context with retrieved chunks is rigid and inefficient. The `find_chunk_containing_context` function only checks for exact substring matches, which doesn't account for semantic similarity or partial matches. |
| **Instruction Following** | ⚠️ | The current model struggles with complex instructions and chain-of-thought reasoning. Testing more advanced models or implementing specialized prompting techniques could improve performance. |
| **Grounding Verification** | ⚠️ | There's no reliable mechanism to verify if the generated answer is actually grounded in the provided context, potentially allowing for hallucinations. |
| **Answer Verification** | ⚠️ | Automated verification of generated answers remains a challenge, currently relying on human feedback which is not scalable for large-scale evaluation. |
| **Reproducibility** | ⚠️ | System reproducibility hasn't been systematically tested to ensure consistent results across multiple runs with the same inputs. |
| **Unknown Question Handling** | ⚠️ | The system lacks effective detection and handling of out-of-context questions, potentially leading to incorrect or misleading answers when information is not available in the source documents. |

### Next Steps

1. **Improve Context Matching**: Implement semantic similarity-based matching between labeled contexts and retrieved chunks.
2. **Enhance Model Capabilities**: Test more advanced models or implement better prompting techniques to improve complex instruction handling and chain-of-thought reasoning.
3. **Implement Grounding Verification**: Develop a mechanism to verify that generated answers are actually grounded in the provided context.
4. **Automate Answer Verification**: Create an automated system for evaluating answer quality without human intervention.
5. **Ensure Reproducibility**: Design and run tests to verify that the system produces consistent results across multiple runs.
6. **Improve Unknown Question Handling**: Implement detection for questions that cannot be answered from the available context and craft appropriate responses.



## Structure
- `baseline/`: Common starter system (retriever + generator)
  - `config/`: Configuration settings for the RAG system
  - `data/`: Holds different kinds of data files
    - `db/`: Vector databases and indexes
    - `insight/`: Storage for analytics data from experiments
    - `raw/`: Original input text files
    - `tests/`: Test data for system validation
  - `evaluation/`: Tools for evaluating retriever and generator performance
  - `generator/`: Text generation components using LLMs
  - `postprocessor/`: Output formatting and document writing tools
  - `preprocessor/`: Document reading and text chunking services
  - `retriever/`: Vector store and retrieval components
  - `pipeline.py`: Main execution script for the RAG pipeline
- `evaluation/`: Common tools for comparing results
- `homeworks/`: Weekly assignment results and observations
  - `week_2/`: Analysis of RAG system behavior with various questions
  - `week_3/`: Comparison of different chunking strategies
- `utils/`: Helper functions shared across code

## Environment Setup with Conda

For a more controlled environment, we recommend using Conda:

1. **Install Miniconda**
   - Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

2. **Create a Conda Environment**
   ```bash
   conda create -n rag-project python=3.12
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

5. **Running the Baseline Pipeline**
   ```bash
   python baseline/pipeline.py
   ```

6. **Managing Dependencies**
   - Whenever you install a new package, update requirements.txt:
   ```bash
   pip-chill > requirements.txt
   ```
   - This ensures that only direct dependencies (not sub-dependencies) are listed in requirements.txt


## How to Use

### Using the Streamlit UI

The project includes a user-friendly Streamlit application that provides an interactive way to explore different chunking strategies for RAG:

1. **Start the Streamlit app**:
   ```bash
   streamlit run baseline/streamlit/app.py
   ```

2. **Navigate the app's interface**:
   - **Preprocessing Tab**: Select and configure chunking strategies, process documents, and view results
   - **Interaction Tab**: Ask questions and compare answers across different chunking strategies
   - **Chat Tab**: Interact with the RAG system in a conversational manner
   - **Insights Tab**: Analyze performance metrics and visualizations of different strategies

3. **Exploring chunking strategies**:
   - Select one or more strategies from the sidebar (Fixed Size, Sliding Window, Sentence-Based, etc.)
   - Adjust parameters like chunk size and overlap using the provided sliders
   - Process documents with the selected strategies and compare their performance

4. **Evaluating results**:
   - Use predefined sample questions or enter your own queries
   - Compare the answers generated using different strategies
   - Analyze metrics like correctness rate and context retrieval success

### Using the Command Line Interface

For programmatic use or batch processing, you can run the RAG pipeline directly from the command line:

1. **Run the baseline pipeline**:
   ```bash
   python baseline/pipeline.py
   ```

2. **Options and configurations**:
   - Use the `--withindex` flag to load existing vector indexes: `python baseline/pipeline.py --withindex`

3. **Viewing results**:
   - The pipeline outputs query results, expected and generated answers, and context information
   - Retrieval statistics and performance metrics are displayed in the terminal
   - Insights are saved automatically for later analysis


## Homework Observations
- [Week 2 Observations](homeworks/week_2/README.md) - Analysis of RAG system behavior and performance
  - Observation regarding cosine similarity comparison with visualization can be found in [EMBEDDING_VISUALIZATION.md](homeworks/week_2/EMBEDDING_VISUALIZATION.md)
  
- [Week 3 Observations](homeworks/week_3/README.md) - Comparison of different chunking strategies for RAG
  - Detailed analysis of how various chunking methods affect retrieval performance
  - Evaluation of accuracy and similarity scores across chunking approaches
  - Embedding visualizations and similarity analysis can be found in [EMBEDDING_VISUALIZATION.md](homeworks/week_3/EMBEDDING_VISUALIZATION.md)
