# RAG Project – Summer Semester 2025

## Table of Contents
- [RAG Project – Summer Semester 2025](#rag-project--summer-semester-2025)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Overall Goal](#overall-goal)
  - [Project Status: Working Components and Areas for Improvement](#project-status-working-components-and-areas-for-improvement)
    - [What's Working Well](#whats-working-well)
    - [Areas for Improvement](#areas-for-improvement)
    - [Next Steps](#next-steps)
  - [Structure](#structure)
  - [Environment Setup with Conda](#environment-setup-with-conda)
  - [How to Use](#how-to-use)
    - [Using the Streamlit UI](#using-the-streamlit-ui)
    - [Using the Command Line Interface](#using-the-command-line-interface)
  - [🛠 Development Guidelines](#-development-guidelines)
    - [Task Flow](#task-flow)
    - [Branching Strategy](#branching-strategy)
      - [Branch Naming Conventions](#branch-naming-conventions)
      - [Commit Message Conventions](#commit-message-conventions)
    - [Coding Guideline](#coding-guideline)
  - [Homework Observations](#homework-observations)

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Overall Goal
The primary goal of this project is to develop and analyze various Retrieval-Augmented Generation (RAG) systems to understand how different chunking strategies and retrieval methods impact the performance and accuracy of AI assistants. By comparing approaches such as fixed-size chunking, sliding window, semantic chunking, and others, this project aims to identify optimal configurations for various document types and query scenarios.

In later phase, extend the baseline RAG application to a certain specialization.


## Project Status: Working Components and Areas for Improvement

This section provides an overview of the current state of the RAG project, highlighting both the strengths and the areas that need further development.

### What's Working Well

| Component | Status | Description |
|-----------|--------|-------------|
| **Document Loader** | ✅ | Support for multiple file formats (PDF, DOCX, TXT) is implemented and functioning well, though more extensive testing with diverse document structures is needed. |
| **Chunking Strategies** | ✅ | Multiple chunking strategies are implemented and working effectively, including fixed-size, sliding window, sentence-based, paragraph-based, semantic, and markdown header-based approaches. |
| **Response Generation** | ✅ | The model consistently generates responses for all questions using the retrieved context. |
| **Automated Testing** | ✅ | The testing pipeline successfully processes documents, runs queries, and evaluates responses against labeled data. |
| **Insight Generation** | ✅ | The system generates detailed logs and insights about performance metrics for analysis. |
| **Analytical Visualization** | ✅ | Tools for visualizing embeddings, similarity scores, and performance metrics are implemented and working. |

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
   streamlit run streamlit/app.py
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

## Homework Observations
- [Week 2 Observations](homeworks/week_2/README.md) - Analysis of RAG system behavior and performance
  - Observation regarding cosine similarity comparison with visualization can be found in [EMBEDDING_VISUALIZATION.md](homeworks/week_2/EMBEDDING_VISUALIZATION.md)
  
- [Week 3 Observations](homeworks/week_3/README.md) - Comparison of different chunking strategies for RAG
  - Detailed analysis of how various chunking methods affect retrieval performance
  - Evaluation of accuracy and similarity scores across chunking approaches
  - Embedding visualizations and similarity analysis can be found in [EMBEDDING_VISUALIZATION.md](homeworks/week_3/EMBEDDING_VISUALIZATION.md)
