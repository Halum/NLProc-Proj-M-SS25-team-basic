# RAG Project – Summer Semester 2025

## Table of Contents
- [RAG Project – Summer Semester 2025](#rag-project--summer-semester-2025)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Overall Goal](#overall-goal)
  - [System Architecture](#system-architecture)
    - [Data Processing Pipeline](#data-processing-pipeline)
    - [Suggestion Generation Pipeline](#suggestion-generation-pipeline)
  - [Structure](#structure)
  - [Environment Setup with Conda](#environment-setup-with-conda)
  - [How to Use](#how-to-use)
    - [🎬 Specialization Overview](#-specialization-overview)
    - [🚀 Getting Started with Specialization](#-getting-started-with-specialization)
      - [1. Prepare Movie Data](#1-prepare-movie-data)
      - [2. Interactive Pipeline Execution](#2-interactive-pipeline-execution)
      - [3. Individual Pipeline Commands](#3-individual-pipeline-commands)
    - [📊 Data Management](#-data-management)
      - [Data Flow](#data-flow)
    - [🔧 Pipeline Details](#-pipeline-details)
      - [1. Raw to Processed Pipeline](#1-raw-to-processed-pipeline)
      - [2. Processed to Embeddings Pipeline](#2-processed-to-embeddings-pipeline)
      - [3. Evaluation Pipeline](#3-evaluation-pipeline)
      - [4. User Query Pipeline](#4-user-query-pipeline)
    - [🎯 Usage Examples](#-usage-examples)
      - [Movie Question Examples](#movie-question-examples)
      - [Evaluation Metrics](#evaluation-metrics)
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

## System Architecture

The RAG system is designed with a modular architecture consisting of two main processing workflows:

### Data Processing Pipeline
![Data Processing Architecture](Design%20Documents/Data%20Processing.png)

The data processing pipeline handles the ingestion and preparation of documents for retrieval. This includes document loading, chunking strategies, embedding generation, and vector store indexing.

### Suggestion Generation Pipeline
![Suggestion Generation Architecture](Design%20Documents/Suggestion%20Generation.png)

The suggestion generation pipeline manages the retrieval and generation process, taking user queries through the retrieval phase and generating contextually relevant responses using the configured language model.

## Structure
- `baseline/`: Implementation before moving to specilization
- `homeworks/`: Weekly assignment results and observations
- `specialization/`: Extended RAG system with specialized features
  - `config/`: Specialized configuration settings that extend baseline
  - `data/`: Specialized data management (isolated from baseline)
    - `raw/`: Raw CSV movie data files
    - `processed/`: CSV data processed to JSON structure
    - `db/`: ChromaDB databases
    - `insight/`: Logs, Analytics and evaluation insights
    - `tests/`: Test data and gold standard evaluations
  - `evaluation/`: Specialized evaluation tools and metrics
  - `generator/`: Enhanced text generation with specialized models
  - `pipelines/`: Processing pipelines for data transformation
  - `preprocessor/`: Extended document readers for CSV processing
  - `retriever/`: Enhanced retrieval components
  - `streamlit/`: Specialized Streamlit interface
  - `main.py`: Interactive CLI for running specialized pipelines


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

4. **Environment Variables Setup**
   - Copy `specialization/.env.example` as `specialization/.env`
   - Edit the .env file and add your OpenAI API key

5. **Environment Management**
   - Activate environment: `conda activate rag-project`
   - Deactivate environment: `conda deactivate`
   - List environments: `conda env list`

6. **Running the Pipelines**

   ```bash
   # Interactive CLI
   python specialization/main.py
   ```

7. **Managing Dependencies**
   - Whenever you install a new package, update requirements.txt:
   ```bash
   pip-chill > requirements.txt
   ```
   - This ensures that only direct dependencies (not sub-dependencies) are listed in requirements.txt


## How to Use

The specialization module extends the baseline RAG system with advanced features for movie data processing, enhanced retrieval strategies, and comprehensive evaluation frameworks. It processes CSV movie datasets and creates specialized embeddings for improved question-answering about films.

### 🎬 Specialization Overview

The specialization track focuses on:
- **Movie Data Processing**: Converting raw CSV movie datasets into processed JSON with genre filtering
- **Enhanced Embeddings**: Creating specialized vector embeddings from movie metadata and descriptions
- **Interactive Querying**: Providing a conversational interface for movie-related questions
- **Comprehensive Evaluation**: Testing system performance with gold standard movie questions

### 🚀 Getting Started with Specialization

#### 1. Prepare Movie Data

Place your raw movie CSV files in the `specialization/data/raw/` directory:
- `movies_metadata.csv`: Main movie information
- `credits.csv`: Cast and crew information  
- `keywords.csv`: Movie keywords and tags

#### 2. Interactive Pipeline Execution

Run the interactive CLI to execute pipelines:

```bash
python specialization/main.py
```

This will present you with an interactive menu:
```
Select a pipeline to execute:
1. Raw to Processed - Convert raw CSV files to processed JSON
2. Processed to Embeddings - Create vector embeddings from processed data  
3. Evaluation Pipeline - Run system evaluation with gold standard data
4. User Query Pipeline - Interactive query mode
5. Run All Pipelines (1-3 in sequence)
q. Quit
```

#### 3. Individual Pipeline Commands

You can also run specific pipelines directly:

```python
# Raw to Processed Pipeline
python specialization/pipelines/raw_to_processed.py

# Processed to Embeddings Pipeline  
python specialization/pipelines/processed_to_embeddings

# Evaluation Pipeline
python specialization/pipelines/evaluation_pipeline
```


### 📊 Data Management

The specialization module follows a strict data isolation principle:

#### Data Flow
1. **Raw CSV Files** → `specialization/data/raw/`
   - Original movie datasets (metadata, credits, keywords)
2. **Processed JSON** → `specialization/data/processed/`
   - Filtered and joined movie data in JSON format
3. **Vector Embeddings** → `specialization/data/db/`
   - ChromaDB collections with movie embeddings
4. **Evaluation Results** → `specialization/data/insight/`
   - Performance metrics and analysis results
5. **Test Data** → `specialization/data/tests/`
   - Question sets and gold standard answers

### 🔧 Pipeline Details

#### 1. Raw to Processed Pipeline
- Loads and joins multiple CSV files (movies, credits, keywords)
- Filters by target genres (Family, Mystery, Western)
- Samples data for manageable processing
- Outputs structured JSON with movie metadata

#### 2. Processed to Embeddings Pipeline
- Creates vector embeddings from processed movie data
- Stores embeddings in ChromaDB for efficient retrieval
- Includes metadata for enhanced search capabilities
- Batch processing for optimal performance

#### 3. Evaluation Pipeline
- Tests system performance against gold standard questions
- Measures retrieval accuracy and answer quality
- Generates detailed evaluation insights
- Saves results with timestamps for comparison

#### 4. User Query Pipeline
- Interactive movie question-answering interface
- Real-time retrieval and generation
- Contextual responses based on movie database
- Graceful handling of out-of-scope queries

### 🎯 Usage Examples

#### Movie Question Examples
```
Q: "What are some family-friendly Western movies?"
Q: "Tell me about mystery movies with high ratings"
Q: "Which movies star specific actors in family films?"
Q: "What are the keywords associated with Western genres?"
```

#### Evaluation Metrics
The system tracks:
- **Retrieval Accuracy**: How well relevant movie information is found
- **Answer Quality**: Relevance and accuracy of generated responses
- **Context Utilization**: Whether answers are grounded in retrieved data
- **Response Time**: Performance metrics for user experience

The evaluation generates detailed insights including:
- Precision, recall, and F1 scores
- Response quality metrics
- Retrieval effectiveness analysis
- Error analysis and improvement suggestions

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
