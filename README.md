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
  - [Homework Observations](#homework-observations)

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Structure
- `baseline/`: Common starter system (retriever + generator)
  - `config/`: Configuration settings for the RAG system
  - `data/`: Sample text files for testing
  - `evaluation/`: Tools for evaluating retriever and generator performance
  - `generator/`: Text generation components using LLMs
  - `insight/`: Storage for analytics data from experiments
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
