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

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Structure
- `baseline/`: Common starter system (retriever + generator)
- `experiments/`: Each team's independent exploration
- `evaluation/`: Common tools for comparing results
- `utils/`: Helper functions shared across code

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