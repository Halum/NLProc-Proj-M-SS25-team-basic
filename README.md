# RAG Project – Summer Semester 2025

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Structure
- `baseline/`: Common starter system (retriever + generator)
- `experiments/`: Each team's independent exploration
- `evaluation/`: Common tools for comparing results
- `utils/`: Helper functions shared across code

## Getting Started
1. Clone the repo
2. `cd baseline/`
3. Install dependencies: `pip install -r ../requirements.txt`

## Teams & Tracks

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