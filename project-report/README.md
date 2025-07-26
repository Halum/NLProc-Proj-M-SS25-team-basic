# Project Report - LaTeX Document

This folder contains the academic paper "MovieQA-RAG: A Modular RAG Framework with Metadata-Guided Retrieval and Explainable Evaluation" in LaTeX format.

## Quick Start

### Editing the Paper
- **Main file to edit**: `report.tex`
- Contains the complete paper content including sections, figures, tables, and references
- Uses ACL2023 conference format

### Compilation

#### Option 1: Quick Compile (Recommended)
```bash
./quick_compile.sh
```

#### Option 2: Manual Compilation
```bash
./compile.sh
```

#### Option 3: Manual LaTeX Commands
```bash
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

### Output
- Generated PDF: `report.pdf`
- View the compiled paper with any PDF reader

## File Structure

- `report.tex` - Main LaTeX source file
- `report.pdf` - Compiled PDF output
- `acl2023.sty` - ACL 2023 conference style file
- `tikzstyles.sty` - Custom TikZ styles for figures
- `custom.bib` - Bibliography file
- `compile.sh` / `quick_compile.sh` - Compilation scripts

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Required packages: tikz, pgfplots, hyperref, tabularx, etc.

## Notes

- The paper uses ACL2023 format for academic conference submission
- Includes custom visualizations with TikZ/pgfplots
- All figures are generated directly in LaTeX for publication quality
