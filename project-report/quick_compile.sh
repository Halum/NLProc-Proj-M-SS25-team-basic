#!/bin/bash

# Quick LaTeX compilation script
# Usage: ./quick_compile.sh

echo "Quick LaTeX compilation..."

# Clean previous artifacts
rm -f *.aux *.log *.out

# Compile twice to resolve references
echo "Running pdflatex (1/2)..."
pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1

echo "Running pdflatex (2/2)..."
pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1

if [ -f "main.pdf" ]; then
    echo "✓ Compilation successful! PDF updated."
    ls -lh main.pdf | awk '{print "PDF size:", $5}'
else
    echo "✗ Compilation failed!"
    exit 1
fi
