#!/bin/bash

# LaTeX Compilation Script for Academic Paper
# This script compiles the report.tex file and handles all necessary steps

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "report.tex" ]; then
    print_error "report.tex not found! Please run this script from the project-report directory."
    exit 1
fi

print_status "Starting LaTeX compilation process..."

# Clean up previous compilation artifacts
print_status "Cleaning up previous compilation files..."
rm -f *.aux *.bbl *.blg *.log *.out *.synctex.gz *.toc *.lof *.lot *.nav *.snm *.vrb

# First compilation
print_status "Running first pdflatex compilation..."
if pdflatex -interaction=nonstopmode report.tex > latex_output.log 2>&1; then
    print_success "First compilation completed"
else
    print_error "First compilation failed! Check latex_output.log for details"
    tail -n 20 latex_output.log
    exit 1
fi

# Check if bibliography is needed (look for \bibliography or \cite commands)
if grep -q "\\\\bibliography\|\\\\cite" report.tex; then
    print_status "Bibliography detected, running bibtex..."
    if bibtex main >> latex_output.log 2>&1; then
        print_success "BibTeX completed"
        
        # Second compilation after bibtex
        print_status "Running second pdflatex compilation (for citations)..."
        if pdflatex -interaction=nonstopmode report.tex >> latex_output.log 2>&1; then
            print_success "Second compilation completed"
        else
            print_warning "Second compilation had issues, but continuing..."
        fi
        
        # Third compilation to resolve all references
        print_status "Running third pdflatex compilation (final)..."
        if pdflatex -interaction=nonstopmode report.tex >> latex_output.log 2>&1; then
            print_success "Third compilation completed"
        else
            print_warning "Third compilation had issues, but PDF should be ready"
        fi
    else
        print_warning "BibTeX failed, continuing without bibliography"
        # Second compilation anyway for cross-references
        print_status "Running second pdflatex compilation (for cross-references)..."
        pdflatex -interaction=nonstopmode report.tex >> latex_output.log 2>&1
    fi
else
    print_status "No bibliography detected, running second compilation for cross-references..."
    if pdflatex -interaction=nonstopmode report.tex >> latex_output.log 2>&1; then
        print_success "Second compilation completed"
    else
        print_warning "Second compilation had issues, but PDF should be ready"
    fi
fi

# Check if PDF was generated
if [ -f "main.pdf" ]; then
    print_success "Compilation successful! PDF generated: main.pdf"
    
    # Show PDF size and page count
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo main.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
        if [ ! -z "$pages" ]; then
            print_status "PDF contains $pages pages"
        fi
    fi
    
    size=$(ls -lh main.pdf | awk '{print $5}')
    print_status "PDF size: $size"
    
    # Option to open PDF
    echo
    read -p "Would you like to open the PDF? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v open &> /dev/null; then
            open main.pdf
        elif command -v xdg-open &> /dev/null; then
            xdg-open main.pdf
        else
            print_info "Please open main.pdf manually"
        fi
    fi
else
    print_error "PDF generation failed! Check latex_output.log for errors"
    echo
    print_status "Last 30 lines of compilation log:"
    tail -n 30 latex_output.log
    exit 1
fi

# Clean up compilation artifacts (optional)
echo
read -p "Would you like to clean up compilation artifacts? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Cleaning up compilation artifacts..."
    rm -f *.aux *.bbl *.blg *.log *.out *.synctex.gz *.toc *.lof *.lot *.nav *.snm *.vrb latex_output.log
    print_success "Cleanup completed"
else
    print_status "Keeping compilation artifacts for debugging"
fi

print_success "LaTeX compilation script completed!"
