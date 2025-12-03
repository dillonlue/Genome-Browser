#!/bin/bash

# LaTeX Document Compilation Script
# ==================================

# Add TeXLive to PATH if it exists in the user's home directory
if [ -d "$HOME/texlive/2024/bin/x86_64-linux" ]; then
    export PATH="$HOME/texlive/2024/bin/x86_64-linux:$PATH"
fi

echo "LaTeX Document Compiler"
echo "======================"
echo ""

# Compile the document
echo "Compiling main.tex..."
echo ""

# Run pdflatex twice to resolve references with -interaction=nonstopmode to prevent keyboard input
pdflatex -interaction=nonstopmode main.tex

if [ $? -eq 0 ]; then
    echo ""
    echo "First pass complete. Running second pass for references..."
    pdflatex -interaction=nonstopmode main.tex

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Compilation successful!"
        echo "  Output: main.pdf"
        echo ""
        ls -lh main.pdf
    else
        echo ""
        echo "✗ Second pass failed. Check the log file: main.log"
        exit 1
    fi
else
    echo ""
    echo "✗ First pass failed. Check the log file: main.log"
    echo ""
    echo "Common issues:"
    echo "  - Missing packages: Install texlive-latex-extra"
    echo "  - Syntax errors: Check main.tex for LaTeX errors"
    exit 1
fi
