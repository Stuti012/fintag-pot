#!/bin/bash

echo "=========================================="
echo "Financial QA System Setup"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/finqa
mkdir -p data/edgar
mkdir -p chroma_db
mkdir -p evaluation_results

# Copy environment file
echo "Setting up environment..."
cp .env.example .env

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Download FinQA dataset to data/finqa/"
echo "3. Run: python app.py --help"
echo ""
echo "Quick start examples:"
echo "  # Evaluate FinQA:"
echo "  python app.py --mode eval --num-examples 10"
echo ""
echo "  # Download and query EDGAR:"
echo "  python app.py --mode edgar --download --ticker AAPL"
echo "  python app.py --mode edgar --build-index --ticker AAPL"
echo "  python app.py --mode edgar --query 'What was revenue?' --ticker AAPL"
echo ""
echo "  # Compare baseline vs proposed:"
echo "  python app.py --mode compare --num-examples 20"
echo ""
