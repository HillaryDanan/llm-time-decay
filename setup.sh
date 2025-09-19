#!/bin/bash

# LLM Time Decay - Quick Setup Script
# Run: bash setup.sh

echo "🚀 Setting up LLM Time Decay experiment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate it
echo "Activating venv..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed}
mkdir -p results/{figures,tables}
mkdir -p tests

# Create .gitkeep files to track empty dirs
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch results/tables/.gitkeep

# Copy env template if .env doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Test: python3 src/runner.py --models gpt-3.5 --depths fractional --trials 2"
echo ""
echo "🔬 Let's do science!"