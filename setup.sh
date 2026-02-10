#!/bin/bash

# LegoGen Setup Script (Linux/macOS)
# This script sets up the LegoGen development environment

set -e  # Exit on error

echo "========================================="
echo "LegoGen Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "Virtual environment created."
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated."
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created. Please edit it to add your API keys."
    echo ""
fi

# Create exports and saves directories
echo "Creating export and save directories..."
mkdir -p exports
mkdir -p saves
echo "Directories created."
echo ""

# Run configuration check
echo "Checking configuration..."
python3 config.py
echo ""

# Run tests to verify installation
echo "Running tests to verify installation..."
python3 -m unittest discover tests -v
TEST_EXIT_CODE=$?
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "Setup Complete!"
    echo "========================================="
    echo ""
    echo "To get started:"
    echo "  1. Edit .env file and add your API keys (optional)"
    echo "  2. Activate virtual environment: source venv/bin/activate"
    echo "  3. Run the application: python3 main.py"
    echo ""
    echo "For more information, see README.md"
    echo ""
else
    echo "========================================="
    echo "Setup completed with test failures"
    echo "========================================="
    echo ""
    echo "Some tests failed, but the environment is set up."
    echo "This may be due to missing optional dependencies."
    echo "The application should still work for basic functionality."
    echo ""
fi
