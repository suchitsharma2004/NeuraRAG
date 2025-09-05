#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Navigate to Django project directory
cd RAG

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate

# Create data directories for RAG functionality
mkdir -p data/faiss_index
mkdir -p data/documents
