#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies from parent directory
pip install -r ../requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate

# Create data directories for RAG functionality
mkdir -p data/faiss_index
mkdir -p data/documents
