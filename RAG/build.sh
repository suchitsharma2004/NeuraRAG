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
# Only create documents directory - we're using Pinecone for vectors
mkdir -p data/documents
