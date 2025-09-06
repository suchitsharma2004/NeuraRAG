#!/usr/bin/env python3
"""
Generate a secure Django secret key for production deployment.
Run this script and copy the output to your Render environment variables.
"""

import os
import sys

# Add the Django project to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'RAG'))

try:
    from django.core.management.utils import get_random_secret_key
    
    print("=" * 60)
    print("🔐 DJANGO SECRET KEY GENERATOR")
    print("=" * 60)
    print()
    print("Copy this secret key to your Render environment variables:")
    print()
    print("DJANGO_SECRET_KEY=" + get_random_secret_key())
    print()
    print("⚠️  IMPORTANT: Keep this key secret and never commit it to version control!")
    print("=" * 60)
    
except ImportError:
    print("Django not found. Please install requirements first:")
    print("pip install -r requirements.txt")
