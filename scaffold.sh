#!/bin/bash

# scaffold.sh - Create RAG application folder structure

set -e

echo "Creating RAG application folder structure..."

# Create main directories
mkdir -p app
mkdir -p data/inbox
mkdir -p data/chroma
mkdir -p eval

# Create app files
touch app/main.py
touch app/config.py
touch app/retriever.py
touch app/generator.py
touch app/ingestion.py
touch app/context.py
touch app/utils.py

# Create eval files
touch eval/qa.jsonl
touch eval/run_ragas.py

# Create root files
touch .env
touch requirements.txt
touch README.md
