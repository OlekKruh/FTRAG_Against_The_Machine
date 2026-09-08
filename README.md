# FTRAG Against The Machine

A modular, local Retrieval-Augmented Generation (RAG) pipeline built from scratch. This project handles document ingestion, BM25-based indexing, search, evaluation, and LLM answer generation using local Hugging Face models.

## Architecture

The pipeline is composed of the following sequential stages:
1. Index: Scans data/raw/ for .py and .md files, splits text into language-aware chunks using LangChain, calculates BM25 statistics, and saves the index to data/processed/.
2. Search: Queries the BM25 index to retrieve the top-k most relevant chunks for a given prompt.
3. Answer: Loads retrieved context and generates an answer using a local Causal LM (Qwen/Qwen3-0.6B).
4. Evaluate: Tests search accuracy (Recall@k) against a ground-truth dataset using character offset Intersection over Union (IoU).

## Prerequisites

- Python 3.12+
- uv (Fast Python package and project manager)
- Linux environment (required for the moulinette evaluation binary)

## Installation

Install all strictly typed dependencies in an isolated virtual environment:

    make install

## Usage

The project utilizes fire to provide a clean CLI interface.

Run static analysis and linting (MyPy & Flake8):

    make lint

Run local evaluation against Moulinette:

    make test

Manual CLI Commands:

    # 1. Build the index
    uv run python -m src index --max_chunk_size 2000

    # 2. Search for a specific query
    uv run python -m src search --query "How does the caching work?" --k 5

    # 3. Generate an answer
    uv run python -m src answer --query "How does the caching work?" --k 5

    # 4. Local Evaluation (Recall@k)
    uv run python -m src evaluate --student_search_results_path path/to/results.json --dataset_path path/to/dataset.json