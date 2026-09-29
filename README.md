> *This project has been created as part of the 42 curriculum by okruhlia.*

# FTRAG: Against The Machine

## [01] DESCRIPTION: MISSION BRIEFING

**FTRAG** is a modular, locally hosted Retrieval-Augmented Generation (RAG) pipeline built from scratch. 
* **Objective:** Handle full-cycle document ingestion, highly optimized BM25-based indexing, fast search retrieval, and LLM answer generation without relying on external APIs.
* **Tech Stack:** Pure Python 3.12+, strict type hinting (`mypy`), `uv` for lightning-fast dependency management, and local Hugging Face Causal LMs.
* **Constraints:** The system rigorously enforces memory-efficient local execution and requires a strictly controlled Linux environment to run the `moulinette` evaluation binary.

## [02] INSTRUCTIONS: DEPLOYMENT PROCEDURES

To initialize the pipeline and evaluate the retrieval engine, execute the following commands:

```bash
# Dependency bootstrapping (Installs packages in an isolated uv environment)
make install

# Code structure scan (Runs Flake8 and MyPy type checks)
make lint

# Local evaluation (Tests search accuracy against Moulinette)
make test
```

**Manual CLI Execution:**
```bash
# Build the index
uv run python -m src index --max_chunk_size 2000

# Search query
uv run python -m src search --query "How does the caching work?" --k 5

# Generate answer
uv run python -m src answer --query "How does the caching work?" --k 5

# Evaluate locally
uv run python -m src evaluate --student_search_results_path path/to/results.json --dataset_path path/to/dataset.json
```

## [03] ALGORITHM: LOGIC_CORE & INDEXING

The retrieval matrix relies on a custom-built, dependency-free statistical calculation engine.
* **Language-Aware Chunking:** The ingestor scans `data/raw/` for `.py` and `.md` files, dynamically splitting text into logical slices using LangChain, and saving the processed index to `data/processed/`.
* **BM25 Search Matrix:** Queries the pre-calculated BM25 index to retrieve the top-k most relevant chunks for any given prompt, penalizing extreme document lengths and preventing term saturation.
* **Recall Evaluation:** Tests search accuracy (Recall@k) against a ground-truth dataset by utilizing character offset Intersection over Union (IoU) to verify exact spatial matches.

## [04] ARCHITECTURE: TACTICAL PIPELINE

The system abandons monolithic design in favor of a 4-stage sequential pipeline.
* **Phase 1 - Index:** Parses raw data and calculates the mathematical weights for every token.
* **Phase 2 - Search:** Executes fast retrieval of the highest-scoring text chunks based on user prompts.
* **Phase 3 - Answer:** Loads the retrieved context into memory and forces a local Causal LM (`Qwen/Qwen3-0.6B`) to synthesize a concrete answer.
* **Phase 4 - Evaluate:** Employs a strict mathematical comparator to measure retrieval effectiveness against expected outcomes.

## [05] RESOURCES: INFORMATION ASSETS

The pipeline was stabilized utilizing the following data havens:
* **Hugging Face Infrastructure:** Exploited for downloading and executing quantized local models for secure, offline text generation.
* **Python Typing Vault:** Deep-level integration of strict typing (`BaseModel`, `Field`, `defaultdict`) to survive static analysis.
* **AI Collaboration:** Utilized as a tactical peer-reviewer for architectural design patterns, memory optimization strategies, and generating strict CLI documentation.