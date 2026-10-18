> *This project has been created as part of the 42 curriculum by okruhlia.*

# FTRAG: Against The Machine

## [01] DESCRIPTION: MISSION BRIEFING

**FTRAG** is a modular, locally hosted Retrieval-Augmented Generation (RAG) pipeline built from scratch. 
* **Objective:** Handle full-cycle document ingestion, highly optimized BM25-based indexing, fast search retrieval, and LLM answer generation without relying on external APIs.
* **Tech Stack:** Pure Python 3.12+, strict type hinting (`mypy`), `uv` for lightning-fast dependency management, and local Hugging Face Causal LMs.
* **Constraints:** The system rigorously enforces memory-efficient local execution and requires a strictly controlled Linux environment to run the `moulinette` evaluation binary.

## [02] INSTRUCTIONS: DEPLOYMENT & USAGE

To initialize the pipeline and evaluate the retrieval engine, execute the following commands:

```
# Dependency bootstrapping (Installs packages in an isolated uv environment)
make install

# Code structure scan (Runs Flake8 and MyPy type checks)
make lint

# Local evaluation (Tests search accuracy)
make test
```

**Example Usage (Manual CLI Execution):**
```
# Build the index
uv run python -m src index --max_chunk_size 1800

# Search query
uv run python -m src search --query "How does the caching work?" --k 5

# Generate answer
uv run python -m src answer --query "How does the caching work?" --k 5

# Evaluate locally
uv run python -m src evaluate --student_search_results_path path/to/results.json --dataset_path path/to/dataset.json
```

## [03] ALGORITHM: CHUNKING & RETRIEVAL METHOD

The retrieval matrix relies on a custom-built, dependency-free statistical calculation engine.

* **Chunking Strategy:** The ingestor utilizes LangChain's `RecursiveCharacterTextSplitter` to process `.py` and `.md` files dynamically. To prevent breaking semantic code blocks (like functions or classes) in half, we implemented a robust `chunk_overlap` of 300 characters and locked the ideal `max_chunk_size` at 1800. This ensures the algorithm has a wide enough "capture window" for complex logic.
* **Retrieval Method (BM25):** We deployed a highly tuned BM25 algorithm (`k1 = 1.2`, `b = 0.75`). To maximize search accuracy, the system uses an extremely aggressive custom tokenization approach (shattering CamelCase words and stripping all non-alphanumeric characters). This normalizes the data and prevents term saturation.

## [04] SYSTEM ARCHITECTURE

The system abandons monolithic design in favor of a 4-stage sequential pipeline:
* **Phase 1 - Index:** Parses raw data, tokenizes it aggressively, and calculates the mathematical weights (TF, IDF, DL) for every token, saving the coordinates and stats independently.
* **Phase 2 - Search:** Executes fast retrieval of the highest-scoring text chunks based on user prompts using a binary heap (`heapq`) for optimized top-k sorting.
* **Phase 3 - Answer:** Loads the retrieved context into memory and forces a local Causal LM (`Qwen/Qwen3-0.6B`) to synthesize a concrete, grounded answer.
* **Phase 4 - Evaluate:** Employs a strict mathematical comparator (Coverage validation) to measure retrieval effectiveness against expected outcomes.

## [05] ENGINEERING: DESIGN DECISIONS & CHALLENGES FACED

Building a purely lexical search engine from scratch presented several architectural hurdles:

* **Design Decision (Hyper-Aggressive Tokenization):** We opted to aggressively shatter tokens (e.g., splitting `my_super_variable` and `vLLM-0.10.1` into atomic parts) because it is impossible to predict the user's expertise level. A beginner might use vague, descriptive keywords, while a professional uses exact nomenclature. Stripping the text to its atomic components bridges this gap.
* **Challenge Faced (Parameter Volatility vs. Chunk Size):** Tuning BM25 parameters alongside aggressive text cleaning proved highly volatile. Small deviations in the length penalty (`b`) caused up to 5% metric swings depending on the chunk size. Instead of chasing an elusive "universal" parameter set for highly variable sizes, we solved this by fixing a stable chunk size (1800 characters).
* **Challenge Faced (Evaluation Logic):** Initially, using standard Intersection over Union (IoU) artificially penalized large chunks that correctly contained the answer. Transitioning the local evaluation to an exact "Coverage" formula (`intersection / truth_length >= 0.05`) allowed us to expand the chunk size without catastrophic scoring penalties.
* **Performance Analysis:** After stabilizing the chunk size at 1800 and tuning the tokenization, the system achieved highly stable recall@k scores: safely passing the **>80% threshold for documentation** and comfortably exceeding the **>50% threshold for code**, peaking at ~66% for code retrieval.

## [06] RESOURCES & AI COLLABORATION

The pipeline was stabilized utilizing the following data havens:
* **Hugging Face Infrastructure:** Exploited for downloading and executing quantized local models for secure, offline text generation.
* **LangChain:** Utilized for recursive semantic text splitting.
* **AI Collaboration:** AI was utilized not just as a code assistant, but as a consultant, a "psychologist" for debugging morale, and a strict "controller" conducting mathematical interrogations to ensure the developer's full comprehension of the BM25 algorithm and pipeline architecture.