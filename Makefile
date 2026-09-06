.PHONY: install run debug clean lint test

install:
	@echo "=== Installing dependencies ==="
	uv sync

run:
	@echo "Running RAG..."
	@echo "=== 1. Indexing Corpus ==="
	uv run python -m src index --max_chunk_size 2000

	@echo "\n=== 2. Processing DOCS Dataset ==="
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results/UnansweredQuestions --k 5
	uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
	uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

	@echo "\n=== 3. Processing CODE Dataset ==="
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results/UnansweredQuestions --k 5
	uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
	uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

debug:
	@echo "Running RAG in debug mode..."
	uv run python -m pdb -m src $(CMD)

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	@echo "Running flake8..."
	uv run flake8 .
	@echo "Running mypy..."
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	@echo "=== Testing DOCS with Moulinette ==="
	./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5 --max_context_length 2000

	@echo "\n=== Testing CODE with Moulinette ==="
	./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --max_context_length 2000
