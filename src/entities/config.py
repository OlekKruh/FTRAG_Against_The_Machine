from dataclasses import dataclass
from pathlib import Path


@dataclass
class DefaultConfig:
    raw_dir: Path = Path("data/raw")

    processed_files_dir: Path = Path("data/processed")
    index_file_name: str = "index.json"
    bm25s_file_name: str = "bm25s.json"

    srch_res_dir: Path = Path("data/output/search_results")
    srch_res_file_name: str = "search_dataset.json"
    srch_res_ans_dir: Path = Path("data/output/search_results_and_answer")
    srch_res_ans_file_name: str = "answer_dataset.json"
