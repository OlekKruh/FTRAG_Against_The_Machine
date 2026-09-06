from pathlib import Path
from src.entities.BM25Builder import BM25Methods as BM25m
from src.entities.models import MinimalSource, BM25Stat
from src.utils.exits import error_exit
from src.utils.log_location import log_location
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, max_size: int = 2000) -> None:
        self.splitters = {
            ".py": RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=max_size, chunk_overlap=0, add_start_index=True,
            ),
            ".md": RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN, chunk_size=max_size, chunk_overlap=0, add_start_index=True,
            )
        }

    def process_file(self, raw_text: str, path: Path, bm25s: BM25Stat) -> list[MinimalSource]:
        res_obj_list = []

        ext = path.suffix
        splitter = self.splitters.get(ext)
        if not splitter:
            error_exit(f">>> {log_location()}\n"
                       f">>> Context: Have no suitable method for '{ext}'")

        docs = splitter.create_documents([raw_text])  # грызем текст

        for i, doc in enumerate(docs):
            start_pos = doc.metadata["start_index"]
            end_pos = start_pos + len(doc.page_content)

            res_obj_list.append(MinimalSource(
                file_path=str(path),
                first_character_index=start_pos,
                last_character_index=end_pos,
            ))

            BM25m.despatch_bm25_stat(doc.page_content, bm25s)

        return res_obj_list
