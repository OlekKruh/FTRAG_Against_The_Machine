from pathlib import Path
import sys
from src.entities.BM25Builder import BM25Methods as BM25m
from src.entities.models import MinimalSource, BM25Stat
from src.utils.exits import error_exit
from src.utils.log_location import log_location
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


class Chunker:
    """
    Splits text documents into smaller chunks based on their language syntax.

    Utilizes LangChain's RecursiveCharacterTextSplitter to ensure semantic
    blocks (like functions or paragraphs) remain intact during splitting.
    """
    def __init__(self, max_size) -> None:
        """
        Initializes the text splitters for supported file extensions.

        Args:
            max_size (int): The maximum allowed character length for a single chunk.
        """

        py_size = max(600, min(max_size, 1500))
        md_size = max(600, min(max_size, 1500))

        print(f"-> [DEBUG] Input: {max_size} | py_size: {py_size} | md_size: {md_size}")

        self.splitters = {
            ".py": RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=py_size,
                chunk_overlap=0,
                add_start_index=True,
            ),
            ".md": RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=md_size,
                chunk_overlap=250,
                add_start_index=True,
            )
        }

    def process_file(self, raw_text: str, path: Path, bm25s: BM25Stat) -> list[MinimalSource]:
        """
        Slices the raw text into chunks, records their coordinates, and updates BM25 stats.

        Args:
            raw_text (str): The full textual content of the target file.
            path (Path): The file path, used to determine the correct language splitter.
            bm25s (BM25Stat): The statistics tracker updated with token frequencies from each chunk.

        Returns:
            list[MinimalSource]: A list of objects containing the file path and exact
                                 character indices (start and end) for each chunk.
        """
        res_obj_list = []

        ext = path.suffix
        splitter = self.splitters.get(ext)
        if not splitter:
            error_exit(f"{log_location()}\n"
                       f">>> Context: Have no suitable method for '{ext}'")

        docs = splitter.create_documents([raw_text])

        for i, doc in enumerate(docs):
            start_pos = doc.metadata["start_index"]
            end_pos = start_pos + len(doc.page_content)

            res_obj_list.append(MinimalSource(
                file_path=str(path),
                first_character_index=start_pos,
                last_character_index=end_pos,
            ))

            BM25m.despatch_bm25_stat(doc.page_content, bm25s, ext)

        return res_obj_list
