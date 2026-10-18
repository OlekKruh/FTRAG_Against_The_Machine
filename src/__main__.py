import fire
from tqdm import tqdm
from pathlib import Path
from src.entities.BM25Builder import BM25Methods as BM25m
from src.entities.chunker import Chunker
from src.entities.config import DefaultConfig as Def_Con
from src.entities.file_io import FileIO as F_io
from src.entities.llm_engine import LLMEngine
from src.entities.models import BM25Stat, IndexFile, StudentSearchResultsAndAnswer, MinimalAnswer
from src.entities.path_finder import PathFinder
from src.entities.models import RagDataset as RagData
from src.entities.models import MinimalSource as MinSou
from src.entities.models import MinimalSearchResults as MinSrcRes
from src.entities.models import StudentSearchResults as StSeRe


class RAGPipeline:
    """
    Main pipeline for the RAG (Retrieval-Augmented Generation) system.
    Provides CLI commands for indexing documents, searching, generating answers,
    and evaluating retrieval metrics.
    """
    def index(self, max_chunk_size: int = 800) -> None:
        """
        Parses raw data and builds the BM25 index.

        Scans the raw directory for valid files, splits them into chunks
        based on max_chunk_size, calculates BM25 statistics, and saves
        the processed index and statistics to the disk.

        Args:
            max_chunk_size (int): Maximum character length for a single chunk.
                                  Defaults to 800.
        """
        # === Activate objects ===
        pathfinder = PathFinder(Def_Con.raw_dir)
        chunker = Chunker(max_chunk_size)

        pathfinder.pars_dir()
        bm25s = BM25Stat()
        minimal_source_to_save = []

        # === Indexing Pipeline ===
        all_paths = [path for paths_list in pathfinder.path_map.values() for path in paths_list]

        for path in tqdm(all_paths, desc="File Indexation", unit="file"):
            raw_data = F_io.file_read(path=path)
            org_data = chunker.process_file(raw_text=raw_data, path=path, bm25s=bm25s)
            minimal_source_to_save.extend(org_data)

        BM25m.calculate_avgdl(bm25s=bm25s)
        BM25m.calculate_idf(bm25s=bm25s)

        index_wrapper = IndexFile(chunks=minimal_source_to_save)

        F_io.file_write_json(
            dir_path=Def_Con.processed_files_dir,
            file_name=Def_Con.index_file_name,
            data=index_wrapper
        )
        F_io.file_write_json(
            dir_path=Def_Con.processed_files_dir,
            file_name=Def_Con.bm25s_file_name,
            data=bm25s
        )

    def search(self, query: str, k: int = 5) -> None:
        """
        Retrieves and prints the top-k document chunks for a single search query.

        Args:
            query (str): The search prompt entered by the user.
            k (int): The number of top results to retrieve. Defaults to 5.
        """
        index_file_path = Def_Con.processed_files_dir / Def_Con.index_file_name
        bm25s_file_path = Def_Con.processed_files_dir / Def_Con.bm25s_file_name

        index_obj = F_io.file_read_json(path=index_file_path, obj=IndexFile)
        bm25s_obj = F_io.file_read_json(path=bm25s_file_path, obj=BM25Stat)

        top_results = BM25m.get_top_k(query=query, bm25s=bm25s_obj, k=k)

        for chunk_id, score in top_results:
            source = index_obj.chunks[int(chunk_id)]
            print(f"-> Score: {score:.4f}\n"
                  f"-> File: {source.file_path}\n"
                  f"-> Pos: {source.first_character_index}-{source.last_character_index}\n")

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        """
        Executes a batch search over an entire dataset of questions.

        Evaluates each question in the dataset using the BM25 index and
        saves the aggregated results to a JSON file.

        Args:
            dataset_path (str): Path to the input dataset JSON.
            save_directory (str): Directory where the search results will be saved.
            k (int): The number of chunks to retrieve per question. Defaults to 5.
        """
        index_file_path = Def_Con.processed_files_dir / Def_Con.index_file_name
        bm25s_file_path = Def_Con.processed_files_dir / Def_Con.bm25s_file_name

        index_obj = F_io.file_read_json(path=index_file_path, obj=IndexFile)
        bm25s_obj = F_io.file_read_json(path=bm25s_file_path, obj=BM25Stat)
        dataset_obj = F_io.file_read_json(path=Path(dataset_path), obj=RagData)

        all_results = []

        for obj in tqdm(dataset_obj.rag_questions, desc="Search through a dataset.", unit="query"):
            top_results = BM25m.get_top_k(query=obj.question, bm25s=bm25s_obj, k=k)

            sources_for_question = []

            for chunk_id, score in top_results:
                source = index_obj.chunks[int(chunk_id)]

                source_model = MinSou(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index
                )
                sources_for_question.append(source_model)

            question_result = MinSrcRes(
                question_id=obj.question_id,
                question=obj.question,
                retrieved_sources=sources_for_question
            )

            all_results.append(question_result)

        final_output = StSeRe(
            search_results=all_results,
            k=k
        )
        F_io.file_write_json(
            dir_path=Path(save_directory),
            file_name=Path(dataset_path).name,
            data=final_output
        )

    def answer(self, query: str, k: int = 5) -> None:
        """
        Generates an LLM answer for a single query using retrieved context.

        Args:
            query (str): The user's question.
            k (int): The number of chunks to retrieve and use as context. Defaults to 5.
        """
        llm_engine = LLMEngine()
        llm_engine.load_model()

        index_file_path = Def_Con.processed_files_dir / Def_Con.index_file_name
        bm25s_file_path = Def_Con.processed_files_dir / Def_Con.bm25s_file_name

        index_obj = F_io.file_read_json(path=index_file_path, obj=IndexFile)
        bm25s_obj = F_io.file_read_json(path=bm25s_file_path, obj=BM25Stat)

        top_results = BM25m.get_top_k(query=query, bm25s=bm25s_obj, k=k)

        context: str = ""
        for chunk_id, score in top_results:
            source = index_obj.chunks[int(chunk_id)]
            extracted = F_io.file_read(Path(source.file_path))[source.first_character_index:source.last_character_index]
            context += extracted

        res = llm_engine.generate_answer(context=context, query=query)

        print(res)

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """
        Generates answers for a batch of previously retrieved search results.

        Reads the chunks coordinates, loads their text content to form a context window,
        and prompts the LLM to answer each question. Saves the final dataset to disk.

        Args:
            student_search_results_path (str): Path to the JSON containing search results.
            save_directory (str): Directory to save the final answers JSON.
        """
        llm_engine = LLMEngine()
        llm_engine.load_model()

        query_list_obj = F_io.file_read_json(path=Path(student_search_results_path), obj=StSeRe)

        all_answers = []
        for obj in tqdm(query_list_obj.search_results, desc="Answering dataset", unit="query"):
            context = ""
            for source in obj.retrieved_sources:
                raw_text = F_io.file_read(Path(source.file_path))
                context += raw_text[source.first_character_index:source.last_character_index] + "\n\n"

            answer_text = llm_engine.generate_answer(context=context, query=obj.question)

            question_answer = MinimalAnswer(
                question_id=obj.question_id,
                question=obj.question,
                retrieved_sources=obj.retrieved_sources,
                answer=answer_text
            )
            all_answers.append(question_answer)

        final_output = StudentSearchResultsAndAnswer(
            search_results=all_answers,
            k=query_list_obj.k
        )

        F_io.file_write_json(
            dir_path=Path(save_directory),
            file_name=Path(student_search_results_path).name,
            data=final_output
        )

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """
        Locally evaluates search accuracy (Recall@k) against a ground-truth dataset.

        Uses Coverage (intersection / truth_length >= 0.05) to determine if a
        retrieved chunk successfully matches the expected source document.

        Args:
            student_search_results_path (str): Path to the generated search results JSON.
            dataset_path (str): Path to the ground-truth dataset containing correct sources.
        """
        student_results = F_io.file_read_json(path=Path(student_search_results_path), obj=StSeRe)
        ground_truth = F_io.file_read_json(path=Path(dataset_path), obj=RagData)

        truth_dict = {q.question_id: q for q in ground_truth.rag_questions if hasattr(q, 'sources')}

        def calculate_coverage(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
            """
            Calculates the percentage of the ground-truth text covered by the retrieved chunk.

            Args:
                start_a (int): Start index of the retrieved chunk.
                end_a (int): End index of the retrieved chunk.
                start_b (int): Start index of the ground-truth chunk.
                end_b (int): End index of the ground-truth chunk.

            Returns:
                float: The coverage ratio (Intersection / Truth Length).
            """
            truth_len = max(1, end_b - start_b)
            intersection_start = max(start_a, start_b)
            intersection_end = min(end_a, end_b)
            intersection = max(0, intersection_end - intersection_start)

            return intersection / truth_len

        total_score = 0.0
        total_questions = len(student_results.search_results)

        for student_q in student_results.search_results:
            truth_q = truth_dict.get(student_q.question_id)
            if not truth_q or not truth_q.sources:
                continue

            found_sources = 0
            top_k_retrieved = student_q.retrieved_sources[:student_results.k]

            for truth_source in truth_q.sources:
                for student_source in top_k_retrieved:
                    if student_source.file_path == truth_source.file_path:
                        coverage = calculate_coverage(
                            student_source.first_character_index,
                            student_source.last_character_index,
                            truth_source.first_character_index,
                            truth_source.last_character_index
                        )
                        if coverage >= 0.05:
                            found_sources += 1
                            break

            question_score = found_sources / len(truth_q.sources)
            total_score += question_score

        recall = (total_score / total_questions) * 100 if total_questions > 0 else 0

        print(f"-> Total questions evaluated: {total_questions}")
        print(f"-> Recall@{student_results.k}: {recall:.2f}%")


if __name__ == "__main__":
    fire.Fire(RAGPipeline)
