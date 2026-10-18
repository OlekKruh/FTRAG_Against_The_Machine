import heapq
import math
from collections import Counter

from src.entities.cleaner import Cleaner
from src.entities.models import BM25Stat as BM25s
from src.utils.log_location import log_location
from src.utils.exits import error_exit
from collections import defaultdict


class BM25Methods:
    """
    Provides static methods to build and query a BM25 sparse retrieval index.
    """
    @staticmethod
    def despatch_bm25_stat(text: str, bm25s: BM25s, ext: str = "") -> None:
        """
        Processes a raw text chunk and updates the global BM25 statistics.

        Args:
            text (str): The raw text content of the chunk.
            bm25s (BM25s): The BM25 statistics dataclass to update.
            ext (str, optional): The file extension used to route text to the correct cleaner. Defaults to "".
        """
        tokens = BM25Methods.tokenize(text, ext)
        tokens_freq = BM25Methods._count_frequencies(tokens)
        word_quan = len(tokens)
        chunk_id = str(bm25s.N_total)

        BM25Methods._update_n_total(bm25s=bm25s)
        BM25Methods._update_total_dl(bm25s=bm25s, word_quan=word_quan)
        BM25Methods._update_df(tokens=tokens, bm25s=bm25s)
        BM25Methods._update_tf(tokens_freq=tokens_freq, chunk_id=chunk_id, bm25s=bm25s)
        BM25Methods._update_dl(word_quan=word_quan, chunk_id=chunk_id, bm25s=bm25s)

    @staticmethod
    def tokenize(text: str, ext: str = "") -> list[str]:
        """
        Routes text through the appropriate cleaner based on file extension,
        then splits it into tokens.

        Args:
            text (str): The raw text to tokenize.
            ext (str, optional): The file extension (e.g., '.py', '.md'). Defaults to "".

        Returns:
            list[str]: A list of cleaned tokens.
        """
        if ext == ".py":
            clean_str = Cleaner.clean_python(text)
        elif ext == ".md":
            clean_str = Cleaner.clean_markdown(text)
        else:
            clean_str = Cleaner.clean_query(text)

        return clean_str.split()

    @staticmethod
    def _count_frequencies(tokens: list[str]) -> dict[str, int]:
        """
        Counts the frequency of each token in a given list.

        Args:
            tokens (list[str]): A list of string tokens.

        Returns:
            dict[str, int]: A dictionary mapping each token to its frequency count.
        """
        return dict(Counter(tokens))

    @staticmethod
    def _update_n_total(bm25s: BM25s) -> None:
        """
        Increments the total document count in the BM25 statistics.

        Args:
            bm25s (BM25s): The BM25 statistics object.
        """
        bm25s.N_total += 1

    @staticmethod
    def _update_total_dl(bm25s: BM25s, word_quan: int) -> None:
        """
        Adds the current document's length to the total corpus length.

        Args:
            bm25s (BM25s): The BM25 statistics object.
            word_quan (int): The number of tokens in the current document.
        """
        bm25s.total_dl += word_quan

    @staticmethod
    def _update_df(tokens: list[str], bm25s: BM25s) -> None:
        """
        Updates the Document Frequency (DF) for each unique token in the document.

        Args:
            tokens (list[str]): A list of tokens from a single document.
            bm25s (BM25s): The BM25 statistics object.
        """
        tokens_set = set(tokens)
        for word in tokens_set:
            bm25s.df[word] = bm25s.df.get(word, 0) + 1

    @staticmethod
    def _update_tf(tokens_freq: dict[str, int], chunk_id: str, bm25s: BM25s) -> None:
        """
        Records the Term Frequency (TF) of each token for a specific document chunk.

        Args:
            tokens_freq (dict[str, int]): A dictionary of token frequencies in the current document.
            chunk_id (str): The unique identifier (index) of the current chunk.
            bm25s (BM25s): The BM25 statistics object.
        """
        for word, freq in tokens_freq.items():
            bm25s.tf[word][chunk_id] = freq

    @staticmethod
    def _update_dl(word_quan: int, chunk_id: str, bm25s: BM25s) -> None:
        """
        Records the length of a specific document chunk.

        Args:
            word_quan (int): The number of tokens in the current document.
            chunk_id (str): The unique identifier (index) of the current chunk.
            bm25s (BM25s): The BM25 statistics object.
        """
        bm25s.dl[chunk_id] = bm25s.dl.get(chunk_id, 0) + word_quan

    @staticmethod
    def calculate_avgdl(bm25s: BM25s) -> None:
        """
        Calculates the Average Document Length (avgdl) for the entire corpus.
        Must be called after all documents have been processed.

        Args:
            bm25s (BM25s): The BM25 statistics object containing N_total and total_dl.

        Raises:
            SystemExit: If N_total is 0 (i.e., no documents were indexed).
        """
        if bm25s.N_total == 0:
            error_exit(f"{log_location()}\n"
                       f">>> Context: N_total=={bm25s.N_total}. Can't calculate 'avgdl (Average Document Length)'.")
        bm25s.avgdl = bm25s.total_dl / bm25s.N_total

    @staticmethod
    def calculate_idf(bm25s: BM25s) -> None:
        """
        Calculates the Inverse Document Frequency (IDF) for all unique tokens.
        Applies smoothing to prevent division by zero and negative IDF scores.
        Must be called after all documents have been processed.

        Args:
            bm25s (BM25s): The BM25 statistics object to update with calculated IDF values.

        Raises:
            SystemExit: If the document frequency (df) dictionary is empty.
        """
        if not bm25s.df:
            error_exit(f"{log_location()}\n"
                       f">>> Context: df parameter is empty. Can't calculate 'idf (Inverse Document Frequency)'.")

        for word, df_value in bm25s.df.items():
            numerator = bm25s.N_total - df_value + 0.5
            denominator = df_value + 0.5

            idf_value = math.log((numerator / denominator) + 1.0)

            bm25s.idf[word] = round(idf_value, 5)

    @staticmethod
    def get_top_k(query: str, bm25s: BM25s, k: int) -> list[tuple[str, float]]:
        """
        Scores and retrieves the top-k document chunks for a given query based on BM25.

        Args:
            query (str): The raw search query entered by the user.
            bm25s (BM25s): The populated BM25 statistics object.
            k (int): The number of top-scoring chunks to return.

        Returns:
            list[tuple[str, float]]: A list containing tuples of chunk IDs and their
                                     corresponding BM25 scores, sorted in descending order.
        """
        query_tokens = BM25Methods.tokenize(query)
        uniq_query_tokens = set(query_tokens)

        scores: dict[str, float] = defaultdict(float)

        for value in uniq_query_tokens:
            if value not in bm25s.idf:
                continue

            weight_idf = bm25s.idf[value]
            chunks_dict_tf = bm25s.tf[value]

            for chunk_id, tf in chunks_dict_tf.items():
                dl = bm25s.dl[chunk_id]
                numerator = tf * (bm25s.k1 + 1)
                denominator = tf + bm25s.k1 * (1 - bm25s.b + bm25s.b * (dl / bm25s.avgdl))
                score = weight_idf * (numerator / denominator)
                scores[chunk_id] += score
        res = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return res
