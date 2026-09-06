import math
import string
from collections import Counter
from src.entities.models import BM25Stat as BM25s
from src.utils import log_location
from src.utils.exits import error_exit
from collections import defaultdict


class BM25Methods:
    @staticmethod
    def despatch_bm25_stat(text: str, bm25s: BM25s) -> None:
        tokens = BM25Methods.tokenize(text)
        tokens_freq = BM25Methods._count_frequencies(tokens)
        word_quan = len(tokens)
        chunk_id = str(bm25s.N_total)

        BM25Methods._update_n_total(bm25s=bm25s)
        BM25Methods._update_total_dl(bm25s=bm25s, word_quan=word_quan)
        BM25Methods._update_df(tokens=tokens, bm25s=bm25s)
        BM25Methods._update_tf(tokens_freq=tokens_freq, chunk_id=chunk_id, bm25s=bm25s)
        BM25Methods._update_dl(word_quan=word_quan, chunk_id=chunk_id, bm25s=bm25s)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        custom_punctuation = string.punctuation.replace('_', '')
        spaces = ' ' * len(custom_punctuation)
        translator = str.maketrans(custom_punctuation, spaces)
        clean_text = text.translate(translator).lower()
        tokens = clean_text.split()
        return tokens

    @staticmethod
    def _count_frequencies(tokens: list[str]) -> dict[str, int]:
        return dict(Counter(tokens))

    @staticmethod
    def _update_n_total(bm25s: BM25s) -> None:
        bm25s.N_total += 1

    @staticmethod
    def _update_total_dl(bm25s: BM25s, word_quan: int) -> None:
        bm25s.total_dl += word_quan

    @staticmethod
    def _update_df(tokens: list[str], bm25s: BM25s) -> None:
        tokens_set = set(tokens)
        for word in tokens_set:
            bm25s.df[word] = bm25s.df.get(word, 0) + 1

    @staticmethod
    def _update_tf(tokens_freq: dict[str, int], chunk_id: str, bm25s: BM25s) -> None:
        for word, freq in tokens_freq.items():
            bm25s.tf[word][chunk_id] = freq

    @staticmethod
    def _update_dl(word_quan: int, chunk_id: str, bm25s: BM25s) -> None:
        bm25s.dl[chunk_id] = bm25s.dl.get(chunk_id, 0) + word_quan

    @staticmethod
    def calculate_avgdl(bm25s: BM25s) -> None:
        if bm25s.N_total == 0:
            error_exit(f">>> {log_location}\n"
                       f">>> Context: N_total=={bm25s.N_total}. Can't calculate 'avgdl (Average Document Length)'.")
        bm25s.avgdl = bm25s.total_dl / bm25s.N_total

    @staticmethod
    def calculate_idf(bm25s: BM25s) -> None:
        if not bm25s.df:
            error_exit(f">>> {log_location}\n"
                       f">>> Context: df parameter is empty. Can't calculate 'idf (Inverse Document Frequency)'.")

        for word, df_value in bm25s.df.items():
            numerator = bm25s.N_total - df_value + 0.5
            denominator = df_value + 0.5

            idf_value = math.log((numerator / denominator) + 1.0)

            bm25s.idf[word] = round(idf_value, 5)

    @staticmethod
    def get_top_k(query: str, bm25s: BM25s, k: int) -> list[tuple[str, float]]:
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
        res = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return res
