import uuid
from collections import defaultdict
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    A minimal source: a slice of a file by character offsets.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class IndexFile(BaseModel):
    """
    A container for storing all indexed document chunks.

    Used to serialize and deserialize the entire corpus of MinimalSource
    objects when saving to or loading from the disk.
    """
    chunks: list[MinimalSource] = Field(default_factory=list)


class BM25Stat(BaseModel):
    """
    Storage of statistical data and parameters for the BM25 algorithm.

    Attributes:
        N_total (int): Total quantity of chunks in the corpus.
        total_dl (int): Total quantity of words across all chunks to calculate the average document length.
        df (dict[str, int]): Document Frequency. The quantity of documents in which a specific word appears.
        k1 (float): Saturation parameter (usually 1.5). Limits the effect of multiple occurrences of a word.
        b (float): Length normalization parameter (usually 0.75). Penalizes for document length.
        tf (dict[str, dict[str, int]]): Term Frequency. Inverted index in the format `word -> {document_ID: frequency}`.
        dl (dict[str, int]): Dictionary of lengths for each specific document, measured in words.
        avgdl (float): Average document length
        idf (dict[str, float]): Inverse Document Frequency
    """
    N_total: int = 0
    total_dl: int = 0
    df: dict[str, int] = Field(default_factory=dict)

    k1: float = 1.5
    b: float = 0.75

    tf: dict[str, dict[str, int]] = Field(default_factory=lambda: defaultdict(dict))
    dl: dict[str, int] = Field(default_factory=dict)

    avgdl: float = 0.0
    idf: dict[str, float] = Field(default_factory=dict)


class UnansweredQuestion(BaseModel):
    """
    A question that has not been answered yet.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    A question with a ground-truth answer and its source slices.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    A dataset of RAG questions (answered or unanswered).
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    Search results for a single question.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Search results enriched with a generated answer.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Full batch of search results emitted by the student system.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Batch of search results plus generated answers.
    """
    search_results: List[MinimalAnswer]
    k: int
