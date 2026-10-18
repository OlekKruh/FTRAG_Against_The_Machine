import re


class Cleaner:
    """
    Provides unified text cleaning and tokenization methods for the BM25 index.

    Applies aggressive splitting on non-alphanumeric characters and CamelCase
    boundaries, while filtering out common stop words and single-character noise.
    """
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "in", "on", "of", "to", "for", "with",
        "and", "or", "not", "that", "this", "it", "as", "at", "by", "from", "can", "be",
        "how", "what", "why", "where", "when", "which", "who", "i", "my", "me", "do", "does", "did",
        "we", "you", "your", "def", "self", "class", "return", "import", "pass"
    }

    SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
    CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Aggressively splits text into atomic tokens using regex boundaries,
        converts them to lowercase, and removes stop words and single characters.

        Args:
            text (str): The raw text to be cleaned and tokenized.

        Returns:
            str: A space-separated string of cleaned tokens.
        """
        if not text:
            return ""

        parts = Cleaner.SPLIT_RE.split(text)
        tokens = []

        for part in parts:
            if not part:
                continue

            for sub in Cleaner.CAMEL_RE.split(part):
                sub_lower = sub.lower()

                if len(sub_lower) < 2:
                    continue
                if sub_lower in Cleaner.STOP_WORDS:
                    continue

                tokens.append(sub_lower)

        return " ".join(tokens)

    @staticmethod
    def clean_python(text: str) -> str:
        """
        Cleans and tokenizes Python source code.
        Routes to the unified clean_text method.

        Args:
            text (str): Raw Python code or docstrings.

        Returns:
            str: Cleaned tokens.
        """
        return Cleaner.clean_text(text)

    @staticmethod
    def clean_markdown(text: str) -> str:
        """
        Cleans and tokenizes Markdown documentation.
        Routes to the unified clean_text method.

        Args:
            text (str): Raw Markdown text.

        Returns:
            str: Cleaned tokens.
        """
        return Cleaner.clean_text(text)

    @staticmethod
    def clean_query(text: str) -> str:
        """
        Cleans and tokenizes the user's search query.
        Routes to the unified clean_text method to ensure token symmetry
        with the indexed documents.

        Args:
            text (str): The raw search query.

        Returns:
            str: Cleaned tokens.
        """
        return Cleaner.clean_text(text)
