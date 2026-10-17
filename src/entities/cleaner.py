import re


class Cleaner:
    # Единый список стоп-слов для кода и текста
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "in", "on", "of", "to", "for", "with",
        "and", "or", "not", "that", "this", "it", "as", "at", "by", "from", "can", "be",
        "how", "what", "why", "where", "when", "which", "who", "i", "my", "me", "do", "does", "did",
        "we", "you", "your", "def", "self", "class", "return", "import", "pass"
    }

    # Жестко рубим по всему, что не является буквой или цифрой
    SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
    # Идеальное разбиение CamelCase с учетом цифр
    CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Универсальный метод очистки: агрессивно дробит текст на атомы,
        удаляет мусор, стоп-слова и одиночные символы.
        """
        if not text:
            return ""

        parts = Cleaner.SPLIT_RE.split(text)
        tokens = []

        for part in parts:
            if not part:
                continue

            # Добиваем CamelCase внутри каждого куска
            for sub in Cleaner.CAMEL_RE.split(part):
                sub_lower = sub.lower()

                # Возвращаем фильтр коротких слов, так как агрессивный SPLIT_RE
                # порождает много одиночного мусора
                if len(sub_lower) < 2:
                    continue
                if sub_lower in Cleaner.STOP_WORDS:
                    continue

                tokens.append(sub_lower)

        return " ".join(tokens)

    # Перенаправляем все вызовы из BM25Builder в единый метод
    @staticmethod
    def clean_python(text: str) -> str:
        return Cleaner.clean_text(text)

    @staticmethod
    def clean_markdown(text: str) -> str:
        return Cleaner.clean_text(text)

    @staticmethod
    def clean_query(text: str) -> str:
        return Cleaner.clean_text(text)
