import re
import string


class Cleaner:
    PYTON_STOP_WORDS = {
        "a", "an", "the", "is", "are", "in", "on", "of", "to", "for", "with",
        "def", "self", "class", "return", "import", "from", "pass", "and", "or", "not"
    }
    PYTON_TRANSLATOR = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    MD_STOP_WORDS = {
        "a", "an", "the", "is", "are", "in", "on", "of", "to", "for", "with",
        "and", "or", "not", "that", "this", "it", "as", "at", "by", "from", "can", "be",
        "how", "what", "why", "where", "when", "which", "who", "i", "my", "me", "do", "does", "did", "we", "you", "your"
    }

    # Для жесткой очистки Markdown заменяем на пробелы всё, кроме подчеркивания
    md_punct = string.punctuation.replace('_', '')
    MD_PUNCT_TRANSLATOR = str.maketrans(md_punct, ' ' * len(md_punct))

    @staticmethod
    def clean_python(text: str) -> str:
        """
        Двойная очистка кода: сохраняем монолитные слова + бьем CamelCase.
        """
        # 1. Ленивый проход (сохраняем слова без разбиения CamelCase)
        raw_words = [w for w in text.lower().translate(Cleaner.PYTON_TRANSLATOR).split()
                     if w not in Cleaner.PYTON_STOP_WORDS and len(w) > 1]

        # 2. Агрессивный проход (рубим CamelCase)
        text_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text_camel = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text_camel)

        clean_text = text_camel.lower().translate(Cleaner.PYTON_TRANSLATOR)
        split_words = [w for w in clean_text.split()
                       if w not in Cleaner.PYTON_STOP_WORDS and len(w) > 1]

        return " ".join(raw_words + split_words)

    @staticmethod
    def clean_markdown(text: str) -> str:
        """
        Двойная очистка текста: сохраняем версии целиком + рубим всё в пыль.
        """
        text_lower = text.lower()

        # 1. Ленивый проход (оставляем точки и дефисы для версий вроде vLLM-0.10.1)
        lazy_punct = ''.join(c for c in string.punctuation if c not in '_-.')
        lazy_trans = str.maketrans(lazy_punct, ' ' * len(lazy_punct))

        lazy_words = []
        for w in text_lower.translate(lazy_trans).split():
            w = w.strip('_-.')
            if w and w not in Cleaner.MD_STOP_WORDS and len(w) > 1:
                lazy_words.append(w)

        # 2. Агрессивный проход (рубим всё через пробелы)
        agg_words = []
        for w in text_lower.translate(Cleaner.MD_PUNCT_TRANSLATOR).split():
            w = w.strip('_-.')
            if w and w not in Cleaner.MD_STOP_WORDS and len(w) > 1:
                agg_words.append(w)

        return " ".join(lazy_words + agg_words)

    @staticmethod
    def clean_query(text: str) -> str:
        """
        Гибридная очистка запроса.
        Объединяет токены из обоих методов и ГАРАНТИРОВАННО вырезает все стоп-слова.
        """
        py_tokens = Cleaner.clean_python(text).split()
        md_tokens = Cleaner.clean_markdown(text).split()

        combined_tokens = set(py_tokens + md_tokens)
        all_stop_words = Cleaner.PYTON_STOP_WORDS.union(Cleaner.MD_STOP_WORDS)

        final_tokens = [w for w in combined_tokens if w not in all_stop_words]
        return " ".join(final_tokens)