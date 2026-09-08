from pathlib import Path
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMEngine:
    """
    Handles local inference using a Hugging Face causal language model.

    Attributes:
        model_name (str): The name or path of the Hugging Face model.
        max_new_tokens (int): The maximum number of tokens to generate.
        cache_path (Path): The local directory path for storing downloaded models.
        tokenizer (Any): The loaded tokenizer instance.
        model (Any): The loaded causal language model instance.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initializes the language model engine configuration.

        Args:
            model_name (str): The Hugging Face model identifier to load.
        """
        self.model_name: str = model_name
        self.max_new_tokens: int = 256
        self.cache_path: Path = Path(__file__).resolve().parents[2] / "models_cache"

        self.tokenizer: Any = None
        self.model: Any = None

    def load_model(self) -> None:
        """
        Loads the tokenizer and model weights into memory, utilizing a local cache.
        """
        print(f"-> Loading the tokenizer from {self.cache_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_path
        )

        print("-> Loading model weights (may take some time on first run)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_path,
            device_map="auto"
        )
        print("-> Model loaded successfully!")

    def generate_answer(self, context: str, query: str) -> str:
        """
        Generates a response grounded in the provided context using the model's chat template.

        Args:
            context (str): The retrieved text chunks to serve as the knowledge base.
            query (str): The user's original question.

        Returns:
            str: The generated answer text, excluding the input prompt.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The model is not loaded. Call load_model() first.")

        # 1. Strict format: list with role and content keys
        messages = [
            {
                "role": "system",
                "content": "Provide a comprehensive answer, using provided context. "
                           "If the answer is not in the context, output exactly and only: 'Sorry, I don't know'."
            },
            {
                "role": "user",
                "content": "Use this text to find the answer:\n"
                           f"{context}\n\n"
                           f"Query: {query}"
            }
        ]

        # 2. Token generation and passing to model
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
            enable_thinking=False,
        ).to(self.model.device)

        # 3. Answer generation
        tensors = self.model.generate(
            **tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        # 4. Decoding answer
        prompt_length = tokens.input_ids.shape[1]
        res = self.tokenizer.decode(
            tensors[0][prompt_length:],
            skip_special_tokens=True
        )

        return str(res)
