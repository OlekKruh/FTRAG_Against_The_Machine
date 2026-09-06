from pathlib import Path
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMEngine:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name: str = model_name
        self.max_new_tokens: int = 256
        self.max_context_length: int = 2000
        self.cache_path: Path = Path(__file__).resolve().parents[2] / "models_cache"

        self.tokenizer: Any = None
        self.model: Any = None

    def load_model(self) -> None:
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
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The model is not loaded. Call load_model() first.")

        # 1. Strict format: list with role and content keys
        massages = [
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
            massages,
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
