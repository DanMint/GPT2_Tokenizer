from tokenizers import Tokenizer
from .base import BaseTokenizer

class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_name: str = "gpt2"):
        self._model_name = model_name
        self._tokenizer = Tokenizer.from_pretrained(model_name)

    @property
    def name(self) -> str:
        return f"huggingface-{self._model_name}"

    def encode(self, text: str) -> list[int]:
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)