import tiktoken

from .base import BaseTokenizer


class TikTokenTokenizer(BaseTokenizer):
    def __init__(self, encoding_name: str = "o200k_base"):
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def name(self) -> str:
        return f"tiktoken-{self._encoding_name}"

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)