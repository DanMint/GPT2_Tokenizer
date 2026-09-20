from abc import ABC, abstractmethod


class BaseTokenizer(ABC):

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass