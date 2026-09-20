from Tokenizers import BPE
from Tokenizers import TikTokenTokenizer
from Tokenizers import HuggingFaceTokenizer

import pytest

train_text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has survived not only many decades, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised thanks to these sheets and more recently with desktop publishing software like Aldus PageMaker and Microsoft Word including versions of Lorem Ipsum."
test_text = "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like)."

def test_bpe_round_trip():
    tokenizer = BPE(train_text)

    tokenizer.train(vocab_size = 300)

    tokens = tokenizer.encode(test_text)
    output_text = tokenizer.decode(tokens)

    assert test_text == output_text

def test_tiktoken_round_trip():
    tokenizer = TikTokenTokenizer()

    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    assert decoded == test_text


def test_HuggingFaceTokenizers_round_trip():
    tokenizer = HuggingFaceTokenizer()
    
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    assert decoded == test_text

@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello world",
        "123456789",
        "Hello 👋🌎",
        "你好世界",
        "مرحبا بالعالم",
        "C++ Python Rust",
        "newlines\nand\ttabs",
        "punctuation!!!???",
    ]
)
def test_bpe_round_trip_cases(text):
    tokenizer = BPE()
    tokenizer.train(text * 10, 300)

    tokens = tokenizer.encode(text)

    assert tokenizer.decode(tokens) == text