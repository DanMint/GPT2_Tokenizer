# GPT2_Tokenizer(BPE) Tokenizer From Scratch

A Python implementation of **Byte Pair Encoding (BPE)** built from scratch to better understand how modern Large Language Models convert text into tokens.

The project includes:

* A custom byte-level BPE tokenizer
* UTF-8 and Unicode support
* Text encoding and decoding
* Wrappers for **OpenAI `tiktoken`**
* Wrappers for **Hugging Face Tokenizers**
* Unit tests using `pytest`
* A Jupyter notebook explaining the tokenization process

---

## How It Works

The tokenizer follows the basic LLM tokenization pipeline:

```text
Text
  ↓
UTF-8 Bytes
  ↓
Byte Pair Encoding
  ↓
Token IDs
  ↓
Transformer / LLM
```

The custom tokenizer begins with the 256 possible byte values and repeatedly merges frequently occurring adjacent token pairs.

For example:

```text
A A B A A B
```

If `(A, A)` is the most frequent pair, BPE can assign it a new token:

```text
(A, A) → Token 256
```

The sequence then becomes:

```text
256 B 256 B
```

This process continues until the requested vocabulary size is reached.

---

## Project Structure

```text
GPT2_Tokenizer/
│
├── Tokenizers/
│   ├── BPE.py
│   ├── base.py
│   ├── Tiktoken.py
│   ├── huggingFaceTokenizer.py
│   └── __init__.py
│
├── tests/
│   └── test_bpe.py
│
├── Tokenization.ipynb
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DanMint/GPT2_Tokenizer.git
cd GPT2_Tokenizer
```

### 2. Create a virtual environment

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pytest tiktoken tokenizers jupyter
```

---

## Running the Custom BPE Tokenizer

Import the tokenizer:

```python
from Tokenizers import BPE
```

Create some training text:

```python
training_text = """
Large language models process text as sequences of tokens.
Tokenization converts text into numerical representations.
"""
```

Create and train the tokenizer:

```python
tokenizer = BPE(training_text)

tokenizer.train(vocab_size=300)
```

Encode text:

```python
text = "Hello world!"

tokens = tokenizer.encode(text)

print(tokens)
```

Decode the tokens back into text:

```python
decoded = tokenizer.decode(tokens)

print(decoded)
```

You can verify that encoding and decoding are lossless:

```python
assert tokenizer.decode(tokenizer.encode(text)) == text
```

---

## Running the Tests

From the root directory of the project:

```bash
python -m pytest tests -v
```

The tests verify that different types of text can be encoded and correctly decoded back to their original form.

Examples include:

* English text
* Numbers
* Punctuation
* Emojis
* Unicode characters
* Multilingual text

---

## Running the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Tokenization.ipynb
```

The notebook walks through concepts including:

```text
Unicode
   ↓
UTF-8
   ↓
Bytes
   ↓
BPE
   ↓
Tokens
```

---

## Using tiktoken

The project also includes a wrapper around OpenAI's `tiktoken` library:

```python
from Tokenizers import TikTokenTokenizer

tokenizer = TikTokenTokenizer()

tokens = tokenizer.encode("Hello world!")

print(tokens)
print(tokenizer.decode(tokens))
```

---

## Using Hugging Face

You can also use the Hugging Face tokenizer wrapper:

```python
from Tokenizers import HuggingFaceTokenizer

tokenizer = HuggingFaceTokenizer()

tokens = tokenizer.encode("Hello world!")

print(tokens)
print(tokenizer.decode(tokens))
```

---

## What I Learned

This project was built to better understand tokenization instead of treating tokenizer libraries as a black box.

Some of the main concepts explored include:

* Unicode and UTF-8
* Byte-level tokenization
* Byte Pair Encoding
* Vocabulary construction
* BPE merge rules
* Encoding and decoding
* How raw text becomes token IDs before entering a Transformer

---

## Technologies

* Python
* Byte Pair Encoding (BPE)
* Unicode / UTF-8
* OpenAI `tiktoken`
* Hugging Face Tokenizers
* Pytest
* Jupyter Notebook

---

## Note

This project implements the core concepts of **byte-level BPE for educational purposes**. It is not intended to be an exact reproduction of the complete production GPT-2 tokenizer.

The goal is to understand the algorithm and tokenization pipeline from first principles.

---

## Author

**Daniel Mints**

Ph.D. Student in Electrical & Computer Engineering
University of Massachusetts Dartmouth

GitHub: [DanMint](https://github.com/DanMint)
