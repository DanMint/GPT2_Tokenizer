import csv
import os
import statistics
import time

from Tokenizers import BPE
from Tokenizers import TikTokenTokenizer
from Tokenizers import HuggingFaceTokenizer


TRAINING_TEXT = """
The quick brown fox jumps over the lazy dog.
Hello world. This is training text for a byte pair encoding tokenizer.
Python, C++, machine learning, artificial intelligence, and tokenization.
Unicode characters are supported: 你好世界 مرحبا بالعالم 👋🌎.
""" * 1000


BENCHMARK_TEXT = """
Lorem Ipsum is simply dummy text of the printing and typesetting industry.
The quick brown fox jumps over the lazy dog.
Python, C++, Rust, machine learning, artificial intelligence, tokenization.
Hello 👋🌎. 你好世界. مرحبا بالعالم.
""" * 10000


RESULTS_PATH = "results/benchmark_results.csv"
RUNS = 10


def benchmark_encode(
    tokenizer,
    text: str,
    runs: int = RUNS
) -> dict:

    times = []

    # Warm-up
    tokenizer.encode(text)

    tokens = []

    for _ in range(runs):

        start = time.perf_counter()

        tokens = tokenizer.encode(text)

        end = time.perf_counter()

        times.append(end - start)

    avg_time = statistics.mean(times)
    median_time = statistics.median(times)

    std_dev = (
        statistics.stdev(times)
        if len(times) > 1
        else 0.0
    )

    input_bytes = len(text.encode("utf-8"))
    input_mb = input_bytes / (1024 * 1024)

    throughput = (
        input_mb / avg_time
        if avg_time > 0
        else 0.0
    )

    bytes_per_token = (
        input_bytes / len(tokens)
        if tokens
        else 0.0
    )

    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "average_time_seconds": avg_time,
        "median_time_seconds": median_time,
        "std_dev_seconds": std_dev,
        "throughput_mb_per_second": throughput,
        "bytes_per_token": bytes_per_token,
    }


def benchmark_decode(
    tokenizer,
    text: str,
    tokens: list,
    runs: int = RUNS
) -> dict:

    times = []

    # Warm-up
    tokenizer.decode(tokens)

    decoded = ""

    for _ in range(runs):

        start = time.perf_counter()

        decoded = tokenizer.decode(tokens)

        end = time.perf_counter()

        times.append(end - start)

    round_trip_correct = decoded == text

    if not round_trip_correct:
        print("\nWARNING: round-trip mismatch detected")

        print("Original sample:")
        print(repr(text[:200]))

        print("\nDecoded sample:")
        print(repr(decoded[:200]))

    avg_time = statistics.mean(times)
    median_time = statistics.median(times)

    std_dev = (
        statistics.stdev(times)
        if len(times) > 1
        else 0.0
    )

    input_bytes = len(text.encode("utf-8"))
    input_mb = input_bytes / (1024 * 1024)

    throughput = (
        input_mb / avg_time
        if avg_time > 0
        else 0.0
    )

    return {
        "average_time_seconds": avg_time,
        "median_time_seconds": median_time,
        "std_dev_seconds": std_dev,
        "throughput_mb_per_second": throughput,
        "round_trip_correct": round_trip_correct,
    }


def save_results(results: list[dict]) -> None:

    os.makedirs("results", exist_ok=True)

    fieldnames = [
        "tokenizer",
        "input_size_mb",
        "token_count",
        "bytes_per_token",
        "round_trip_correct",
        "encode_average_time_seconds",
        "encode_median_time_seconds",
        "encode_std_dev_seconds",
        "encode_throughput_mb_per_second",
        "decode_average_time_seconds",
        "decode_median_time_seconds",
        "decode_std_dev_seconds",
        "decode_throughput_mb_per_second",
    ]

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


def print_results(
    name: str,
    encode_results: dict,
    decode_results: dict
) -> None:

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nEncoding")

    print(
        f"Tokens:              "
        f"{encode_results['token_count']:,}"
    )

    print(
        f"Average time:        "
        f"{encode_results['average_time_seconds']:.6f} s"
    )

    print(
        f"Median time:         "
        f"{encode_results['median_time_seconds']:.6f} s"
    )

    print(
        f"Std deviation:       "
        f"{encode_results['std_dev_seconds']:.6f} s"
    )

    print(
        f"Throughput:          "
        f"{encode_results['throughput_mb_per_second']:.2f} MB/s"
    )

    print(
        f"Bytes per token:     "
        f"{encode_results['bytes_per_token']:.3f}"
    )

    print("\nDecoding")

    print(
        f"Average time:        "
        f"{decode_results['average_time_seconds']:.6f} s"
    )

    print(
        f"Median time:         "
        f"{decode_results['median_time_seconds']:.6f} s"
    )

    print(
        f"Std deviation:       "
        f"{decode_results['std_dev_seconds']:.6f} s"
    )

    print(
        f"Throughput:          "
        f"{decode_results['throughput_mb_per_second']:.2f} MB/s"
    )

    print(
        f"Round-trip correct:  "
        f"{decode_results['round_trip_correct']}"
    )


def main():

    print("Preparing tokenizers...")

    custom_bpe = BPE(TRAINING_TEXT)
    custom_bpe.train(300)

    tiktoken_tokenizer = TikTokenTokenizer()

    huggingface_tokenizer = HuggingFaceTokenizer()

    tokenizers = [
        ("Custom BPE", custom_bpe),
        ("TikToken", tiktoken_tokenizer),
        ("Hugging Face", huggingface_tokenizer),
    ]

    input_size_bytes = len(
        BENCHMARK_TEXT.encode("utf-8")
    )

    input_size_mb = (
        input_size_bytes / (1024 * 1024)
    )

    print(
        f"Benchmark input size: "
        f"{input_size_mb:.2f} MB"
    )

    all_results = []

    for name, tokenizer in tokenizers:

        print(f"\nBenchmarking: {name}")

        encode_results = benchmark_encode(
            tokenizer,
            BENCHMARK_TEXT,
        )

        decode_results = benchmark_decode(
            tokenizer,
            BENCHMARK_TEXT,
            encode_results["tokens"],
        )

        print_results(
            name,
            encode_results,
            decode_results,
        )

        result_row = {
            "tokenizer": name,

            "input_size_mb":
                input_size_mb,

            "token_count":
                encode_results["token_count"],

            "bytes_per_token":
                encode_results["bytes_per_token"],

            "round_trip_correct":
                decode_results["round_trip_correct"],

            "encode_average_time_seconds":
                encode_results[
                    "average_time_seconds"
                ],

            "encode_median_time_seconds":
                encode_results[
                    "median_time_seconds"
                ],

            "encode_std_dev_seconds":
                encode_results[
                    "std_dev_seconds"
                ],

            "encode_throughput_mb_per_second":
                encode_results[
                    "throughput_mb_per_second"
                ],

            "decode_average_time_seconds":
                decode_results[
                    "average_time_seconds"
                ],

            "decode_median_time_seconds":
                decode_results[
                    "median_time_seconds"
                ],

            "decode_std_dev_seconds":
                decode_results[
                    "std_dev_seconds"
                ],

            "decode_throughput_mb_per_second":
                decode_results[
                    "throughput_mb_per_second"
                ],
        }

        all_results.append(result_row)

    save_results(all_results)

    print(
        f"\nBenchmark results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()