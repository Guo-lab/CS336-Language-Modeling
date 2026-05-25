from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from cs336_basics.tokenizer import Tokenizer


SPECIAL_TOKEN = "<|endoftext|>"
PILE_BYTES = 825 * 1000**3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokenizer compression and throughput experiments.")
    parser.add_argument("--tinystories-data", default="data/TinyStoriesV2-GPT4-valid.txt", type=Path)
    parser.add_argument("--owt-data", default="data/owt_valid.txt", type=Path)
    parser.add_argument("--tinystories-train-data", default="data/TinyStoriesV2-GPT4-train.txt", type=Path)
    parser.add_argument("--owt-train-data", default="data/owt_train.txt", type=Path)
    parser.add_argument("--tinystories-tokenizer", default="artifacts/tokenizers_chunked_mp8/tinystories_train_10k", type=Path)
    parser.add_argument("--owt-tokenizer", default="artifacts/tokenizers_chunked_mp8/owt_train_32k", type=Path)
    parser.add_argument("--out-dir", default="artifacts/tokenizer_experiments", type=Path)
    parser.add_argument("--num-docs", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--throughput-bytes", default=20_000_000, type=int)
    parser.add_argument("--serialize", action="store_true", help="Also encode train/dev datasets to uint16 .npy files.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tinystories_tokenizer = load_tokenizer(args.tinystories_tokenizer)
    owt_tokenizer = load_tokenizer(args.owt_tokenizer)

    tinystories_docs = sample_documents(args.tinystories_data, args.num_docs, args.seed)
    owt_docs = sample_documents(args.owt_data, args.num_docs, args.seed)

    results = {
        "tinystories_with_tinystories_tokenizer": compression_result(
            tinystories_tokenizer, tinystories_docs
        ),
        "owt_with_owt_tokenizer": compression_result(owt_tokenizer, owt_docs),
        "owt_with_tinystories_tokenizer": compression_result(tinystories_tokenizer, owt_docs),
    }
    results["throughput"] = throughput_result(
        owt_tokenizer, read_prefix(args.owt_data, args.throughput_bytes)
    )

    if args.serialize:
        encoded_dir = args.out_dir / "encoded"
        encoded_dir.mkdir(exist_ok=True)
        results["serialized"] = {
            "tinystories_train": serialize_dataset(
                tinystories_tokenizer,
                args.tinystories_train_data,
                encoded_dir / "tinystories_train.npy",
            ),
            "tinystories_valid": serialize_dataset(
                tinystories_tokenizer,
                args.tinystories_data,
                encoded_dir / "tinystories_valid.npy",
            ),
            "owt_train": serialize_dataset(
                owt_tokenizer,
                args.owt_train_data,
                encoded_dir / "owt_train.npy",
            ),
            "owt_valid": serialize_dataset(
                owt_tokenizer,
                args.owt_data,
                encoded_dir / "owt_valid.npy",
            ),
        }

    write_json(args.out_dir / "summary.json", results)
    write_answers(args.out_dir / "answers.md", results)
    print(json.dumps(results, indent=2))


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_files(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.json"),
        special_tokens=[SPECIAL_TOKEN],
    )


def iter_documents(path: Path, special_token: str = SPECIAL_TOKEN) -> Iterable[str]:
    buffer = ""
    with path.open(encoding="utf-8") as f:
        while chunk := f.read(1024 * 1024):
            buffer += chunk
            parts = buffer.split(special_token)
            yield from parts[:-1]
            buffer = parts[-1]
    if buffer:
        yield buffer


def sample_documents(path: Path, num_docs: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sample: list[str] = []
    for seen, doc in enumerate(iter_documents(path), start=1):
        if len(sample) < num_docs:
            sample.append(doc)
        else:
            idx = rng.randrange(seen)
            if idx < num_docs:
                sample[idx] = doc
    return sample


def compression_result(tokenizer: Tokenizer, docs: list[str]) -> dict[str, float | int]:
    text = SPECIAL_TOKEN.join(docs)
    num_bytes = len(text.encode("utf-8"))
    num_tokens = len(tokenizer.encode(text))
    return {
        "num_documents": len(docs),
        "bytes": num_bytes,
        "tokens": num_tokens,
        "bytes_per_token": num_bytes / num_tokens,
    }


def read_prefix(path: Path, max_bytes: int) -> str:
    with path.open("rb") as f:
        return f.read(max_bytes).decode("utf-8", errors="ignore")


def throughput_result(tokenizer: Tokenizer, text: str) -> dict[str, float | int]:
    num_bytes = len(text.encode("utf-8"))
    start = time.perf_counter()
    num_tokens = len(tokenizer.encode(text))
    elapsed_seconds = time.perf_counter() - start
    bytes_per_second = num_bytes / elapsed_seconds
    pile_seconds = PILE_BYTES / bytes_per_second
    return {
        "bytes": num_bytes,
        "tokens": num_tokens,
        "elapsed_seconds": elapsed_seconds,
        "bytes_per_second": bytes_per_second,
        "pile_825gb_seconds": pile_seconds,
        "pile_825gb_hours": pile_seconds / 3600,
    }


def serialize_dataset(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> dict[str, int | str]:
    num_tokens = count_dataset_tokens(tokenizer, input_path)
    encoded = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(num_tokens,),
    )

    offset = 0
    for ids in iter_encoded_documents(tokenizer, input_path):
        chunk = np.asarray(ids, dtype=np.uint16)
        encoded[offset : offset + len(chunk)] = chunk
        offset += len(chunk)

    encoded.flush()
    return {"path": str(output_path), "tokens": num_tokens, "dtype": "uint16"}


def count_dataset_tokens(tokenizer: Tokenizer, input_path: Path) -> int:
    return sum(len(ids) for ids in iter_encoded_documents(tokenizer, input_path))


def iter_encoded_documents(tokenizer: Tokenizer, input_path: Path) -> Iterable[list[int]]:
    special_id = tokenizer.token_to_id[SPECIAL_TOKEN.encode("utf-8")]
    first = True
    for doc in iter_documents(input_path):
        if not first:
            yield [special_id]
        yield tokenizer.encode(doc)
        first = False


def write_answers(path: Path, results: dict) -> None:
    ts = results["tinystories_with_tinystories_tokenizer"]["bytes_per_token"]
    owt = results["owt_with_owt_tokenizer"]["bytes_per_token"]
    cross = results["owt_with_tinystories_tokenizer"]["bytes_per_token"]
    throughput = results["throughput"]["bytes_per_second"]
    pile_hours = results["throughput"]["pile_825gb_hours"]

    path.write_text(
        "\n".join(
            [
                "# Tokenizer Experiments",
                "",
                f"(a) TinyStories tokenizer: {ts:.3f} bytes/token. OpenWebText tokenizer: {owt:.3f} bytes/token.",
                f"(b) The TinyStories tokenizer gets {cross:.3f} bytes/token on the OpenWebText sample, compared with {owt:.3f} bytes/token for the OpenWebText tokenizer; it usually fragments web text more because its vocabulary was learned from simpler story text.",
                f"(c) Throughput is about {throughput:,.0f} bytes/second. At that rate, tokenizing 825GB would take about {pile_hours:.2f} hours.",
                "(d) uint16 is appropriate because the 10K and 32K vocabularies both have fewer than 2^16 token IDs, so each token fits in 16 bits while using half the space of uint32.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
