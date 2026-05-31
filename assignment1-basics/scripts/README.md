# Python Scripts

Run these from the `assignment1-basics/` directory.

## `train_bpe_experiment.py`

Train experiment tokenizers. The script refuses to write into a non-empty `--out-dir` by default, so existing artifacts are not overwritten accidentally.

Downscaled TinyStories validation run with profiling:

```bash
.venv/bin/python scripts/train_bpe_experiment.py --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizers_unoptimized/tinystories_valid_10k --profile --monitor-interval 30
```

Full TinyStories run:

```bash
.venv/bin/python scripts/train_bpe_experiment.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizers_unoptimized/tinystories_train_10k --profile --monitor-interval 30
```

Full OpenWebText sample run:

```bash
.venv/bin/python scripts/train_bpe_experiment.py --input data/owt_train.txt --vocab-size 32000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizers_unoptimized/owt_train_32k --profile --monitor-interval 30
```

## `tokenizer_experiments.py`

Run tokenizer compression and throughput experiments. By default this samples TinyStories and OpenWebText validation documents, loads the TinyStories 10K and OpenWebText 32K tokenizers, and evaluates the configured tokenizer/dataset combinations.

```bash
.venv/bin/python scripts/tokenizer_experiments.py --out-dir artifacts/tokenizer_experiments
```

## `transformer_accounting.py`

Print Transformer LM parameter and FLOPs accounting tables in the terminal.

```bash
.venv/bin/python scripts/transformer_accounting.py
```

## `sgd_lr_tuning.py`

Run the toy SGD learning-rate tuning experiment from Section 4.2.1.

```bash
.venv/bin/python scripts/sgd_lr_tuning.py
```

Change the tested learning rates or number of steps:

```bash
.venv/bin/python scripts/sgd_lr_tuning.py --lrs 10 100 1000 --steps 10
```

## `adamw_accounting.py`

Print AdamW training memory/FLOPs accounting for the GPT-2 XL-shaped model.

```bash
.venv/bin/python scripts/adamw_accounting.py
```
