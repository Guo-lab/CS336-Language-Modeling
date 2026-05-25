`CHECK_LOG=1 DESC=train_bpe_speed_try3_mac_m5 PYTHON=.venv/bin/python ./tests.sh 7`

Logs go under `logs/YYYYMMDD/`, for example:

`logs/20260524/pytest_train_bpe_speed_try3_mac_m5.log`

`PYTHON=.venv/bin/python ./sanity_check.sh all`

Train experiment tokenizers:

The experiment script refuses to write into a non-empty `--out-dir` by default, so existing artifacts are not overwritten accidentally. Use a new output directory name for new runs, for example `artifacts/tokenizers_unoptimized/owt_valid_32k_try2`.

Downscaled TinyStories validation run with profiling:

`.venv/bin/python scripts/train_bpe_experiment.py --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizers_unoptimized/tinystories_valid_10k --profile --monitor-interval 30`

Full TinyStories run:

`.venv/bin/python scripts/train_bpe_experiment.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizers_unoptimized/tinystories_train_10k --profile --monitor-interval 30`

Full OpenWebText sample run:

`.venv/bin/python scripts/train_bpe_experiment.py --input data/owt_train.txt --vocab-size 32000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizers_unoptimized/owt_train_32k --profile --monitor-interval 30`

Tokenizer compression and throughput experiments:

`.venv/bin/python scripts/tokenizer_experiments.py --out-dir artifacts/tokenizer_experiments`

Also serialize train/dev token IDs as `uint16` NumPy arrays:

`.venv/bin/python scripts/tokenizer_experiments.py --out-dir artifacts/tokenizer_experiments --serialize`
