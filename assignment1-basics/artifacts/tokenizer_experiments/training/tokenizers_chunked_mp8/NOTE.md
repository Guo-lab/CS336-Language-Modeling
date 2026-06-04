```
CS336_BPE_NUM_WORKERS=1 .venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_valid.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_chunked_serial/owt_valid_32k \
  --profile \
  --monitor-interval 30
```


```
CS336_BPE_NUM_WORKERS=8 .venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_valid.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_chunked_mp8/owt_valid_32k \
  --profile \
  --monitor-interval 30
```


```
CS336_BPE_NUM_WORKERS=1 caffeinate -i .venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_train.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_chunked_serial/owt_train_32k \
  --profile \
  --monitor-interval 30
```

```
CS336_BPE_NUM_WORKERS=8 CS336_BPE_CHUNK_BYTES=134217728 caffeinate -i .venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_train.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_chunked_mp8/owt_train_32k \
  --profile \
  --monitor-interval 300
```



# OWT Train 32K BPE

Command:

```bash
CS336_BPE_NUM_WORKERS=8 .venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_train.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_chunked_mp8/owt_train_32k \
  --profile \
  --monitor-interval 30
```

Result:

- Input: `data/owt_train.txt` (~11GB).
- Elapsed: `4344.97s` = `72.4min` = `1.21h`.
- Max RSS: `9550479360` bytes = `8.89GiB`.
- Final vocab size: `32000`.
- Number of merges: `31743`.
- Longest token: 64 bytes, a repeated mojibake-looking sequence (`ÃÂ...`), which is plausible for noisy web text.

The run completed well within the assignment resource limits (`<=12h`, `<=100GB RAM`). The output files are complete: `vocab.json`, `merges.json`, `merges.txt`, `summary.json`, `profile.stats`, and `profile_top.txt`.

Profile note: the bottleneck has moved to the merge-update loop, especially `remove`, `add`, `count_pairs_in_pretoken`, and `push_pair`. Pretokenization is no longer the dominant bottleneck in this optimized run.
