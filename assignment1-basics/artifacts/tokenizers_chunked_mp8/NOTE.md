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
CS336_BPE_NUM_WORKERS=8 caffeinate -i .venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_train.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_chunked_mp8/owt_train_32k \
  --profile \
  --monitor-interval 30
```

```
CS336_BPE_NUM_WORKERS=4 caffeinate -i .venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_train.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_chunked_mp4/owt_train_32k \
  --profile \
  --monitor-interval 30
```
