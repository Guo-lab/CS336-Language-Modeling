# BPE Training Notes

TinyStories runs use:

```sh
.venv/bin/python scripts/train_bpe_experiment.py \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --profile
```

## Runs

| Run | Input | Output dir | Time | Max RSS | Vocab | Merges | Longest token |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Debug / validation | `data/TinyStoriesV2-GPT4-valid.txt` | `artifacts/tokenizers_unoptimized/tinystories_valid_10k` | `15.97s` | `146636800` bytes, about `140 MiB` | `10000` | `9743` | `" accomplishment"` |
| Full train | `data/TinyStoriesV2-GPT4-train.txt` | `artifacts/tokenizers_unoptimized/tinystories_train_10k` | `722.75s`, about `12.0 min` | `6519652352` bytes, about `6.1 GiB` | `10000` | `9743` | `" accomplishment"` |

The merge count is `9743` in both runs because `10000 - 256 byte tokens - 1 special token = 9743`.

## Longest Tokens

The longest learned tokens are common TinyStories-style English words, usually with a leading space. In the full training run, examples include:

- `" responsibility"`
- `" disappointment"`
- `" accomplishment"`
- `" granddaughter"`
- `" extraordinary"`
- `" congratulated"`
- `" determination"`

This makes sense because TinyStories is simple English prose, so common whole words can be frequent enough to become single BPE tokens. The special token `<|endoftext|>` appears as a single vocab item and does not show up as ordinary subword fragments.

## Profiling

| Run | Main bottleneck | Important cumulative times |
| --- | --- | --- |
| Debug / validation | `best_pair`, then pretoken counting | `best_pair`: `8.92s`; `build_pretoken_counts`: `6.41s`; `iter_pretokens`: `4.82s`; regex `findall`: `0.70s`; `merge_pair_in_state`: `0.55s` |
| Full train | pretoken counting / regex-byte conversion | `build_pretoken_counts`: `667.67s`; `iter_pretokens`: `503.29s`; regex `findall`: `72.87s`; `best_pair`: `42.52s`; `merge_pair_in_state`: `3.25s` |

The debug run is small enough that repeatedly scanning pair counts in `best_pair` is the largest single cost. On the full TinyStories training set, the bottleneck shifts strongly to pre-tokenization and pretoken counting, especially iterating through regex matches and converting each pretoken into byte tuples. The incremental merge update is relatively cheap in both runs.

## Deliverable Summary

Training the 10K TinyStories tokenizer on the full training file took about `12.0 min` and peaked at about `6.1 GiB` RSS. The longest token was `" accomplishment"`, which is reasonable because frequent full English words in TinyStories can become single BPE tokens.

Profiling showed that the full run was dominated by pre-tokenization / pretoken counting rather than the merge update. A natural next optimization would be multiprocessing the pre-tokenization stage by chunking on `<|endoftext|>` boundaries.

# OpenWebText Validation 32K BPE Tokenizer

Command:

```sh
.venv/bin/python scripts/train_bpe_experiment.py \
  --input data/owt_valid.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --out-dir artifacts/tokenizers_unoptimized/owt_valid_32k \
  --profile \
  --monitor-interval 30
```

Result:

- Input: `data/owt_valid.txt`
- Requested vocabulary size: `32000`
- Actual vocabulary size: `32000`
- Number of merges: `31743`
- Special token: `<|endoftext|>`
- Elapsed time: `3946.07s`, about `65.8 min`
- Max RSS: `3309355008` bytes, about `3.1 GiB`
- Longest token: `"----------------------------------------------------------------"`
- Longest token length: `64` bytes

The longest OWT validation tokens include long separator-like strings such as `"________________________________________________________________"`, `"----------------------------------------------------------------"`, `"................................"`, and `"————————"`, plus longer web/news words like `" telecommunications"`, `" disproportionately"`, `" environmentalists"`, and `" counterterrorism"`. This is plausible for web text, which contains formatting separators, repeated punctuation, technical terms, and broader vocabulary than TinyStories.

Profiling:

- `best_pair`: `4448.49s` cumulative in the profiled run
- built-in `max` inside `best_pair`: `4328.42s`
- `best_pair` lambda scoring: `2708.17s`
- `iter_pretokens`: `70.79s`
- `merge_pair_in_state`: `49.56s`
- `build_pretoken_counts`: `4.14s`

This run is strongly bottlenecked by best-pair selection. The current implementation scans all pair counts on every merge, and `32000` vocabulary size requires `31743` merge iterations. For OWT-style 32K tokenizers, optimizing `best_pair` selection is more important than optimizing pre-tokenization.
