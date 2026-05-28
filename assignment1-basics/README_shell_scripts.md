# Shell Scripts

Run these from the `assignment1-basics/` directory.

Python scripts under `scripts/` are documented in [`scripts/README.md`](scripts/README.md).

## `tests.sh`

List pytest files:

```bash
./tests.sh list
```

Run all pytest files:

```bash
./tests.sh all
```

Run selected pytest files by index:

```bash
./tests.sh 2
./tests.sh 2 6
```

Run with extra pytest args:

```bash
PYTEST_ARGS="-q -k test_transformer_lm --tb=short" ./tests.sh 2
```

Write a dated log under `logs/YYYYMMDD/`:

```bash
CHECK_LOG=1 DESC=train_bpe_speed_try3_mac_m5 PYTHON=.venv/bin/python ./tests.sh 7
```

Logs go under `logs/YYYYMMDD/`, for example:

`logs/20260524/pytest_train_bpe_speed_try3_mac_m5.log`

## `sanity_check.sh`

List sanity checks:

```bash
./sanity_check.sh list
```

Run all sanity checks:

```bash
PYTHON=.venv/bin/python ./sanity_check.sh all
```

Run selected sanity checks by index:

```bash
./sanity_check.sh 1
./sanity_check.sh 1 2
```

Write a dated log under `logs/YYYYMMDD/`:

```bash
CHECK_LOG=1 DESC=bpe_sanity_1 ./sanity_check.sh 1
```
