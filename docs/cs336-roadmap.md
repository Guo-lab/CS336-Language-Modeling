# CS336 One-Month Roadmap

Goal: learn CS336 by building, testing, and explaining the core systems. The loop is:

1. Read just enough lecture material.
2. Implement the assignment piece.
3. Run tests.
4. Write down what broke, why, and what the system is doing.

This is not a course to merely watch. It becomes real when the code passes tests and you can explain the failure modes.

## Time Budget

Target pace: 3-5 hours per day, 5-6 days per week.

If a day is short, prioritize assignments over lectures. Lectures are the map; assignments are the terrain.

## Week 0-1: Orientation And Assignment 1 Core

Read:

- `lectures/lecture_01.py`
- `lectures/lecture_02.py`
- `assignment1-basics/README.md`
- `cs336_assignment1_basics.pdf`

Do:

- Set up `uv`.
- Run the failing tests once:

```sh
cd assignment1-basics
uv run pytest
```

Implementation order:

1. Tokenizer / BPE.
2. Tensor basics.
3. Linear, embedding, RMSNorm.
4. Attention and RoPE.

Milestone:

- You can explain tokenization, embeddings, attention shapes, and why RoPE changes attention scores.
- A meaningful subset of Assignment 1 tests passes.

## Week 2: Finish Assignment 1

Focus:

1. MLP / SwiGLU.
2. Transformer block.
3. Transformer LM.
4. Optimizer / AdamW.
5. Serialization.
6. Data loading.
7. Tiny training run.

Run frequently:

```sh
cd assignment1-basics
uv run pytest
```

Milestone:

- Assignment 1 tests pass or only a small known set remains.
- You can train a tiny LM and describe the full path from text to loss.

Write down:

- What tensors flow through the model?
- Where does the causal mask apply?
- What exactly does AdamW update?
- What does checkpointing need to save?

## Week 3: Assignment 2 Systems

Read:

- `assignment2-systems/README.md`
- `cs336_assignment2_systems.pdf`
- `lectures/lecture_06.py`
- `lectures/lecture_07.py`

Focus:

1. Attention optimization.
2. Profiling mindset.
3. DDP.
4. FSDP.
5. Sharded optimizer.

Run:

```sh
cd assignment2-systems
uv run pytest
```

Milestone:

- You understand where compute, memory, and communication costs come from.
- You can explain DDP vs FSDP in terms of what is replicated, sharded, communicated, and synchronized.

For each module, write one sentence:

- Bottleneck:
- Communication:
- Memory saved:
- Main correctness risk:

## Week 4: Scaling, Data, And Lecture Consolidation

Read:

- `assignment3-scaling/README.md`
- `cs336_assignment3_scaling.pdf`
- `assignment4-data/README.md`
- `lectures/lecture_10.py`
- `lectures/lecture_12.py`
- Recent PDF lectures as needed.

Focus:

1. Compute-optimal training.
2. Loss scaling laws.
3. Data/model/compute tradeoffs.
4. Experiment budgeting.
5. Data quality, filtering, deduplication, and contamination.
6. Inference and evaluation.

Milestone:

- You can explain why scaling is a resource allocation problem, not just a bigger-model problem.
- You can describe the role of data quality and evaluation in the language model lifecycle.

## Daily Template

Use this routine:

1. 20-40 min: read lecture or assignment handout.
2. 2-4 hr: implement one tested unit.
3. 20 min: run tests and inspect failures.
4. 10 min: write a small note.

Daily note format:

```md
Date:
Topic:
Tests run:
What passed:
What failed:
Main concept learned:
One thing still confusing:
```

## Before Each Implementation Section

Before starting a new implementation section, answer these three questions:

1. What part of the PDF should I read, and where should I stop?
2. What must I understand before implementing the first thing in this section?
3. What notes should I write down before coding?

Use these questions as a gate. If the answers are fuzzy, read less broadly and more carefully before writing code.

## Priority Rules

- If time is tight, do Assignment 1 deeply before skimming later topics.
- Do not read three lectures in a row without writing code.
- Do not move past a failed test until you know whether the issue is shape, math, dtype, masking, initialization, or API mismatch.
- Keep a small scratch notebook for tensor shapes.
- Prefer tiny experiments over vague understanding.

## Final One-Month Outcome

By the end of the month, aim to have:

- Assignment 1 substantially complete.
- Assignment 2 core systems understood and partially implemented.
- Scaling/data topics summarized in your own words.
- A personal map of the language model stack:
  tokenizer -> data loader -> transformer -> optimizer -> training loop -> distributed systems -> scaling -> data quality -> inference/eval.
