from __future__ import annotations

import argparse

import numpy as np
import torch

from . import console
from .data import get_batch
from .experiment import ExperimentLogger
from .generation import decode
from .nn_utils import cross_entropy
from .optimizer import clip_gradients, get_lr_cosine_schedule
from .serialization import save_checkpoint
from .tokenizer import Tokenizer


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_learning_rate(args: argparse.Namespace, step: int) -> float:
    return get_lr_cosine_schedule(
        it=step,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
    )


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_tokens: np.ndarray,
        valid_tokens: np.ndarray,
        logger: ExperimentLogger,
        args: argparse.Namespace,
        tokenizer: Tokenizer | None = None,
        start_iter: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_tokens = train_tokens
        self.valid_tokens = valid_tokens
        self.logger = logger
        self.args = args
        self.tokenizer = tokenizer
        self.start_iter = start_iter

    def run(self) -> None:
        console.header("Starting training loop")
        console.info(
            "range",
            f"from iteration {self.start_iter} to {self.args.max_iters}",
        )

        for step in range(self.start_iter, self.args.max_iters):
            lr = get_learning_rate(self.args, step)
            train_loss = self.train_step(lr)
            iteration = step + 1
            self.maybe_log_train(iteration, train_loss, lr)
            self.maybe_eval(iteration)
            self.maybe_log_sample(iteration)
            self.maybe_save_checkpoint(iteration)

    def train_step(self, lr: float) -> float:
        set_learning_rate(self.optimizer, lr)
        x, y = get_batch(
            dataset=self.train_tokens,
            batch_size=self.args.batch_size,
            context_length=self.args.context_length,
            device=self.args.device,
        )
        logits = self.model(x)
        loss = cross_entropy(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip > 0:
            clip_gradients(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def estimate_loss(self, dataset: np.ndarray) -> float:
        was_training = self.model.training
        self.model.eval()

        losses = []
        for _ in range(self.args.eval_iters):
            x, y = get_batch(
                dataset=dataset,
                batch_size=self.args.batch_size,
                context_length=self.args.context_length,
                device=self.args.device,
            )
            logits = self.model(x)
            losses.append(cross_entropy(logits, y).item())

        if was_training:
            self.model.train()
        return sum(losses) / len(losses)

    def maybe_log_train(self, iteration: int, train_loss: float, lr: float) -> None:
        if iteration % self.args.log_every != 0 and iteration != 1:
            return
        self.logger.log_metrics(iteration, {"train_loss": train_loss, "lr": lr})
        console.info(
            f"step {iteration:>6}",
            f"train_loss={train_loss:.4f}, lr={lr:.6g}",
        )

    def maybe_eval(self, iteration: int) -> None:
        if iteration % self.args.eval_every != 0 and iteration != self.args.max_iters:
            return
        valid_loss = self.estimate_loss(self.valid_tokens)
        self.logger.log_metrics(iteration, {"valid_loss": valid_loss})
        console.warn(
            f"eval {iteration:>6}",
            f"valid_loss={valid_loss:.4f}",
        )

    def maybe_save_checkpoint(self, iteration: int) -> None:
        if iteration % self.args.save_every != 0 and iteration != self.args.max_iters:
            return
        checkpoint_path = self.logger.checkpoint_path(iteration)
        save_checkpoint(self.model, self.optimizer, iteration, checkpoint_path)
        print(console.dim(f"checkpoint saved: {checkpoint_path}"))

    def maybe_log_sample(self, iteration: int) -> None:
        if self.tokenizer is None or self.args.sample_prompt is None:
            return
        if self.args.sample_every <= 0:
            return
        if iteration % self.args.sample_every != 0 and iteration != self.args.max_iters:
            return

        was_training = self.model.training
        self.model.eval()

        prompt_token_ids = self.tokenizer.encode(self.args.sample_prompt)
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.args.device)
        eos_token_id = self.tokenizer.token_to_id.get(b"<|endoftext|>")
        generated = decode(
            model=self.model,
            prompt_token_ids=prompt,
            max_new_tokens=self.args.sample_max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=self.args.sample_temperature,
            top_p=self.args.sample_top_p,
            context_length=self.args.context_length,
        )

        generated_ids = generated[0].tolist()
        completion_ids = generated_ids[len(prompt_token_ids) :]
        completion = self.tokenizer.decode(completion_ids)
        self.logger.log_sample(iteration, self.args.sample_prompt, completion)
        console.warn(
            f"sample {iteration:>4}",
            repr((self.args.sample_prompt + completion)[:200]),
        )

        if was_training:
            self.model.train()
