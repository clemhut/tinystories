from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from src.dpnlp_pipeline.config import ModelSpec, PipelineDefaults, PipelinePaths
from src.dpnlp_pipeline.datasets import build_dataset
from src.dpnlp_pipeline.generation import generate_completion
from src.dpnlp_pipeline.io import load_training_checkpoint, save_training_checkpoint
from src.dpnlp_pipeline.modeling import build_model
from tqdm import tqdm

TRAINING_PREVIEW_PROMPT = "Once upon a time, there was a little girl named Lucy. She"
TRAINING_PREVIEW_INTERVAL = 1_000
TRAINING_PREVIEW_MAX_NEW_TOKENS = 128


@dataclass
class TrainingArtifacts:
    model: nn.Module
    train_loss_history: list[float]
    parameter_count: int
    global_step: int


def _serializable_defaults(defaults: PipelineDefaults) -> dict:
    if is_dataclass(defaults):
        return asdict(defaults)
    return {
        "context_length": defaults.context_length,
        "train_epochs": defaults.train_epochs,
    }


def _training_checkpoint_metadata(
    model_spec: ModelSpec,
    defaults: PipelineDefaults,
    parameter_count: int,
    losses: list[float],
    global_step: int,
    epoch: int,
    batch_index: int,
    completed: bool,
) -> dict:
    return {
        "model_spec": asdict(model_spec),
        "defaults": _serializable_defaults(defaults),
        "parameter_count": parameter_count,
        "losses": losses,
        "global_step": global_step,
        "epoch": epoch,
        "batch_index": batch_index,
        "completed": completed,
    }


def maybe_print_training_preview(
    model,
    tokenizer,
    step: int,
    device: str,
    interval: int = TRAINING_PREVIEW_INTERVAL,
    prompt: str = TRAINING_PREVIEW_PROMPT,
    max_new_tokens: int = TRAINING_PREVIEW_MAX_NEW_TOKENS,
) -> None:
    if step % interval != 0:
        return

    was_training = model.training
    model.eval()
    completion = generate_completion(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        device=device,
    )
    print(f"[batch {step}] {completion}")
    if was_training:
        model.train()


def train_model(
    model_spec: ModelSpec,
    tokenizer,
    defaults: PipelineDefaults,
    paths: PipelinePaths,
    batch_size: int = 8,
    max_steps: int | None = None,
    device: str | None = None,
    checkpoint_path: Path | None = None,
    resume_from_checkpoint: Path | None = None,
    save_every_steps: int | None = None,
) -> TrainingArtifacts:
    train_dataset = build_dataset(tokenizer, path=paths.train_path, block_size=defaults.context_length, train=True)
    model = build_model(tokenizer.get_vocab_size(), model_spec, defaults.context_length)
    print(model.get_param_number())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.device(device))

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    losses: list[float] = []
    global_step = 0
    start_epoch = 0
    start_batch_index = 0

    if resume_from_checkpoint is not None and resume_from_checkpoint.exists():
        checkpoint_payload = load_training_checkpoint(resume_from_checkpoint)
        metadata = checkpoint_payload["metadata"]
        model.load_state_dict(checkpoint_payload["state_dict"])
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        losses = list(metadata.get("losses", []))
        global_step = int(metadata.get("global_step", len(losses)))
        start_epoch = int(metadata.get("epoch", 0))
        start_batch_index = int(metadata.get("batch_index", -1)) + 1

        if metadata.get("completed"):
            return TrainingArtifacts(
                model=model,
                train_loss_history=losses,
                parameter_count=model.get_param_number(),
                global_step=global_step,
            )

    if max_steps is not None and global_step >= max_steps:
        return TrainingArtifacts(model=model, train_loss_history=losses, parameter_count=model.get_param_number(), global_step=global_step)

    def save_state(epoch: int, batch_index: int, completed: bool) -> None:
        if checkpoint_path is None:
            return
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            metadata=_training_checkpoint_metadata(
                model_spec=model_spec,
                defaults=defaults,
                parameter_count=model.get_param_number(),
                losses=losses,
                global_step=global_step,
                epoch=epoch,
                batch_index=batch_index,
                completed=completed,
            ),
        )

    last_epoch = start_epoch
    last_batch_index = start_batch_index - 1

    for epoch in range(start_epoch, defaults.train_epochs):
        generator = torch.Generator()
        generator.manual_seed(epoch)
        dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        for batch_index, (input_ids, labels) in enumerate(tqdm(dataloader, desc="Training model...")):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue

            input_ids = input_ids.to(torch.device(device))
            labels = labels.to(torch.device(device))

            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids)
            loss = loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            global_step += 1
            last_epoch = epoch
            last_batch_index = batch_index
            maybe_print_training_preview(model, tokenizer, step=global_step, device=device)

            if save_every_steps is not None and save_every_steps > 0 and global_step % save_every_steps == 0:
                save_state(epoch, batch_index, completed=False)

            if max_steps is not None and global_step >= max_steps:
                save_state(epoch, batch_index, completed=False)
                return TrainingArtifacts(
                    model=model,
                    train_loss_history=losses,
                    parameter_count=model.get_param_number(),
                    global_step=global_step,
                )

    save_state(last_epoch, last_batch_index, completed=True)
    return TrainingArtifacts(model=model, train_loss_history=losses, parameter_count=model.get_param_number(), global_step=global_step)
