from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from tqdm import tqdm

from src.dpnlp_pipeline.config import MODEL_SPECS, ModelSpec, PipelineDefaults, PipelinePaths
from src.dpnlp_pipeline.evaluation import build_ngram_counter, ngram_overlap_against_counts
from src.dpnlp_pipeline.generation import generate_completion
from src.dpnlp_pipeline.gpt_eval import grade_completion
from src.dpnlp_pipeline.io import load_checkpoint, load_json, save_checkpoint, save_json
from src.dpnlp_pipeline.modeling import build_model
from src.dpnlp_pipeline.prompts import DEFAULT_GPT_EVAL_PROMPTS
from src.dpnlp_pipeline.reporting import build_claim_summary
from src.dpnlp_pipeline.tokenizer import TokenizerSpec, build_tokenizer
from src.dpnlp_pipeline.training import train_model


def resolve_paths(repo_root: Path | None = None) -> PipelinePaths:
    return PipelinePaths.from_repo_root(repo_root or Path.cwd())


def prepare_tokenizer(paths: PipelinePaths, tokenizer_spec: TokenizerSpec | None = None):
    return build_tokenizer(tokenizer_spec, train_path=paths.train_path, vocab_path=paths.vocab_path)


def train(
    model_size: str,
    repo_root: Path | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    save_every_steps: int | None = 1000,
) -> dict:
    paths = resolve_paths(repo_root)
    defaults = PipelineDefaults()
    tokenizer = prepare_tokenizer(paths, TokenizerSpec(vocab_size=defaults.vocab_size))
    model_spec = MODEL_SPECS[model_size]
    training_state_path = paths.checkpoints_dir / f"{model_size}_training_state.pt"
    artifacts = train_model(
        model_spec,
        tokenizer,
        defaults,
        paths=paths,
        max_steps=max_steps,
        checkpoint_path=training_state_path,
        resume_from_checkpoint=training_state_path if resume else None,
        save_every_steps=save_every_steps,
    )

    checkpoint_path = paths.checkpoints_dir / f"{model_size}.pt"
    metadata = {
        "model_spec": asdict(model_spec),
        "defaults": asdict(defaults),
        "parameter_count": artifacts.parameter_count,
        "losses": artifacts.train_loss_history,
        "global_step": artifacts.global_step,
        "training_state_path": str(training_state_path),
    }
    save_checkpoint(checkpoint_path, artifacts.model, metadata)
    metrics_path = paths.reports_dir / f"{model_size}_train_metrics.json"
    save_json(metrics_path, metadata)
    return {"checkpoint_path": str(checkpoint_path), "metrics_path": str(metrics_path), **metadata}


def train_all(
    repo_root: Path | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    save_every_steps: int | None = 1000,
) -> dict[str, dict]:
    return {name: train(name, repo_root=repo_root, max_steps=max_steps, resume=resume, save_every_steps=save_every_steps) for name in MODEL_SPECS}


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0, device: str = "cpu") -> str:
    return generate_completion(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, device=device)


def load_trained_model(model_size: str, repo_root: Path | None = None):
    paths = resolve_paths(repo_root)
    checkpoint_payload = load_checkpoint(paths.checkpoints_dir / f"{model_size}.pt")
    model_spec = ModelSpec(**checkpoint_payload["metadata"]["model_spec"])
    defaults = PipelineDefaults(**checkpoint_payload["metadata"]["defaults"])
    tokenizer = prepare_tokenizer(paths, TokenizerSpec(vocab_size=defaults.vocab_size))
    model = build_model(tokenizer.get_vocab_size(), model_spec, defaults.context_length)
    model.load_state_dict(checkpoint_payload["state_dict"])
    return model, tokenizer, checkpoint_payload["metadata"]


def generate_from_checkpoint(
    model_size: str,
    prompt: str,
    repo_root: Path | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cpu",
) -> dict:
    model, tokenizer, metadata = load_trained_model(model_size, repo_root=repo_root)
    completion = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, device=device)
    return {
        "model_size": model_size,
        "prompt": prompt,
        "completion": completion,
        "parameter_count": metadata["parameter_count"],
    }


def generate_for_prompt_set(
    model_size: str,
    repo_root: Path | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cpu",
) -> list[dict]:
    prompts = load_prompt_set(repo_root=repo_root)
    return [
        generate_from_checkpoint(
            model_size,
            prompt,
            repo_root=repo_root,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        for prompt in prompts
    ]


def generate_all_for_prompt_set(
    repo_root: Path | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cpu",
) -> list[dict]:
    rows: list[dict] = []
    for model_size in MODEL_SPECS:
        rows.extend(
            generate_for_prompt_set(
                model_size,
                repo_root=repo_root,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
        )
    return rows


def evaluate_overlap(outputs: list[dict[str, str]], repo_root: Path | None = None) -> list[dict]:
    paths = resolve_paths(repo_root)
    train_text = paths.train_path.read_text(encoding="utf-8")
    valid_text = paths.valid_path.read_text(encoding="utf-8")
    train_ngrams = build_ngram_counter(train_text)
    print("Built train ngrams")
    valid_ngrams = build_ngram_counter(valid_text)
    print("Built test ngrams")
    rows: list[dict] = []
    for row in tqdm(outputs, desc="Evaluating candidates"):
        candidate = row["completion"]
        train_overlap = ngram_overlap_against_counts(train_ngrams, candidate)
        valid_overlap = ngram_overlap_against_counts(valid_ngrams, candidate)
        rows.append(
            {
                **row,
                "train_overlap": train_overlap,
                "valid_overlap": valid_overlap,
                "max_overlap": max(train_overlap, valid_overlap),
            }
        )
    return rows


def evaluate_gpt(
    outputs: list[dict[str, str]],
    prompts: list[str] | None = None,
    client=None,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    prompt_lookup = {prompt: prompt for prompt in (prompts or [])}
    rows: list[dict] = []
    for row in tqdm(outputs, desc="Evaluating completions with GPT"):
        prompt_prefix = prompt_lookup.get(row["prompt"], row["prompt"])
        scores = grade_completion(prompt_prefix, row["completion"], client=client, model=model)
        rows.append({**row, **scores})
    return rows


def write_default_prompts(repo_root: Path | None = None) -> str:
    paths = resolve_paths(repo_root)
    save_json(paths.prompts_path, DEFAULT_GPT_EVAL_PROMPTS)
    return str(paths.prompts_path)


def write_generated_outputs(
    outputs: list[dict],
    filename: str,
    repo_root: Path | None = None,
) -> str:
    paths = resolve_paths(repo_root)
    output_path = paths.outputs_dir / filename
    save_json(output_path, outputs)
    return str(output_path)


def write_evaluation_outputs(
    outputs: list[dict],
    evaluation_json: str | Path,
    result_dirname: str,
    result_suffix: str,
    repo_root: Path | None = None,
) -> str:
    paths = resolve_paths(repo_root)
    evaluation_path = Path(evaluation_json)
    stem = evaluation_path.stem
    model_prefix = stem[: -len("_outputs")] if stem.endswith("_outputs") else stem
    output_path = paths.artifacts_dir / result_dirname / f"{model_prefix}_{result_suffix}.json"
    save_json(output_path, outputs)
    return str(output_path)


def write_loss_plot(model_size: str, repo_root: Path | None = None, log_scale: bool = False) -> str:
    paths = resolve_paths(repo_root)
    metrics_path = paths.reports_dir / f"{model_size}_train_metrics.json"
    metrics = load_json(metrics_path)
    losses = metrics.get("losses", [])
    if not losses:
        raise RuntimeError(f"no losses found in {metrics_path}")

    suffix = "_loss_plot_log.png" if log_scale else "_loss_plot.png"
    output_path = paths.reports_dir / f"{model_size}{suffix}"
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=0.8)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"{model_size.upper()} model training loss" + (" (log scale)" if log_scale else ""))
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def report(evaluation_rows: list[dict], repo_root: Path | None = None) -> dict:
    paths = resolve_paths(repo_root)
    payload = {"claim_summary": build_claim_summary(evaluation_rows), "rows": evaluation_rows}
    output_path = paths.reports_dir / "claim_report.json"
    save_json(output_path, payload)
    return {"report_path": str(output_path), **payload}


def load_prompt_set(repo_root: Path | None = None) -> list[str]:
    paths = resolve_paths(repo_root)
    if paths.prompts_path.exists():
        payload = load_json(paths.prompts_path)
        return list(payload)
    return list(DEFAULT_GPT_EVAL_PROMPTS)
