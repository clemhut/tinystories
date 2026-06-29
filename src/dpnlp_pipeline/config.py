from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_params: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int


@dataclass(frozen=True)
class PipelineDefaults:
    context_length: int = 512
    attention_window: int = 256
    vocab_size: int = 10_000
    generation_temperature: float = 0.0
    gpt_eval_temperature: float = 1.0
    gpt_eval_samples_per_prompt: int = 10
    gpt_eval_prompt_count: int = 50
    train_epochs: int = 1


@dataclass(frozen=True)
class PipelinePaths:
    repo_root: Path
    data_dir: Path
    train_path: Path
    valid_path: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    prompts_path: Path
    outputs_dir: Path
    vocab_path: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "PipelinePaths":
        data_dir = repo_root / "datasets" / "tinystories"
        artifacts_dir = repo_root / "artifacts" / "dpnlp_pipeline"
        return cls(
            repo_root=repo_root,
            data_dir=data_dir,
            train_path=data_dir / "TinyStories-train.txt",
            valid_path=data_dir / "TinyStories-valid.txt",
            artifacts_dir=artifacts_dir,
            checkpoints_dir=artifacts_dir / "checkpoints",
            reports_dir=artifacts_dir / "reports",
            prompts_path=artifacts_dir / "gpt_eval_prompts.json",
            outputs_dir=artifacts_dir / "outputs",
            vocab_path=artifacts_dir / "top_10k_vocab.json",
        )


MODEL_SPECS: dict[str, ModelSpec] = {
    "xs": ModelSpec("xs", 1_000_000, d_model=96, n_layers=3, n_heads=8, d_ff=128),
    "small": ModelSpec("small", 3_000_000, d_model=128, n_layers=7, n_heads=8, d_ff=512),
    "medium": ModelSpec("medium", 10_000_000, d_model=224, n_layers=10, n_heads=8, d_ff=896),
    "large": ModelSpec("large", 29_000_000, d_model=512, n_layers=10, n_heads=8, d_ff=1024),
    "xl": ModelSpec("xl", 56_000_000, d_model=544, n_layers=12, n_heads=8, d_ff=2048),
}
