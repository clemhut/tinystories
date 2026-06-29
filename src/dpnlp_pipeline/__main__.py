from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.dpnlp_pipeline.config import MODEL_SPECS, PipelineDefaults, PipelinePaths
from src.dpnlp_pipeline.pipeline import (
    evaluate_gpt,
    evaluate_overlap,
    generate_all_for_prompt_set,
    generate_for_prompt_set,
    generate_from_checkpoint,
    report,
    train,
    train_all,
    write_default_prompts,
    write_evaluation_outputs,
    write_generated_outputs,
    write_loss_plot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyStories reproduction pipeline")
    parser.add_argument(
        "command",
        choices=[
            "describe",
            "write-prompts",
            "train",
            "train-all",
            "generate",
            "generate-for-prompt-set",
            "generate-all",
            "plot-loss",
            "evaluate-overlap",
            "evaluate-gpt",
            "report",
        ],
        help="Pipeline command",
    )
    parser.add_argument("--model-size", choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--evaluation-json")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-scale", action="store_true")
    args = parser.parse_args()

    if args.command == "describe":
        paths = PipelinePaths.from_repo_root(Path.cwd())
        payload = {
            "defaults": asdict(PipelineDefaults()),
            "model_specs": {name: asdict(spec) for name, spec in MODEL_SPECS.items()},
            "paths": {key: str(value) for key, value in asdict(paths).items()},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "write-prompts":
        print(write_default_prompts(Path.cwd()))
    elif args.command == "train":
        if not args.model_size:
            raise SystemExit("--model-size is required for train")
        print(
            json.dumps(
                train(
                    args.model_size,
                    repo_root=Path.cwd(),
                    max_steps=args.max_steps,
                    resume=args.resume,
                    save_every_steps=args.save_every_steps,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "train-all":
        print(
            json.dumps(
                train_all(
                    repo_root=Path.cwd(),
                    max_steps=args.max_steps,
                    resume=args.resume,
                    save_every_steps=args.save_every_steps,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "generate":
        if not args.model_size:
            raise SystemExit("--model-size is required for generate")
        if not args.prompt:
            raise SystemExit("--prompt is required for generate")
        print(
            json.dumps(
                generate_from_checkpoint(
                    args.model_size,
                    args.prompt,
                    repo_root=Path.cwd(),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "generate-for-prompt-set":
        if not args.model_size:
            raise SystemExit("--model-size is required for generate-for-prompt-set")
        outputs = generate_for_prompt_set(
            args.model_size,
            repo_root=Path.cwd(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        output_path = write_generated_outputs(outputs, f"{args.model_size}_outputs.json", repo_root=Path.cwd())
        print(json.dumps({"output_path": output_path, "rows": outputs}, indent=2, sort_keys=True))
    elif args.command == "generate-all":
        outputs = generate_all_for_prompt_set(
            repo_root=Path.cwd(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        output_path = write_generated_outputs(outputs, "all_model_outputs.json", repo_root=Path.cwd())
        print(json.dumps({"output_path": output_path, "rows": outputs}, indent=2, sort_keys=True))
    elif args.command == "plot-loss":
        if not args.model_size:
            raise SystemExit("--model-size is required for plot-loss")
        print(write_loss_plot(args.model_size, repo_root=Path.cwd(), log_scale=args.log_scale))
    elif args.command == "evaluate-overlap":
        if not args.evaluation_json:
            raise SystemExit("--evaluation-json is required for evaluate-overlap")
        outputs = json.loads(Path(args.evaluation_json).read_text())
        rows = evaluate_overlap(outputs, repo_root=Path.cwd())
        output_path = write_evaluation_outputs(
            rows,
            args.evaluation_json,
            result_dirname="overlap",
            result_suffix="overlap",
            repo_root=Path.cwd(),
        )
        print(json.dumps({"output_path": output_path, "rows": rows}, indent=2, sort_keys=True))
    elif args.command == "evaluate-gpt":
        if not args.evaluation_json:
            raise SystemExit("--evaluation-json is required for evaluate-gpt")
        outputs = json.loads(Path(args.evaluation_json).read_text())
        rows = evaluate_gpt(outputs)
        output_path = write_evaluation_outputs(
            rows,
            args.evaluation_json,
            result_dirname="gpt_eval",
            result_suffix="gpt_eval",
            repo_root=Path.cwd(),
        )
        print(json.dumps({"output_path": output_path, "rows": rows}, indent=2, sort_keys=True))
    elif args.command == "report":
        if not args.evaluation_json:
            raise SystemExit("--evaluation-json is required for report")
        evaluation_rows = json.loads(Path(args.evaluation_json).read_text())
        print(json.dumps(report(evaluation_rows, repo_root=Path.cwd()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
