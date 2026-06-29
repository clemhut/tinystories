from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def save_checkpoint(path: Path, model, metadata: dict) -> None:
    ensure_parent_dir(path)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _tmp_checkpoint_path(path: Path) -> Path:
    return Path(str(path) + ".tmp")


def save_training_checkpoint(path: Path, model, optimizer, metadata: dict) -> None:
    ensure_parent_dir(path)
    tmp_path = _tmp_checkpoint_path(path)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        tmp_path,
    )
    for attempt in range(5):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                return
            time.sleep(0.25)


def load_training_checkpoint(path: Path) -> dict:
    candidates = [candidate for candidate in (path, _tmp_checkpoint_path(path)) if candidate.exists()]
    if not candidates:
        raise FileNotFoundError(path)

    payloads = []
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location="cpu")
        except Exception as exc:
            last_error = exc
            continue
        payloads.append((candidate, payload))

    if not payloads:
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(path)

    return max(
        payloads,
        key=lambda item: (
            int(item[1].get("metadata", {}).get("global_step", -1)),
            item[0].stat().st_mtime,
        ),
    )[1]
