from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from flwr.app import ArrayRecord

from hfl_cicids.task import TabularIDSNet


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    input_dim: int,
) -> TabularIDSNet:
    model = TabularIDSNet(input_dim=input_dim)
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)

    return model


def arrayrecord_from_checkpoint(
    checkpoint_path: str | Path,
    input_dim: int,
) -> ArrayRecord:
    model = load_model_from_checkpoint(checkpoint_path, input_dim)
    return ArrayRecord(model.state_dict())


def save_torch_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    save_metadata(checkpoint_path, extra_metadata or {})


def save_arrayrecord_checkpoint(
    arrays: ArrayRecord,
    checkpoint_path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = arrays.to_torch_state_dict()
    torch.save(state_dict, checkpoint_path)
    save_metadata(checkpoint_path, extra_metadata or {})


def metadata_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".metadata.json")


def save_metadata(checkpoint_path: str | Path, metadata: dict[str, Any]) -> None:
    path = metadata_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    path = metadata_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {path}")
    return json.loads(path.read_text())


def ensure_initial_checkpoint(
    checkpoint_path: str | Path,
    input_dim: int,
    overwrite: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists() and not overwrite:
        return checkpoint_path

    model = TabularIDSNet(input_dim=input_dim)
    save_torch_checkpoint(
        model,
        checkpoint_path,
        {
            "level": "global",
            "global_round": 0,
            "input_dim": input_dim,
            "initialized": True,
        },
    )
    return checkpoint_path
