from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from hfl_cicids.metrics import evaluate_binary_classifier
from hfl_cicids.task import (
    TabularIDSNet,
    infer_input_dim,
    load_dataloaders,
    train_one_client,
)

app = ClientApp()


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _msg_config(msg: Message) -> dict[str, Any]:
    if "config" not in msg.content:
        return {}
    return dict(msg.content["config"])


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train a healthcare-site model or return a regional checkpoint as gateway update."""

    role = str(context.node_config.get("role", "hospital"))

    if role == "region-gateway":
        return _train_region_gateway(msg, context)

    return _train_hospital(msg, context)


def _train_hospital(msg: Message, context: Context) -> Message:
    data_dir = Path(str(context.node_config["data-dir"]))
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    config = _msg_config(msg)
    lr = float(config.get("lr", context.run_config["learning-rate"]))

    input_dim = infer_input_dim(data_dir)
    model = TabularIDSNet(input_dim=input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    train_loader, val_loader, _ = load_dataloaders(data_dir, batch_size)
    loss = train_one_client(
        model=model,
        train_loader=train_loader,
        epochs=local_epochs,
        lr=lr,
        device=_device(),
    )

    val_metrics = evaluate_binary_classifier(model, val_loader, _device())

    metrics = {
        "train_loss": float(loss),
        "num-examples": int(len(train_loader.dataset)),
        "val_f1": float(val_metrics["eval_f1"]),
        "val_roc_auc": float(val_metrics["eval_roc_auc"]),
        "val_auprc": float(val_metrics["eval_auprc"]),
        "val_false_positive_rate": float(val_metrics["eval_false_positive_rate"]),
        "val_false_negative_rate": float(val_metrics["eval_false_negative_rate"]),
    }

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(metrics),
            }
        ),
        reply_to=msg,
    )


def _train_region_gateway(msg: Message, context: Context) -> Message:
    """Return an already aggregated regional checkpoint to the global hub.

    The region gateway is a parent-layer client. It does not train on network-flow rows and
    should only mount shared checkpoints, never site partition directories.
    """

    region = str(context.node_config["region"])
    checkpoint_root = Path(str(context.node_config["checkpoint-root"]))
    config = _msg_config(msg)
    global_round = int(config.get("global-round", context.run_config.get("global-round", 1)))

    ckpt = checkpoint_root / region / f"round_{global_round}.pt"
    meta = ckpt.with_suffix(".metadata.json")

    if not ckpt.exists():
        raise FileNotFoundError(f"Regional checkpoint not found: {ckpt}")

    if not meta.exists():
        raise FileNotFoundError(f"Regional checkpoint metadata not found: {meta}")

    metadata = json.loads(meta.read_text())
    input_dim = int(metadata["input_dim"])
    num_examples = int(metadata["num_examples"])

    model = TabularIDSNet(input_dim=input_dim)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    metrics = {
        "num-examples": num_examples,
        "regional_val_f1": float(metadata.get("val_f1", 0.0)),
        "regional_val_roc_auc": float(metadata.get("val_roc_auc", 0.0)),
        "regional_val_auprc": float(metadata.get("val_auprc", 0.0)),
    }

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(metrics),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate a received model on a healthcare-site validation or test split."""

    role = str(context.node_config.get("role", "hospital"))

    if role == "region-gateway":
        return _evaluate_region_gateway(msg, context)

    data_dir = Path(str(context.node_config["data-dir"]))
    batch_size = int(context.run_config["batch-size"])
    split = str(context.run_config.get("eval-split", "val"))

    input_dim = infer_input_dim(data_dir)
    model = TabularIDSNet(input_dim=input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    _, val_loader, test_loader = load_dataloaders(data_dir, batch_size)
    loader = test_loader if split == "test" else val_loader

    metrics = evaluate_binary_classifier(model, loader, _device())

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )


def _evaluate_region_gateway(msg: Message, context: Context) -> Message:
    region = str(context.node_config["region"])
    checkpoint_root = Path(str(context.node_config["checkpoint-root"]))
    config = _msg_config(msg)
    global_round = int(config.get("global-round", context.run_config.get("global-round", 1)))
    ckpt = checkpoint_root / region / f"round_{global_round}.pt"
    meta = ckpt.with_suffix(".metadata.json")

    if not meta.exists():
        metrics = {"num-examples": 1, "eval_f1": 0.0, "eval_roc_auc": 0.0, "eval_auprc": 0.0}
    else:
        metadata = json.loads(meta.read_text())
        metrics = {
            "num-examples": int(metadata.get("num_examples", 1)),
            "eval_f1": float(metadata.get("val_f1", 0.0)),
            "eval_roc_auc": float(metadata.get("val_roc_auc", 0.0)),
            "eval_auprc": float(metadata.get("val_auprc", 0.0)),
        }

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )
