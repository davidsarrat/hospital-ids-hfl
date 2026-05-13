from __future__ import annotations

from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from hfl_cicids.checkpointing import save_arrayrecord_checkpoint
from hfl_cicids.task import TabularIDSNet

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    level = str(context.run_config["level"])

    if level == "regional":
        _run_regional_serverapp(grid, context)
    elif level == "global":
        _run_global_serverapp(grid, context)
    else:
        raise ValueError(f"Unknown FL level: {level}")


def _initial_arrays(context: Context) -> ArrayRecord:
    input_dim = int(context.run_config.get("input-dim", 0))
    if input_dim <= 0:
        raise ValueError("Run config must define input-dim")

    model = TabularIDSNet(input_dim=input_dim)

    init_checkpoint = str(context.run_config.get("init-checkpoint", ""))
    if init_checkpoint:
        path = Path(init_checkpoint)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    return ArrayRecord(model.state_dict())


def _run_regional_serverapp(grid: Grid, context: Context) -> None:
    """Aggregate hospital SuperNode updates inside one region."""

    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    min_train_nodes = int(context.run_config["min-train-nodes"])

    strategy = FedAvg(
        fraction_train=float(context.run_config["fraction-train"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=int(context.run_config["min-evaluate-nodes"]),
        min_available_nodes=int(context.run_config["min-available-nodes"]),
        weighted_by_key="num-examples",
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=_initial_arrays(context),
        num_rounds=num_rounds,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "global-round": int(context.run_config["global-round"]),
            }
        ),
    )

    output_checkpoint = Path(str(context.run_config["output-checkpoint"]))
    save_arrayrecord_checkpoint(
        result.arrays,
        output_checkpoint,
        extra_metadata={
            "level": "regional",
            "region": str(context.run_config["region"]),
            "global_round": int(context.run_config["global-round"]),
            "regional_rounds": num_rounds,
            "input_dim": int(context.run_config["input-dim"]),
            "num_examples": int(context.run_config.get("region-num-examples", 1)),
            "weighted_by_key": "num-examples",
        },
    )


def _run_global_serverapp(grid: Grid, context: Context) -> None:
    """Aggregate regional gateway updates at the global hub."""

    min_train_nodes = int(context.run_config["min-train-nodes"])

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=0,
        min_available_nodes=min_train_nodes,
        weighted_by_key="num-examples",
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=_initial_arrays(context),
        num_rounds=1,
        train_config=ConfigRecord(
            {
                "global-round": int(context.run_config["global-round"]),
            }
        ),
    )

    output_checkpoint = Path(str(context.run_config["output-checkpoint"]))
    save_arrayrecord_checkpoint(
        result.arrays,
        output_checkpoint,
        extra_metadata={
            "level": "global",
            "global_round": int(context.run_config["global-round"]),
            "input_dim": int(context.run_config["input-dim"]),
            "num_examples": int(context.run_config.get("global-num-examples", 1)),
            "weighted_by_key": "num-examples",
        },
    )
