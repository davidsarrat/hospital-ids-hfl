# hospital-ids-hfl

`hospital-ids-hfl` demonstrates a three-layer hierarchical federated learning system using Flower and CIC-IDS2017 network-flow telemetry.

The "hospital" entities in this repository are healthcare network operators, not clinical departments and not patient-record holders. Each simulated hospital trains locally on its own CIC-IDS2017 network-flow partition. Regional ServerApps aggregate site updates using FedAvg and save regional checkpoints.

Each region is represented at the global level by a RegionGateway SuperNode. The gateway does not hold raw site telemetry. It loads the latest regional checkpoint and returns it to the global SuperLink as a model update, weighted by the total number of regional training examples.

The global ServerApp aggregates regional checkpoints into a global model. This demonstrates how local healthcare institutions, regional security hubs, and national/international hubs can collaborate on intrusion detection without centralizing network-flow rows.

## Data Framing

CIC-IDS2017 is not a medical dataset. It does not contain patient records, diagnoses, imaging, lab results, prescriptions, billing data, or any other clinical attributes.

The data consists of labeled network-flow records generated from packet captures with CICFlowMeter. In this demo, the labels mean:

```text
0 = BENIGN network flow
1 = MALICIOUS / ATTACK network flow
```

The hospital framing is therefore a cybersecurity scenario:

> Healthcare organizations collaboratively train an intrusion-detection model for their network traffic while keeping local network-flow telemetry inside each site.

Use "hospital network-flow telemetry" or "healthcare network traffic" when describing the local data boundary. Avoid interpreting this project as medical diagnosis, patient-risk prediction, or clinical decision support.

## Topology

```text
Healthcare-site SuperNodes
  -> Region EU / Region NA SuperLinks
    -> regional checkpoints
      -> RegionGateway SuperNodes
        -> Global SuperLink
          -> global checkpoint
```

The simulated healthcare-network sites are:

```text
region_eu: hospital_eu_01, hospital_eu_02, hospital_eu_03
region_na: hospital_na_01, hospital_na_02, hospital_na_03
```

## Dataset

The demo uses the Kaggle mirror of CIC-IDS2017:

```bash
kaggle datasets download -d chethuhn/network-intrusion-dataset -p data/raw --unzip
```

The authoritative dataset description is the Canadian Institute for Cybersecurity IDS 2017 page. The binary label mapping above is applied during preprocessing.

This project validates the federated-learning infrastructure pattern. It does not claim that CIC-IDS2017 alone is sufficient for a deployable healthcare-network IDS.

## Reproducible Data Pipeline

Configure Kaggle credentials first. The current Kaggle CLI accepts `KAGGLE_API_TOKEN`; legacy credentials can also be provided through `~/.kaggle/kaggle.json` or the standard `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

```bash
make install
make data
make partition SEED=123
```

`make install` installs the package plus the `docs` extra used to execute and render notebooks. Docker runtime images install the base package only, so Flower containers do not carry notebook tooling.

`make data` downloads and cleans CIC-IDS2017 into `data/processed/cicids_clean.parquet`. `make partition` creates six non-IID healthcare-network site folders:

```text
data/partitions/<hospital_id>/train.parquet
data/partitions/<hospital_id>/val.parquet
data/partitions/<hospital_id>/test.parquet
data/partitions/<hospital_id>/metadata.json
```

The default seed is `123`. The partitioning script fits imputation and standardization from the simulated site train splits and stores the scaler in `shared/preprocessing/scaler.json`.

## Running the Demo

Start the Flower deployment topology:

```bash
make compose
make flower-config
make up
```

Run three global hierarchical rounds, each with two regional FedAvg rounds:

```bash
make train GLOBAL_ROUNDS=3 REGIONAL_ROUNDS=2
```

Evaluate the final global checkpoint:

```bash
make eval GLOBAL_ROUNDS=3
```

Generate row-level predictions from the trained global checkpoint:

```bash
make predict GLOBAL_ROUNDS=3
```

For a compact end-to-end demo that launches a local Flower Deployment Runtime
(real SuperLinks, SuperNodes, and SuperExec processes on localhost), runs one
global round, evaluates the resulting checkpoint, generates sample predictions,
captures a verbose transcript, executes and refreshes the rendered notebooks,
and prints the global metrics summary:

```bash
make demo
```

`make demo` avoids Docker image rebuilds so the live demo focuses on the
training workflow. It writes `reports/demo_transcript.txt` with ANSI terminal
control codes stripped, explicit step separators, the launched Flower runtime
commands, submitted `flwr run` commands, checkpoint summaries, evaluation
output, prediction output, notebook rendering, and the final global metrics
table.

To exercise the containerized deployment route instead, run:

```bash
make demo-docker
```

That target generates a temporary hierarchical Compose file under `reports/`,
starts the Docker services, waits for the SuperLink control ports, and then
runs the same demo script against the already-running runtime.

Execute and render the educational notebooks after a local run:

```bash
make vignettes
```

Build the static handbook for GitHub Pages:

```bash
make pages
```

Expected outputs:

```text
shared/checkpoints/region_eu/round_<g>.pt
shared/checkpoints/region_na/round_<g>.pt
shared/checkpoints/global/round_<g>.pt
reports/metrics_summary.csv
reports/metrics_summary_global.csv
reports/predictions_hospital_eu_01.csv
reports/demo_transcript.txt
```

`shared/` is for checkpoints, metrics, and preprocessing metadata only. It should never contain raw CSV or parquet network-flow rows.

## Baselines

The primary architectural demo is PyTorch MLP + FedAvg. Optional comparison commands:

```bash
make centralized
make local
make flat
python scripts/evaluate_global_model.py \
  --checkpoint shared/checkpoints/global/round_3.pt \
  --flat-checkpoint shared/checkpoints/flat/round_3.pt \
  --local-checkpoints-dir shared/checkpoints/local_only
```

## Model

The model is a compact MLP:

```text
Linear(input_dim, 128) -> LayerNorm -> ReLU -> Dropout
Linear(128, 64) -> LayerNorm -> ReLU -> Dropout
Linear(64, 1)
```

It uses `BCEWithLogitsLoss`, sigmoid thresholding, and Flower `FedAvg` weighted by `"num-examples"`.

## Privacy Notes

Federated learning avoids moving raw rows, but model updates/checkpoints can still leak information. This demo validates orchestration, not full privacy hardening.

Local development uses `--insecure`. For serious deployment, enable TLS, SuperNode authentication, audit logging, secure aggregation, differential privacy where appropriate, and stricter container network segmentation.

## Flower Profiles

Flower `1.29.0` uses `~/.flwr/config.toml` for SuperLink connection profiles. Generate the local profiles with:

```bash
make flower-config
```

This writes:

```text
region-eu -> 127.0.0.1:19093
region-na -> 127.0.0.1:29093
global    -> 127.0.0.1:39093
flat      -> 127.0.0.1:49093
```

The orchestrator calls `flwr run . <profile> --stream --run-config ...` for every regional and global step.

## Rendered Walkthroughs

The notebooks in `notebooks/` are committed with executed outputs. They inspect the data pipeline, run safe dry-run commands where full training would be too expensive, show the running SuperLinks/SuperNodes/SuperExec services, expose gateway behavior, display checkpoint metadata, show evaluation summaries, and verify the `shared/` raw-data boundary.

The GitHub Pages handbook is generated into `docs/` and is intended to be served from:

```text
https://davidsarrat.github.io/hospital-ids-hfl/
```
