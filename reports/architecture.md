# Architecture

`hospital-ids-hfl` uses one Flower federation per coordination layer:

- `region_eu`: three healthcare-site SuperNodes connected to the EU regional SuperLink.
- `region_na`: three healthcare-site SuperNodes connected to the NA regional SuperLink.
- `global`: two RegionGateway SuperNodes connected to the global SuperLink.

CIC-IDS2017 is network intrusion-detection telemetry, not medical or clinical data. The `hospital_*` identifiers represent simulated healthcare organizations operating local networks.

Site nodes mount only their own `data/partitions/<hospital_id>` directory. Region gateways mount `shared/checkpoints` and return regional checkpoints as model updates. They do not access raw site network-flow rows.

Weighted FedAvg is used at both levels:

```text
theta_region = sum_h (n_h / N_region) * theta_h
theta_global = sum_r (N_r / N_global) * theta_r
```

Every site and gateway returns `"num-examples"` so Flower can weight updates correctly.
