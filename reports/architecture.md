# Architecture

`hospital-ids-hfl` uses one Flower federation per coordination layer:

- `region_eu`: three hospital SuperNodes connected to the EU regional SuperLink.
- `region_na`: three hospital SuperNodes connected to the NA regional SuperLink.
- `global`: two RegionGateway SuperNodes connected to the global SuperLink.

Hospital nodes mount only their own `data/partitions/<hospital_id>` directory. Region gateways mount `shared/checkpoints` and return regional checkpoints as model updates. They do not access raw hospital rows.

Weighted FedAvg is used at both levels:

```text
theta_region = sum_h (n_h / N_region) * theta_h
theta_global = sum_r (N_r / N_global) * theta_r
```

Every hospital and gateway returns `"num-examples"` so Flower can weight updates correctly.
