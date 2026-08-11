# P2 continuous multitask baseline

P2 reads the frozen `_model_dataset_v6_continuous_multitask` fact layer. It
does not require Step13 events for training and never rewrites Step16 files.

## Training contract

- `samples/sample_index.csv` is authoritative: 24-hour history, six hourly
  horizons, and Step16's absolute `TRAIN / VALIDATION / TEST` boundaries.
- Each window exposes outlet Q and Z independently. Z supervision is
  `delta_Z(t+h) = Z(t+h) - Z(t0)` and is valid only when both the exact forecast
  origin `Z(t0)` and `Z(t+h)` are observed.
- Q loss scale is the population standard deviation of unique valid TRAIN
  outlet-Q target hours in each graph, floored at 1 m3/s. Delta-Z loss scale is
  the population standard deviation of valid TRAIN delta-Z supervision for
  each target station, floored at 0.01 m. Both maps are stored in checkpoint
  config and `*_target_scales.json`; evaluation rejects mismatched maps.
- Q and delta-Z valid elements have separate denominators. A missing task adds
  zero for that task and never creates NaN. Existing Q:Z and auxiliary
  multi-task weights, optimizer, learning rate, batch size, model and physics
  settings are inherited unchanged.
- TRAIN weighted sampling multiplies inverse graph sample frequency by a
  bounded response score built only from TRAIN Q anomaly/change and absolute
  delta-Z. It samples with replacement without deleting ordinary-flow windows
  and never becomes a loss weight. VALIDATION and TEST always use ordinary
  deterministic sampling. Set `train_sampling.enabled: false` for the ablation.
- P2 runs exactly 100 epochs, validates every epoch, writes the best-validation
  checkpoint and an explicit epoch-100 checkpoint. Validation cannot stop it.

## TEST flood events

`scripts/build_v6_test_flood_events.py` rebuilds evaluation events from current
graph membership, outlets, incremental-area-weighted node rainfall and outlet
Q. Rain episodes use a 0.1 mm wet-hour threshold, a six-hour separating dry gap
and 5 mm minimum total. A candidate must reach station Q90 and rise by at least
`max(1 m3/s, 3 * median absolute hourly Q change)`; A grade additionally
requires Q95 and 10 mm. These uniform rules are evaluation-only. Cross-split
events are excluded and only TEST events are written.

`test_flood_event_samples.csv` references overlapping frozen Step16 TEST
windows. It does not copy dynamic data and never affects training.

## Commands

```powershell
python train_hunan.py --config configs/hunan_p2_continuous_multitask.yaml --dataset-root "D:\path\to\_model_dataset_v6_continuous_multitask"

python scripts/build_v6_test_flood_events.py --dataset-root "D:\path\to\_model_dataset_v6_continuous_multitask" --output-dir outputs/p2_continuous_flood_events

python evaluate.py --config configs/hunan_p2_continuous_multitask.yaml --checkpoint outputs/hunan_p2_continuous_multitask_best.pt --dataset-root "D:\path\to\_model_dataset_v6_continuous_multitask" --event-sample-index outputs/p2_continuous_flood_events/test_flood_event_samples.csv --output-dir outputs/hunan_p2_flood_event_test --output outputs/hunan_p2_flood_event_test.json
```

Evaluation writes physical Q, absolute Z and delta-Z observations/predictions,
overall and graph/station/event/horizon metrics, event/horizon Q peak magnitude
and timing errors, and top-error events. Absolute Z is reconstructed only as
`Z_pred(t+h) = observed Z(t0) + predicted delta_Z(t+h)`.
