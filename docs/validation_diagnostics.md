# Validation diagnostics contract

## Scope and audited data flow

This diagnostic layer is evaluation-only. It does not change model forward,
loss/backpropagation, Q:Z weights, optimizer, data split, event construction,
physics, or checkpoint selection.

The authoritative mapping is:

1. `flood_events_final.csv` defines one real flood by globally unique
   `EVENT_ID`, its `GRAPH_ID`, `OUTLET_ID`, and rain/hydro/sample times.
2. `data_split.csv` assigns the complete event to one split.
3. `sample_index.csv` expands that event into hourly forecast samples. Its
   `FORECAST_TIME` is the last history hour; targets are hours 1 through 6.
4. `HunanGraphEventDataset` places targets only at the graph outlet named by
   `OUTLET_ID`, and applies `FLOW_MASK`/`WATER_LEVEL_MASK` there.
5. The loader now carries the already-validated sample/event/time metadata into
   evaluation. These strings never enter model `forward`.

The committed `_model_dataset_v4_candidate` contains 800 formal events and
39,097 forecast samples. VALIDATION contains 120 real events and 5,366 samples,
which confirms that a sample is not an event.

## Real-event aggregation rule

For each variable and each `(EVENT_ID, station, target timestamp)`, evaluation
retains the forecast with the shortest lead time. In an hourly rolling setup
this is the latest available issue time. A tie is resolved by lexical
`SAMPLE_ID`. This produces exactly one prediction per observed event hour while
remaining deterministic and causal.

The detailed rows record both `raw_*_forecast_point_count` and the deduplicated
valid count. The summary also records the pre/post-dedup point counts and their
ratio. Existing window-weighted metrics remain unchanged and keep their old
names; the new CSVs use the deduplicated rolling series.

Event volume is `sum(valid unique hourly Q) * 3600` in m³. Missing Q hours are
not converted to zero. Relative peak error is defined only when observed peak
Q is at least 1 m³/s. Relative volume error is defined only when mean observed
Q over the valid integrated hours is at least 1 m³/s. Otherwise the value is
`NaN` and the status column explains why.

## Delta-Z baseline (no future leakage)

For every retained Z forecast point, the baseline is the latest valid observed
water level at the target station inside that sample's history window, at or
before `FORECAST_TIME`. If water level at the origin is missing, evaluation
searches backward only within the configured history window. If no baseline is
available, that point is excluded from delta-Z metrics and counted as skipped.

Both values use the same known-at-issue baseline:

`delta_z_obs = z_obs(target) - z_obs_baseline`

`delta_z_pred = z_pred(target) - z_obs_baseline`

No target-period observation is used to shift or calibrate a prediction. The
baseline timestamp is asserted not to exceed `FORECAST_TIME`.

## Output files

Best-validation diagnostics are overwritten only when the best checkpoint
improves, under `<checkpoint_stem>_validation_diagnostics/`. Independent
evaluation writes to the explicit `--diagnostics-dir`, or to a deterministic
directory derived from `--output`/checkpoint.

- `validation_q_by_graph.csv`: Q NSE/KGE/MAE/RMSE/bias/SSE, validity reasons,
  valid point count, and event count for each `GRAPH_ID`.
- `validation_q_by_event.csv`: official event provenance, evaluated period,
  peak magnitude/time errors, volume errors, Q regression metrics and SSE.
- `validation_q_top20_error_events.csv`: event `q_rmse` descending.
- `validation_q_top20_sse_events.csv`: event SSE descending, with individual
  and cumulative fractions of total deduplicated event SSE.
- `validation_z_by_station.csv`: absolute Z metrics by target station.
- `validation_delta_z_by_station.csv`: causal delta-Z metrics by target station.
- `validation_diagnostics_summary.json`: SSE concentration, worst/median graph
  Q, median station Z, overall delta-Z, point counts, and exact rules.

NSE is `NaN` for no observations, fewer than two usable points, or zero observed
variance. KGE is additionally `NaN` for zero prediction variance or zero
observed mean. The associated `*_status` field distinguishes these cases.
KGE is intentionally omitted for delta-Z because its bias ratio is unstable
when mean event-relative water-level change is near zero.

Per-epoch training CSVs receive only compact appended summary fields such as
`val_z_station_nse_median`, `val_z_station_kge_median`,
`val_delta_z_mae`, and `val_delta_z_station_nse_median`. Existing columns,
including H1-H6 and sample peak/timing/volume metrics, are unchanged.
