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

This shortest-lead rule removes only sliding-window duplication *inside one
formal EVENT_ID*. It deliberately does not hide duplication across different
EVENT_ID values. Before training, `audit_dataset_quality.py` compares every
temporally adjacent or overlapping pair in the same `GRAPH_ID` and target
station. A pair is `MUST_MERGE` only when at least one of these deterministic
conditions holds:

1. the two events share valid target timestamps and their independently
   evaluated observed peaks occur at exactly the same timestamp; or
2. both official hydro windows are complete and overlap.

Shared target timestamps with different observed peaks are `REVIEW`, not an
automatic merge. A non-overlapping hydro gap no longer than the configured
forecast horizon is also `REVIEW`. If `MUST_MERGE` evidence crosses split, the
status becomes `CROSS_SPLIT_LEAKAGE`. Strict validation rejects both failure
statuses. The audit never merges merely because EVENT_ID values are adjacent
or peak values happen to be equal at different times.

The current repository does not contain the authoritative upstream event
builder named by `metadata/source_manifest.json`. Consequently the audit
reports provisional connected-component counts but does not rewrite EVENT_ID,
split, sample, normalization, or checkpoint artifacts. Those must be rebuilt
together by the upstream data pipeline.

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

## Water-level datum and range QC

`qc/water_level_station_audit.csv` uses unique target timestamps rather than
window-weighted repetitions. For each target station and split it reports the
physical-unit distribution, station TRAIN range, global TRAIN normalization
range, out-of-range counts, and consecutive-hour changes. Non-consecutive
observations are never differenced.

The jump threshold is the station TRAIN Tukey upper outer fence
`Q3 + 3*IQR` of absolute consecutive-hour changes. Station datum consistency
is checked across TRAIN events: an event is a reference-shift failure only if
its entire min/max range lies outside the Tukey outer fences of TRAIN event
median levels. The table records the exact event IDs; no station is silently
excluded.

Input Q/Z histories are standardized with TRAIN-only global statistics, but
formal Q/Z targets and model outputs remain in physical m³/s and metres.
Trainer loss divides prediction and target errors by TRAIN standard deviation;
it does not inverse-transform evaluation values. Therefore a station datum
break is a data problem, not an evaluation rescaling problem.

## Output files

Best-validation diagnostics are overwritten only when the best checkpoint
improves, under `<checkpoint_stem>_validation_diagnostics/`. Independent
evaluation writes to the explicit `--diagnostics-dir`, or to a deterministic
directory derived from `--output`/checkpoint.

- `validation_q_by_graph.csv`: Q NSE/KGE/MAE/RMSE/bias/SSE, validity reasons,
  valid point count, event count, total-SSE fraction, and SSE rank for each
  `GRAPH_ID`.
- `validation_q_by_event.csv`: official event provenance, evaluated period,
  peak magnitude/time errors, volume errors, Q regression metrics and SSE.
- `validation_q_top20_error_events.csv`: event `q_rmse` descending.
- `validation_q_top20_sse_events.csv`: event SSE descending, with individual
  and cumulative fractions of total deduplicated event SSE.
- `validation_z_by_station.csv`: absolute Z metrics by target station.
- `validation_delta_z_by_station.csv`: causal delta-Z metrics by target station.
- `validation_diagnostics_summary.json`: SSE concentration, worst/median graph
  Q, positive-NSE graph fraction, top-1/3/5 graph SSE fractions, station Z and
  delta-Z p25/median/p75, overall delta-Z, point counts, and exact rules.

NSE is `NaN` for no observations, fewer than two usable points, or zero observed
variance. KGE is additionally `NaN` for zero prediction variance or zero
observed mean. The associated `*_status` field distinguishes these cases.
KGE is intentionally omitted for delta-Z because its bias ratio is unstable
when mean event-relative water-level change is near zero.

Per-epoch training CSVs receive only compact appended summary fields such as
`val_z_station_nse_median`, `val_z_station_kge_median`,
`val_delta_z_mae`, and `val_delta_z_station_nse_median`. Existing columns,
including H1-H6 and sample peak/timing/volume metrics, are unchanged.

Pooled absolute-Z metrics remain available for compatibility, but they mix
between-station datum differences with within-station flood variation. They
must not be the sole multi-station water-level conclusion. Report the full
station table and station distribution summaries together with causal delta-Z.
