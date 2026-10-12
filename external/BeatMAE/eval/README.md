# eval/ — unified evaluation entry points

Model evaluation is kept separate from software testing: `tests/` guards code correctness,
`eval/` produces model metrics. Three concerns are decoupled: **encoder/embedding × probe head
(trained or loaded) × test protocol**.

> **Protocol note.** This project's **final output is segment-level**. Earlier subject-pooled
> protocols (`linear`/`mlp` heads over subject-level aggregates) have been retired. The
> current segment-level entry points are materialization + the streaming causal probe
> (`stream/`) and the same protocol under 5-fold CV (`stream5f/`).

## Reuse and boundaries

- Embedding assembly and the training functions are **imported directly from
  `train/probe.py`** (`build_samples`, `_ridge_cv`, `_train_mlp_head`), so the protocol is
  identical bit-for-bit rather than reimplemented
- The streaming head's training functions live in `models/probe_heads/`, with windowing logic
  in `data/windows.py`; the segment aggregation convention is documented in each script's
  docstring

## Campaign scripts

Two levels, pick as needed:

- **Single experiment / ad-hoc combination**: `example_workflow.sh` (new weights → extract →
  full segment-level streaming chain) and `example_matrix.sh` (single-stage subcommands) —
  loop over encoder cache × test set combinations, passing parameters on the command line
  rather than writing new files, consistent with the `config/` base convention
- **Campaign level**: `run_campaign.py` + `config/campaign_*.yaml` — one yaml defines the
  experiment list × stages (`extract`/`stream`), with CLI deltas overriding; fails fast. See
  the campaign section of `config/README.md`

Both scripts read the corpus root from `PPG_CORPUS_ROOT`; **set it to your own corpus path** —
the built-in default will not exist on your machine.
