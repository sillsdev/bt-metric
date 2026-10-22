# bt-metric

Research toward a translation quality metric for minority language Bible translation. See [PLAN.md](PLAN.md) for the research plan.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```

## Past silnlp experiments as system pairs

Index every saved test prediction under the experiments folder (the S3 bucket mounted at `M:` here). It records each output's test set hash, prediction hash, config and test design:

```
uv run btmetric-index-experiments M:/MT/experiments --output data/experiments.tsv
```

Find the pairs of distinct outputs scored on the identical test set (same verses, same reference), labelled as `checkpoint` (two steps of one experiment), `rerun` (same config) or `config` (different configs, with the differing config keys):

```
uv run btmetric-find-system-pairs --index data/experiments.tsv --output data/system_pairs.tsv
```

Generated files go in `data/`, which is not committed.

## Layout

- `btmetric/experiments`: indexing past silnlp experiments and finding system pairs
- `btmetric/meta_eval`: meta-evaluation statistics (pairwise accuracy, acc_eq with tie calibration, bootstrap intervals, paired bootstrap significance)
