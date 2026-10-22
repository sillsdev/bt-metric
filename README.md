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

## Human back translations

Pair each human back translation extract in `MT/scripture` with an extract of the vernacular project it translates, using the base project recorded in each Paratext project's `Settings.xml`:

```
uv run btmetric-pair-back-translations --projects M:/Paratext/projects --scripture M:/MT/scripture
```

## Public human-judgment datasets

Download WMT MQM, WMT DA and AfriMTE into `data/external/`:

```
uv run btmetric-download-datasets
```

## Bible corpora for the simulated low-resource setup

List the Scripture extracts available in each language of the human-judgment sets, with verse counts, and pick the most complete one (or a preferred one, set in `btmetric/scripture/evaluation_languages.tsv`):

```
uv run btmetric-find-bible-corpora --scripture M:/MT/scripture M:/eBible/MT/scripture
```

## Meta-evaluation

Score features against the human judgments: system-level pairwise accuracy on WMT MQM, and segment-level acc_eq (WMT DA, MQM) or Kendall's tau-b (AfriMTE):

```
uv run btmetric-meta-evaluate --features chrf3 chrf3++
```

Generated and downloaded files go in `data/`, which is not committed.

## Layout

- `btmetric/experiments`: indexing past silnlp experiments and finding system pairs
- `btmetric/scripture`: Scripture extracts and Paratext project metadata (back translation pairing, Bible corpora)
- `btmetric/datasets`: downloading public human-judgment datasets
- `btmetric/evaluation`: the common data model (evaluation sets, system outputs, human judgments) and its loaders
- `btmetric/features`: the `Feature` interface every metric component implements, and the chrF baseline
- `btmetric/meta_eval`: meta-evaluation statistics (pairwise accuracy, acc_eq with tie calibration, bootstrap intervals, paired bootstrap significance) and the runner
