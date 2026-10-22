# bt-metric

Research toward a translation quality metric for minority language Bible translation. See [PLAN.md](PLAN.md) for the research plan.

## Setup

Requires Python 3.12 (matching silnlp; pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/).

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
uv run btmetric-meta-evaluate                              # lexical baselines: chrF3, chrF3++, m-chrF3, BLEU, spBLEU
uv run btmetric-meta-evaluate --features comet22 africomet # neural baselines, on the GPU
```

Results accumulate in `data/results/meta_eval.tsv`; rerunning a feature replaces its rows. TER is available with `--features ter`, but sacrebleu's implementation takes over an hour on these sets.

### COMET environment

`unbabel-comet` requires `numpy<2`, which has no Python 3.13 wheels, so COMET runs in its own uv environment in `tools/comet` (Python 3.12, PyTorch with CUDA 12.8). The COMET features start `tools/comet/worker.py` in it automatically; create it once with:

```
cd tools/comet && uv sync
```

## Failure-mode checks

Score other published translations against an experiment's reference, on the same verses as its MT systems (default: Hebrew OT to English NASB). A metric that rewards meaning should place every human translation above most MT outputs:

```
uv run btmetric-test-human-translations --features chrf3 bleu comet22
```

Rank the human translations that WMT MQM judged alongside MT systems, by MQM and by each metric:

```
uv run python -m btmetric.analysis.wmt_human_references
```

Count the verses on which each pair's outputs differ. This adds `differing_verses`, `differing_fraction` and `near_identical` (under 10% of verses differ, by default) to the pairs file. Pairs whose predictions can't be aligned verse by verse get no fraction:

```
uv run btmetric-measure-pair-differences M:/MT/experiments --pairs data/system_pairs.tsv
```

Generated and downloaded files go in `data/`, which is not committed.

## Layout

- `btmetric/experiments`: indexing past silnlp experiments and finding system pairs
- `btmetric/scripture`: Scripture extracts and Paratext project metadata (back translation pairing, Bible corpora)
- `btmetric/datasets`: downloading public human-judgment datasets
- `btmetric/evaluation`: the common data model (evaluation sets, system outputs, human judgments) and its loaders
- `btmetric/features`: the `Feature` interface every metric component implements, and the baselines (chrF, BLEU, TER, COMET)
- `btmetric/analysis`: failure-mode checks (other human translations)
- `tools/comet`: the separate environment and worker process for COMET models
- `btmetric/meta_eval`: meta-evaluation statistics (pairwise accuracy, acc_eq with tie calibration, bootstrap intervals, paired bootstrap significance) and the runner
