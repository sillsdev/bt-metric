# bt-metric

Research toward a translation quality metric for minority language Bible translation. See [PLAN.md](PLAN.md) for the research plan.

## Setup

Requires Python 3.12 (matching silnlp; pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```

On Windows, run isort with `PYTHONUTF8=1` set. Some files contain text in other scripts, and isort otherwise reads them with the system code page and silently skips them.

## Past silnlp experiments as system pairs

Index every saved test prediction under the experiments folder (the S3 bucket mounted at `M:` here). It records each output's test set hash, prediction hash, config and test design:

```
uv run btmetric-index-experiments M:/MT/experiments --output data/experiments.tsv
```

Find the pairs of distinct outputs scored on the identical test set (same verses, same reference), labelled as `checkpoint` (two steps of one experiment), `rerun` (same config) or `config` (different configs, with the differing config keys):

```
uv run btmetric-find-system-pairs --index data/experiments.tsv --output data/system_pairs.tsv
```

Count the verses on which each pair's outputs differ. This adds `differing_verses`, `differing_fraction` and `near_identical` (under 10% of verses differ, by default) to the pairs file. Pairs whose predictions can't be aligned verse by verse get no fraction:

```
uv run btmetric-measure-pair-differences M:/MT/experiments --pairs data/system_pairs.tsv
```

Score the usable config pairs with the baselines: how often baselines disagree on the better system, and how often differences are significant (`--resume` scores only test sets missing from the output):

```
uv run btmetric-score-system-pairs
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
uv run btmetric-meta-evaluate --features xcomet-xl         # xCOMET-XL (gated on Hugging Face: accept its license and log in)
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

## Challenge sets

Build a Bible challenge set: real MT outputs from held-out-book test sets, each paired with a copy carrying one injected error (omission, addition, key term substitution, negation removal, word order, wrong verse, source copy, repetition, truncation; graded ones at a mild and a severe level). Key terms are curated renderings where a project has them and alignment-inferred otherwise. `--perturb-reference` perturbs the human reference instead, for the back-translation memorization gate:

```
uv run btmetric-build-challenge-set --min-term-confidence 0.6
```

Build a set of real errors instead: one system's output for a verse with a decoder loop, over-long or over-short output, or copied source, paired with another system's clean output for the same verse:

```
uv run btmetric-build-natural-errors
```

Measure how often real MT outputs show each error signature, which weights the challenge set's calibrated score:

```
uv run btmetric-measure-error-rates
```

Check which errors each metric notices (and, for graded errors, whether a mild one outscores a severe one):

```
uv run btmetric-evaluate-challenge-set --features chrf3 bleu coverage-target key-terms perplexity
uv run btmetric-evaluate-challenge-set --challenge-set data/natural_errors.tsv --output data/results/natural_errors.tsv
```

## Key term renderings

Infer key term renderings by aligning an English Bible with each project's text (Thot eflomal), and validate them against projects with curated renderings:

```
uv run btmetric-validate-rendering-inference --projects 30
```

## Metric features

Beyond the baselines, these features are opt-in because each trains a model per language pair. Models train on a Bible-sized amount of text **in the domain being scored** (`btmetric/features/corpora.py`): Bible text for Scripture, and news for WMT, from OPUS News Commentary or, failing that, the judged sets' own source-reference pairs scored in folds. The segments being scored are always left out.

| Feature | Measures |
|---|---|
| `coverage-source`, `coverage-target` | Share of source (target) words aligned to the other side, weighted by translation probability, relative to the reference: omissions (additions, loops, untranslated text) |
| `coverage-reference` | Omissions measured only against the source content the reference expresses, so a hypothesis is never asked to be more literal than the reference |
| `coverage-*-detector` | The same as error detectors: 0 while the hypothesis keeps 70% of the reference's coverage, penalized below that |
| `key-terms`, `key-terms-names` | Share of the reference's key term renderings the hypothesis also uses (all terms, or proper names only) |
| `perplexity`, `perplexity-one-sided` | Distance between the hypothesis's and the reference's character perplexity under a 5-gram model of the target language (one-sided: only being less predictable counts) |
| `bt-nllb` | How much the hypothesis tells a reverse NLLB-200 model about the source (forced decoding, no generated back translation), relative to the reference. Verses whose reference tells the model little beyond what it predicts with no input (memorized from Bibles) get no score, and a hypothesis that copies the source gets no credit. Runs on the GPU; NaN for languages NLLB doesn't cover |
| `bt-nllb-finetuned`, `bt-scratch` | The same with a reverse model trained per project for languages NLLB lacks: NLLB fine-tuned in the reverse direction, or a small transformer trained from scratch (which can't have memorized the scored verses). Trained on the GPU and cached under `data/models/reverse` |

PLAN.md §2.2.2 and the Phase 5 status in §2.3 record how each fares. On human judgments of news, none adds to chrF3: coverage and back-translation both reward literal translations there. On the Bible challenge set they catch the errors chrF3 misses (additions, repetition, copies).

Fit the combined metric on human judgments (WMT MQM and DA), with in-domain training, and compare it with each feature alone at segment and system level. `--add` adds features to an existing score table:

```
uv run btmetric-fit-combination-human --build chrf3 bleu perplexity coverage-reference --training-domain news
uv run btmetric-fit-combination-human --add coverage-target-detector --training-domain news
```

Check which features catch the omissions and additions that WMT MQM annotators marked (`--major-only` for Major errors):

```
uv run btmetric-check-mqm-error-categories --major-only
```

Check whether back-translation scoring survives NLLB's memorization of the Bible: inject each error type into other Bibles (German, French, Russian by default) as references for English sources, and count how often the perturbed reference scores lower than the clean one, and how many verses count as memorized:

```
uv run btmetric-memorization-gate
```

## Combining features

Fit the combined metric on the Bible challenge set's per-row scores (written by `btmetric-evaluate-challenge-set`), leaving one test set out, and compare it per error type with each feature alone. `--features` restricts the combination, e.g. to lexical features plus the coverage detectors as guards:

```
uv run btmetric-fit-combination
uv run btmetric-fit-combination --features chrF3 m-chrF3 BLEU spBLEU perplexity-one-sided coverage-source-detector coverage-target-detector coverage-reference-detector
```

On WMT human judgments the fit samples at most 20,000 segment pairs per dataset (`--max-pairs-per-dataset`; 0 keeps all), since all pairs together exhausted memory.

## Silver labels from xCOMET-XL

Measure how often each feature ranks real Bible translations as xCOMET-XL does, on past system outputs whose source and target are both languages XLM-R can read. Three steps, so the GPU step stands alone:

```
uv run btmetric-silver-agreement select                                  # sample verses and every system's output
uv run btmetric-silver-agreement score --features xcomet-xl xcomet-xl-qe # silver labels, on the GPU
uv run btmetric-silver-agreement score --features chrf3 bleu             # features to compare
uv run btmetric-silver-agreement agree                                   # pairwise agreement, within verses and per system
```

`xcomet-xl-qe` is xCOMET-XL without the reference, so its labels don't share the reference the metric scores against.

## Field-test judgments

Translation teams judged Scripture Forge drafts in the 2025 field tests: usability, comparisons of two models' drafts, and corrections. Export the two Google Drive folders as .xlsx with a read-only rclone remote (create it once with `rclone config create gdrive drive scope=drive.readonly`):

```
rclone copy gdrive: data/external/field_tests/nt_field_test --drive-root-folder-id <folder id> --drive-export-formats xlsx
rclone copy gdrive: data/external/field_tests/trial_2 --drive-root-folder-id <folder id> --drive-export-formats xlsx
```

Collect every judged verse into one table, with references from each team's project (`btmetric/datasets/field_test_teams.tsv`) and whether each reference predates the trial's drafts. Then score features and measure how often they side with the teams:

```
uv run btmetric-collect-field-tests
uv run btmetric-field-test-agreement build
uv run btmetric-field-test-agreement score --features chrf3 coverage-source coverage-target
uv run btmetric-field-test-agreement agree
```

## Significance thresholds

How large a system-level difference must be before humans reliably agree with its direction, from the WMT MQM system pairs:

```
uv run btmetric-significance-thresholds
```

The human-translation test can also score with a different source text, e.g. Hebrew script where the experiment used a transliteration that back-translation models can't read:

```
uv run btmetric-test-human-translations --features bt-nllb --source-extract hbo-HEB --source-language he --strip-marks
```

## Running long jobs

- **One GPU job at a time.** The machine crashed with several running at once. Chain GPU jobs in one command rather than starting them in parallel.
- **COMET scores are cached** in `data/cache/comet_scores.sqlite` as each chunk is scored, and progress (segments done, rate, time left) is logged per chunk. A stopped run resumes where it left off.
- **Tests on the CPU while the GPU is busy:** some tests train small models on CUDA, so run them with `CUDA_VISIBLE_DEVICES=-1`. On Windows an empty value does not hide the GPU.

Generated and downloaded files go in `data/`, which is not committed.

## Layout

- `btmetric/experiments`: indexing past silnlp experiments, finding system pairs and how much they differ
- `btmetric/scripture`: Scripture extracts and Paratext project metadata (back translation pairing, Bible corpora)
- `btmetric/datasets`: downloading public human-judgment datasets, and collecting the field-test judgments
- `btmetric/evaluation`: the common data model (evaluation sets, system outputs, human judgments) and its loaders
- `btmetric/features`: the `Feature` interface, the baselines (chrF, BLEU, TER, COMET), the metric features (alignment coverage, key terms, perplexity, back-translation scoring), reverse models, and the in-domain training corpora
- `btmetric/terms`: Major Biblical Terms, rendering and negation-word inference by alignment, and its validation
- `btmetric/perturbations`: the perturbation library, challenge sets (synthetic and real errors) and calibration against real error rates
- `btmetric/combination`: the pairwise log-linear combination, fit on challenge-set rows or WMT human judgments
- `btmetric/analysis`: failure-mode checks (other human translations, MQM error categories, the memorization gate), baselines on past system pairs, silver-label and field-test agreement, and significance thresholds
- `btmetric/meta_eval`: meta-evaluation statistics (pairwise accuracy, acc_eq with tie calibration, bootstrap intervals, paired bootstrap significance) and the runner
- `tools/comet`: the separate environment and worker process for COMET models
