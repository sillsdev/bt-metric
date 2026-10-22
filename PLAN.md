# Research Plan: A Translation Quality Metric for Minority Language Bible Translation

## 1. Goal

Design and validate a reference-based metric for **comparing MT systems on held-out Bible books** in minority languages. It should replace chrF3 as the main evaluation metric in [silnlp](../silnlp).

The metric should:

1. Measure the three main criteria for a good Bible translation, **accuracy, clarity and naturalness**, as a proxy for the overall **usability** of a draft.
2. Work for minority languages, where the only data is usually a parallel Scripture corpus.
3. Be shown to correlate with real translation quality, relying on **existing human-judged test sets** and **simulated low-resource conditions** rather than new human evaluations.

### Scope decisions

- **Reference-based.** A held-out human translation is always available.
- **System comparison is the main use case.** The headline target is system-level pairwise accuracy: does the metric agree with humans on which of two systems is better? Segment-level correlation comes second.
- **No new human evaluations.** Little human usability data exists today; `sample_usability.py` is mostly unused. Validation uses existing human judgments, simulation, and tests that need no human judgments. A small human study at the end is optional and nothing depends on it.

### Limitations of chrF3 this should address

- It penalizes valid paraphrases and different translation styles, because there is only one reference.
- It barely notices omissions, additions and key term errors that change meaning.
- It doesn't measure naturalness or clarity.
- A given score difference has no clear meaning in terms of quality humans would notice.

---

## 2. Metric Design

### 2.1 Overall form

A log-linear (weighted linear) combination of normalized features:

```
score(hyp, ref, src) = Σ_i w_i · norm_L(f_i(hyp, ref, src))
```

- `f_i` are the feature scores, `w_i` the weights (non-negative, summing to 1 so they are easy to interpret), and `norm_L` is a per-language normalization (§2.4).
- Scores are computed per verse and aggregated to book level. Compare a corpus-level aggregate with a mean of verse scores (cf. m-chrF in `silnlp/nmt/test.py`).
- Report a **sub-score for each criterion** (accuracy, clarity, naturalness) next to the combined score, so a fluent but inaccurate draft can't hide behind a good total.

### 2.2 Candidate features

| Criterion | Feature | Description | Status |
|---|---|---|---|
| Accuracy | **chrF3** (and chrF3++) | Character n-gram F-score against the reference, recall-weighted (β = 3) as in silnlp | Core (anchor and baseline) |
| Accuracy | **Key term accuracy** | For each Biblical Term occurring in the verse, check whether the hypothesis contains the project's rendering (Paratext renderings) or the reference's rendering (found by aligning source to reference). Use stem or fuzzy matching to handle morphology. | Core |
| Accuracy | **Alignment coverage** | Align hypothesis to reference and to source (eflomal / fast_align). Unaligned-source rate signals omissions; unaligned-hypothesis rate signals additions. | Core |
| Accuracy / Clarity | **Back-translation features** | Give credit for valid paraphrases that chrF misses. Several options with different costs and memorization risk (§2.3). | Must justify its cost in ablations |
| Naturalness | **Perplexity ratio** | Character-level LM perplexity of the hypothesis divided by that of the reference on the same verses. Use a two-sided penalty, because degenerate output (repetitive, generic) can get *lower* perplexity than human text. | Core |
| Clarity | Back-translation fluency, sentence/clause complexity relative to the reference | Exploratory proxies. Clarity is the hardest criterion to measure automatically. | Exploratory |
| Accuracy | **Edit distance** (TER / charTER) | Likely to duplicate chrF; keep only if ablations show it adds signal | Probation |
| — | MT model confidence | Dropped. It can't be compared across systems with different vocabularies, and each system is biased toward its own output. It remains useful for quality estimation (`quality_estimation.py`), which is a different task. | Dropped |

**Constraints on auxiliary models.** To match real minority-language conditions:

- The LM, aligner and back-translation model are trained **only on the Bible corpus** for that language.
- They are **not trained on the test data**. Many past experiments test on a random sample of 250 verses rather than whole held-out books (§6.2), so exclude those exact verses. For the strictest setting, also exclude whole chapters, because neighbouring verses share wording.
- They are **independent of every MT system being compared**.

### 2.3 Back-translation: options, cost and memorization

Many projects have no human back-translation, so any back-translation feature has to **generate or score with a model**. No existing back-translation is required.

**Memorization is the central risk.** NLLB and LLMs have been pretrained on Bible translations in many major languages. A back-translation model can **recall the canonical English verse instead of translating the hypothesis**. That hides omissions, negation flips and mistranslations, which are exactly the errors back-translation is meant to catch. The risk applies to any model with a pretrained major-language decoder, NMT or LLM.

**Options:**

| Option | Cost | Memorization |
|---|---|---|
| **Reverse model trained from scratch** on non-held-out books only | Cheap to train; weak back-translations | **None by construction:** it has never seen the English text of the held-out books |
| **Fine-tuned reverse NLLB** | One fine-tune per language; cost spread across all comparisons | High (English decoder pretrained on Bibles) |
| **LLM with in-context examples** (as in silnlp's remote/local LLM configs, with retrieved examples) | No training per language; cost grows with languages × systems × verses (long prompts per verse, little caching benefit) | High; likely worse for smaller models |
| **Scoring instead of generating**, with any of the above | One forward pass per verse, no generation; needs probabilities for given text (open-weights models; most hosted APIs don't expose them) | Reduced when the memorized prior is subtracted (below) |

**Scoring that subtracts the memorized prior.** Plain `log P(src | hyp)` still benefits from memorization. Instead, measure how much information the hypothesis adds beyond what the model already expects (cf. conditional cross-mutual information, Fernandes et al. 2021):

```
info(x)  = log P(src | x) − log P(src | ∅)     # ∅ = empty or uninformative input
bt_score = info(hyp) / info(ref)
```

Comparing with the reference cancels per-verse difficulty and the reverse model's overall quality. Verses where `info(ref)` is close to zero (memorization dominates) get a flag rather than a score.

**Cost controls:**

- Back-translate or score the **reference once** and cache it.
- Freeze each language's back-translation model.
- Use temperature 0, a fixed example retrieval and a fixed prompt template.
- Prefer **open-weights models**: hosted models get updated or retired, which silently changes scores.

**Memorization gate.** Every back-translation option must pass the perturbation-based memorization test (§3.3a) before it counts as a feature:

- **Error survival rate:** does the injected error appear in the back-translation?
- **Wrong-verse check:** does a wrong verse back-translate to something unlike the correct source?

### 2.4 Per-language normalization

Feature scales differ across languages. chrF runs lower for languages with rich morphology, and perplexity depends on script and orthography. Normalize each feature per language against anchors computed from that language's own data:

- **Ceiling:** a human translation scored against the reference. Use a second translation where one exists; otherwise a proxy such as self-consistency across held-out books.
- **Floors:** a random verse from the same book, and a copy of the source.

Normalization may remove most of the apparent need for per-language weights (§4.3), so it is applied before any weight adjustment.

---

## 3. Validation Without New Human Evaluations

Validation has three parts.

### 3.1 Existing human-judged test sets

Use sets that have references, multiple systems and human judgments:

| Dataset | Languages | Judgments | Role |
|---|---|---|---|
| **WMT Metrics MQM** (2020–2024) | en-de, zh-en, en-ru, he-en, en-es, ja-zh, … | Expert MQM error spans, many systems | Main system-level benchmark; error categories map onto the three criteria |
| **WMT General DA** | Includes lower-resource pairs over the years (e.g. kk, gu, ps, km, ha, is, sah, liv) | Direct assessment | System-level ranking for lower-resource languages |
| **AfriMTE** | 13 African languages | DA + error spans | Closest match in typology and resources |
| **IndicMT-Eval** | 5 Indic languages, several systems | MQM | System-level checks for morphologically rich languages |
| **MLQE-PE** | Includes si-en, ne-en | DA + post-edits | Additional low-resource segment-level data |

**What has been checked so far (Phase 1):** `btmetric-download-datasets` downloads these into `data/external/`.

| Source | Contents | Supports |
|---|---|---|
| `RicardoRei/wmt-mqm-human-evaluation` (Hugging Face) | MQM 2020–2022: en-de, zh-en (2020–22), en-ru (2021–22); 10–17 systems per test set; source, MT output, reference, score, system | System and segment level, high-resource only |
| `RicardoRei/wmt-da-human-evaluation` (Hugging Face) | DA 2017–2022, 87 year × language-pair sets, including gu, kk, km, ps, ta, ha, is, xh–zu, bn–hi, lt, lv, et | **Segment level only**: there is no system column |
| AfriMTE-ade-devtest-v2 (`masakhane-io/africomet`, WMT 2024 challenge set) | 13 African language pairs with source, reference and human scores | **Segment level only**: one MT system per pair |
| `google/wmt-mqm-human-evaluation` (GitHub) | MQM for generalMT 2022–2024 (adds en-es, ja-zh, …) | Not downloaded yet; check whether references are included |
| IndicMT-Eval | MQM for 5 Indic languages, 7 systems | Data is only shared as a Google Sheet; not downloaded yet |
| mt-metrics-eval (`mt-metrics-eval-v2.tgz`) | Full WMT data with system outputs and human scores per system | The Google Cloud archive refused anonymous downloads (403) in October 2026; needs another route |

**Gap:** none of the downloaded sets supports **system-level** evaluation in a lower-resource language. DA data with system names, from mt-metrics-eval or the WMT findings repositories, or IndicMT-Eval's 7 systems, would fill it. Until then, system-level evidence for lower-resource languages comes from the Bible-domain tests (§3.3).

**Mapping MQM categories to criteria** gives a separate human label for each sub-score:

- **Accuracy:** mistranslation, omission, addition, terminology
- **Naturalness:** fluency, grammar, unnatural phrasing
- **Clarity:** fluency/coherence, ambiguity (the closest available proxy)

### 3.2 Simulating low-resource conditions on these test sets

The human-judged sets cover languages with plenty of resources. In real use, every auxiliary model is trained only on a Bible corpus. Rebuild the features under that constraint:

- **Naturalness LM:** train on that language's Bible only, or a subset of about 8k–30k verses to match a typical minority-language NT.
- **Back-translation model:** train target→English on the **Bible for that language pair only**. Most of the test-set languages have a Bible on eBible.
- **Aligner:** train on the same Bible-only data.
- **Comparison condition:** compute the same features with full-resource models (e.g. NLLB) to measure how much each feature degrades under realistic constraints.

The auxiliary models are trained on the Bible but tested on news. That makes these results conservative, since real use is Bible on Bible.

### 3.3 Bible-domain validation (no human judgments)

No public Bible-domain test set has human judgments, so the Bible domain is covered by these tests:

**a. Perturbation suite.** Build pairs of outputs with less and more degradation, from human reference translations and MT outputs:

- clause omission, addition
- key term substitution
- negation flip
- word-order shuffle
- wrong verse
- copying the source
- repetition

Then:

- Check that the metric ranks each pair correctly, and that **each sub-score responds to its own error types** and not the others.
- This is the only validation available for key term accuracy, and it is the **memorization gate** for back-translation features (§2.3).
- Perturbation frequencies should roughly match real MT error distributions, e.g. by harvesting omissions and mistranslations from weak MT outputs via alignment.
- silnlp's `add_noise_to_corpus.py` (random deletion, replacement, permutation) is a starting point. Most of the suite needs errors that change meaning rather than random token noise.

**b. Test using other human translations.** Many languages, especially major ones, have more than one Bible translation. `MT/scripture` has about 674 English, 111 French and 103 Spanish extracts, including many published versions (BSB, CEV, ESV, GNT, KJV, NASB, NIV, NLT, NRSV, …), with a range from literal to dynamic.

- **Paraphrase robustness:** a different valid human translation scored against the reference should score well and beat MT output. chrF often fails this, which makes it a strong argument for the new metric.
- **Translation styles:** literal vs. dynamic translations of the same verse should not be ranked far apart on accuracy.
- **Degraded versions:** perturbed human translations should rank below the originals.

**c. Silver labels in major languages.** Train a range of Bible-only MT systems into major languages, varying architecture, data size and checkpoints. Score their outputs on held-out books with:

- strong high-resource metrics (MetricX, xCOMET)
- an LLM judge using MQM-style error annotation (GEMBA-MQM style)

Then test whether the metric, computed under low-resource constraints, reproduces the silver system rankings. Watch for **target-side contamination**: pretrained models have memorized major-language Bibles. To control for it:

- Train some forward systems **from scratch**, or measure how much forward-system outputs overlap word for word with known Bible versions and treat high overlap as contamination.
- Prefer **target languages less likely to be memorized** (mid-resource languages, AfriMTE and Indic languages), while remembering that NLLB saw Bible data for many lower-resource languages too.
- Use several silver judges, and check whether they agree more on outputs that are word-for-word close to known Bible versions. That would be a sign of memorization.

**d. Quality ladders.** Train systems on 1k, 2k, 4k, … verses, or take checkpoints, and check that the metric recovers the expected order. Past experiments (§6.2) already provide many of these in real minority languages: series that vary the amount of training data, the sources or the model, plus two saved checkpoints for most experiments. It's a weak test, meaningful only where the quality steps are large. It must not depend on chrF-based checkpoint selection, which would be circular.

**e. Human back-translations as a real-world check.** Some Paratext projects have a human back-translation project (the "Back Translation" project type in silnlp onboarding). These show up in `MT/scripture` as English-coded extracts, about 336 files with "BT" in the name, e.g. `en-ABCBT_2023_11_30`. That count includes dated versions and maybe false matches, so there are fewer distinct projects. Several are already used as training sources in past experiments. Where one exists:

- Compare generated back-translations of the **reference** against the human back-translation.
- This measures back-translation quality and memorization leakage on **real minority-language text**, without new annotation.
- A human back-translation is written in English alongside the vernacular text. It isn't a literal gloss, so it is a noisy target. Compare the human back-translation with the published English source text to estimate that noise.
- Pairing back-translations with vernacular projects needs project metadata. The file names alone aren't enough.

**f. Real minority-language system outputs.** Past experiments (§6.2) provide predictions, references and per-verse scores for thousands of systems in over 200 target languages. They have no human labels, but they support:

- computing every feature offline on real minority-language MT output, with no retraining
- calibrating the perturbation suite against the errors real MT systems make (§3.3a)
- the quality ladders in §3.3d
- per-language normalization anchors (§2.4) and the label-free objectives for per-language weights (§4.3)
- checking where the new metric and chrF3 **disagree** on system rankings. These cases are worth inspecting by hand, and they would be the first items to send to human raters if a human check (Phase 10) happens.

### 3.4 Baselines

Run on every evaluation above:

- chrF3, chrF3++, m-chrF3
- BLEU, spBLEU
- TER
- COMET / xCOMET, which will likely degrade for unseen languages
- AfriCOMET
- an LLM judge where feasible

### 3.5 Meta-evaluation statistics

- **Main statistic:** system-level pairwise accuracy (Kocmi et al. 2021).
- Segment-level Kendall τ, using the WMT acc_eq variant to handle ties.
- Per-criterion correlation of each sub-score with its matching MQM categories.
- Results at both verse and book/chapter granularity.
- Bootstrap confidence intervals throughout.
- Leave-one-dataset-out and leave-one-language-out cross-validation for anything fitted.

---

## 4. Weight Fitting

### 4.1 Global weights

- Fit on the §3.1 data using §3.2 features computed under low-resource constraints.
- **Loss:** pairwise logistic ranking loss at the segment level (different systems' outputs for the same verse), which gives far more training examples than whole-system comparisons. Evaluate at the system level.
- **Regularization:** about 6 features, L2 penalty, non-negativity constraint.
- **Key term weight:** these sets have no Biblical-term annotations. Set this weight from the §3.3c silver labels, or use an expert prior checked with perturbation sensitivity.
- **Bible-domain check:** confirm that weights fitted on news still rank correctly on §3.3b–d. If not, refit on the §3.3c silver labels.

### 4.2 Expert priors and blending

Compare three sources of weights:

1. **Expert priors**, elicited from translation consultants with a multi-criteria decision method (e.g. AHP).
2. **Data-fitted weights** (§4.1).
3. **Bayesian blend:** expert weights as priors, updated with data.

### 4.3 Per-language weights

An optimizer can fit weights for each language, but the **objective** has to come from somewhere other than human labels. With about 6 weights, the choice of algorithm hardly matters. Options:

- a smooth pairwise surrogate with gradient descent
- derivative-free methods (CMA-ES, Nelder-Mead)
- MERT-style coordinate ascent, which optimizes ranking accuracy directly

Candidate label-free objectives:

| Objective | Description | Main risk |
|---|---|---|
| **Perturbation discrimination** | Maximize accuracy at ranking clean translations above perturbed ones in that language's Bible | The perturbation mix effectively sets the weights; tunes to synthetic rather than real errors |
| **Quality ladder ranking** | Maximize agreement with the order of systems trained on more and more data | Noisy and coarse; only constrains features that change with data size |
| **Unsupervised agreement** | Weights = loadings on a shared hidden quality factor (PCA / factor model) | Rewards shared surface-overlap signal; down-weights features with unique information such as key terms |
| **Hierarchical model** (recommended) | Global weights plus per-language deviations pulled back toward them (shrinkage), driven by the label-free objectives above and optionally predicted from language properties (morphological complexity, script, corpus size) | More complex to fit; needs careful choice of shrinkage strength |

**Experiment.** For each language in the human-judged sets:

1. Fit per-language weights using **only** label-free objectives, without looking at that language's human judgments.
2. Compare pairwise accuracy on its human judgments under:
   - global weights
   - global weights with per-language normalization
   - label-free per-language weights
   - the hierarchical model
3. As an upper bound, also fit weights on the language's own human labels (with cross-validation).

If label-free per-language tuning doesn't beat normalized global weights here, it won't help on minority languages either. That is a useful result in itself.

**Safeguards:**

- Tune weights on synthetic data, ladder systems or held-out books, **never on outputs of the systems being compared**, so the metric doesn't drift toward one system's style.
- **Freeze and version** each language's weights so scores stay comparable over time.
- Keep per-language models small, with strong shrinkage and bootstrap confidence intervals on the weights themselves.

---

## 5. Interpreting Scores: Significance Thresholds

For system comparison, knowing **when a difference is meaningful** is as important as the absolute score.

- Following Kocmi et al. (2024), *Navigating the Metrics Maze*, map each metric difference to the probability that humans agree with the direction. Use system pairs from WMT, AfriMTE and IndicMT. Example: "a gain of X points predicts human preference Y% of the time."
- Combine this with **paired bootstrap resampling** over verses for statistical significance on a given book.
- Report results in this form: *"System B beats A by X (95% CI [a, b]); historically, differences this size match human preference Y% of the time."*

---

## 6. Existing Infrastructure and Data

### 6.1 silnlp

Research code lives in this repository and uses silnlp as a library. The final metric is then moved into silnlp.

| Plan component | silnlp starting point | Gap |
|---|---|---|
| Baselines, scoring per book and per verse | `silnlp/nmt/test.py` (BLEU, chrF3/+/++, spBLEU, m-chrF, TER; multiple references) | Final metric added as a new scorer |
| Key term accuracy | Major Biblical Terms assets (`Major-metadata`, `Major-vrefs`, en/fr glosses); project renderings in `MT/terms`; preliminary proper-noun work in `scripts/IDQ/key_terms.ipynb` | Matching that copes with morphology; all term categories, not just proper nouns |
| Alignment coverage | `silnlp/alignment/utils.py` `compute_alignment_scores` (eflomal, fast_align; symmetrized alignment output) | Coverage features; tokenization beyond `LatinWordTokenizer` for scripts that don't separate words with spaces |
| Back-translation by LLM with in-context examples | `remote_llm_config.py`, `local_llm_config.py`, `example_retrieval.py` (TF-IDF, embedding and coverage-aware selection), response caching | Reverse-direction experiment configs; memorization gate |
| Back-translation scoring | Model loading and `compute_transition_scores` in `seq2seq_config.py` / `local_llm_config.py` (scores the model's own output only) | Scoring of given text, plus the memorized-prior term `P(src \| ∅)` |
| Perturbation suite | `silnlp/common/add_noise_to_corpus.py` | Perturbations that change meaning |
| System pairs, quality ladders | ClearML experiment history; `combine_scores.py` | Selecting pairs that share held-out books |
| Downstream use | `quality_estimation.py` / `linear_regression.py` (confidence → projected chrF3 → usability labels) | Could project the new metric instead of chrF3 later |
| Optional human check | `silnlp/common/sample_usability.py` (stratified verse sampling) | Little human data collected so far |

**Operational constraint:** silnlp's `AGENTS.md` allows **only one remote ClearML job at a time** across all sessions. Check with `scripts/check_clearml_jobs.py` before submitting. Training many Bible-only auxiliary models and silver-label systems has to be scheduled with this in mind; run locally where possible.

### 6.2 Experiment and Scripture data (S3 bucket, mounted at `M:`)

**`MT/experiments`** holds about 390 top-level series folders: projects by country, `FT-*` fine-tuning series, `BT-*` back-translation-direction series, and `Research.*` folders such as `Research.LLM.ICL`.

Each experiment directory typically contains:

- `config.yml` / `effective-config-*.yml`: model (e.g. `facebook/nllb-200-distilled-1.3B`), corpus pairs, test/val sizes, seed, tokenizer and training parameters
- `test.vref.txt`, `test.src.detok.txt`, `test.trg.detok.txt`: test verse references, source and reference. Multi-source experiments use `test.<src>.<trg>.*`.
- `test.trg-predictions.detok.txt.<step>`: system output, usually for two checkpoints (best and last)
- `test.trg-predictions.detok.txt.<step>.scores.tsv`: per-verse BLEU with prediction and reference
- `scores-<step>.csv`: corpus-level BLEU, chrF3, WER, TER, spBLEU

**Aggregated index:** `experiments_scores.csv` and `experiments_config.csv`, last updated July 2025.

- About 2,400 experiments in 273 series, 239 target and 204 source languages.
- 153 series × target-language groups have at least 5 experiments; 255 have at least 3.
- The file mixes two row formats (long and wide), so it needs tolerant parsing. It's stale, so rebuild it from the experiment directories rather than relying on it.

**Shared test sets.** A system pair is only valid if both systems were scored on the **same verses against the same reference**. The same series and language don't guarantee that, because data, seeds and test sizes change. A scan of `MT/experiments` (up to 3 levels deep) grouped experiments by a hash of `test.vref.txt` + `test.trg.detok.txt`:

- 12,141 prediction files in 7,722 experiment directories with 5,117 distinct test sets. That is far more than the index lists, which confirms it's out of date.
- Test sets shared by at least 2 experiments: **1,303**; at least 3: 617; at least 5: 219; at least 10: 37. The largest groups have 20–25 systems.
- About **9,900 experiment-level system pairs** on identical test sets. Comparing checkpoints adds more: 2,944 experiments have at least two.
- Median shared test set: 250 verses (quartiles 250–655).

Before using the pairs:

- Remove reruns of the same configuration (compare effective configs) and pairs whose outputs are nearly identical. Neither tells you anything about ranking.
- Record what differs within each pair (data, source texts, model, tokenizer) so results can be broken down by type of difference.

**Test-set design.** Many experiments use `test_size: 250` with a fixed seed. That is a random sample of verses rather than held-out books. Consequences:

- Training verses next to the test verses share context and wording, so scores are somewhat optimistic. This affects all systems similarly, so it matters less for system comparison.
- Auxiliary models must exclude the exact test verses (§2.2).
- 250 verses per test set limits the statistical power of verse-level comparisons. The bootstrap intervals in §5 matter here.
- For the main evaluation, prefer series that hold out whole books, or create such series. Use random-verse series as additional data.

**`MT/scripture`** has about 7,300 verse-aligned extracts (one verse per line, same versification), including many English, French and Spanish versions and human back-translations (§3.3b, §3.3e).

**`MT/terms`** has about 3,300 files: Major Biblical Terms renderings, metadata and vrefs per project, which feed the key term accuracy feature.

**Other relevant series:**

- `BT-*` (e.g. `BT-English`, `BT-Spanish`, `BT-*-NLLB`): earlier experiments translating **into** major languages. These can be starting points or baselines for the reverse models in §2.3.
- `de-to-en-WMT2020+Bibles*`: systems trained on WMT and Bible data mixed, relevant to the simulation in §3.2.
- `Research.LLM.ICL`: LLM in-context translation work, with alignment of minority-language texts against many English and French versions.

---

## 7. Phases

| Phase | Work | Outputs |
|---|---|---|
| **1. Data and infrastructure** | Collect the §3.1 test sets; gather Bible corpora for each test-set language from eBible; rebuild the experiment index from `MT/experiments` and group experiments by shared test set (§6.2); pair human back-translations with their vernacular projects; build data loaders, the feature interface and the meta-evaluation harness on top of silnlp | Reproducible evaluation set; list of valid system pairs |
| **2. Baselines** | Run the §3.4 baselines on everything, including the human-translation test from §3.3b | Baseline results; confirmation of chrF3's failure modes |
| **3. Perturbation suite** | Implement perturbations that change meaning; calibrate their frequencies against real MT errors | Bible challenge set, plus the memorization gate |
| **4. Features** | Implement chrF3, key terms, alignment coverage and perplexity ratio under low-resource constraints; ablate each against its full-resource version; drop features that add nothing | Core feature set with standalone correlations |
| **5. Back-translation comparison** | Compare from-scratch reverse model, reverse NLLB and small open-weights LLM (each in generation and scoring mode), the current hosted LLM setup as a quality upper bound, and no back-translation. Judge on (1) memorization gate, (2) correlation, (3) cost per book. Validate against human back-translations (§3.3e). | Back-translation decision |
| **6. Silver data** | Train Bible-only MT systems and quality ladders into major languages with contamination controls; produce silver labels | Bible-domain system rankings |
| **7. Weights** | Normalization, global fitting, expert priors and blend, per-language experiment (§4.3) | Final weighting approach |
| **8. Significance** | Calibrate score differences against human agreement; build bootstrap tooling | Interpretation guidelines |
| **9. Integration** | Add the metric as a silnlp scorer; optionally project it in quality estimation | Metric in production use |
| **10. (Optional) Human check** | Small ranking or usability study on 1–2 minority languages, using `sample_usability.py` | Direct evidence on target-domain correlation |

---

## 8. Limitations and Risks

- **Clarity** gets only indirect validation through MQM fluency and coherence categories. It will be the weakest-supported criterion.
- **Domain gap:** human judgments come from news and general-domain text; Bible-domain validation relies on perturbations, alternative translations, silver labels and human back-translations.
- **What the claim is:** the evidence shows the metric tracks human judgment under simulated low-resource conditions and behaves correctly on Bible text. It doesn't directly show correlation with consultant judgments on minority-language Scripture. Phase 10 would close most of that gap.
- **Memorization:** pretrained models have memorized Bibles in major languages. This affects back-translation features, forward systems in the simulation, and silver judges. Back-translation features have to pass the memorization gate.
- **Bias in the silver labels:** MetricX, xCOMET and LLM judges have their own biases; use agreement across several of them.
- **Key term coverage:** depends on the availability and quality of Biblical Terms renderings for each project.
- **Goodhart effects:** if the metric drives system selection, systems may overfit to its features. Keep the perturbation suite and the human-translation test as regression checks.
- **Compute scheduling:** only one remote ClearML job at a time, so large training sweeps take longer.
- **Test-set design in past experiments:** random 250-verse test sets give optimistic scores and limited statistical power, and they complicate keeping test data out of auxiliary-model training.

---

## 9. Open Questions

- How complete are Biblical Terms renderings across projects, and which term categories are reliably filled in?
- Which of the ~336 `en-*BT*` extracts are human back-translations, and of which vernacular projects? Project metadata is needed to pair them reliably.
- Are there minority languages with more than one Scripture translation that could serve as additional human-translation tests?
- What computing budget is available for training Bible-only auxiliary models across many test-set languages?
- About 9,900 system pairs share identical test sets (§6.2). How many remain after removing reruns, and how many hold out whole books rather than random verses?
- Should new experiment series be created with whole held-out books to support the main evaluation?

## References

- Kocmi et al. (2021). *To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation.*
- Kocmi et al. (2024). *Navigating the Metrics Maze: Reconciling Score Magnitudes and Accuracies.*
- Freitag et al. (2021–2024). WMT Metrics Shared Task findings (MQM data).
- Wang et al. (2024). *AfriMTE and AfriCOMET.*
- Sai B et al. (2023). *IndicMT Eval.*
- Fomicheva et al. (2022). *MLQE-PE.*
- Amrhein et al. (2022). *ACES: Translation Accuracy Challenge Sets.*
- Kocmi & Federmann (2023). *GEMBA-MQM.*
- Fernandes et al. (2021). *Measuring and Increasing Context Usage in Context-Aware Machine Translation* (CXMI).
- Thompson & Post (2020). *Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing* (Prism).
- Popović (2015). *chrF.*
- m-chrF / m-BLEU: arXiv 2407.12832 (as cited in `silnlp/nmt/test.py`).
