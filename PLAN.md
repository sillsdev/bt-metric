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
| Accuracy | **Key term accuracy** | For each Biblical Term occurring in the verse, check whether the hypothesis contains the rendering the project uses. Most projects have no curated renderings, so they are **inferred by word alignment** (§2.2.1). Curated Paratext renderings, where they exist, serve to validate the inference. Use stem or fuzzy matching to handle morphology. | Core |
| Accuracy | **Alignment coverage** | Align hypothesis to reference and to source (eflomal / fast_align). Unaligned-source rate signals omissions; unaligned-hypothesis rate signals additions. | Core |
| Accuracy / Clarity | **Back-translation features** | Give credit for valid paraphrases that chrF misses. Several options with different costs and memorization risk (§2.3). | Must justify its cost in ablations |
| Naturalness | **Perplexity ratio** | Character-level LM perplexity of the hypothesis divided by that of the reference on the same verses. Use a two-sided penalty, because degenerate output (repetitive, generic) can get *lower* perplexity than human text. | Core |
| Clarity | Back-translation fluency, sentence/clause complexity relative to the reference | Exploratory proxies. Clarity is the hardest criterion to measure automatically. | Exploratory |
| Accuracy | **Edit distance** (TER / charTER) | Likely to duplicate chrF; keep only if ablations show it adds signal | Probation |
| — | MT model confidence | Dropped. It can't be compared across systems with different vocabularies, and each system is biased toward its own output. It remains useful for quality estimation (`quality_estimation.py`), which is a different task. | Dropped |

#### 2.2.1 Inferring key term renderings by alignment

Most projects have no Biblical Terms renderings, so the key term feature and the key-term perturbations (§3.3a) infer them from the parallel corpus:

1. **Locate each term in English.** The Major Biblical Terms list gives every verse a term occurs in. silnlp's gloss files (`en-Major-glosses.txt`, also fr, es, id, pt) locate it in an English Bible verse such as BSB. This doesn't depend on the source text of any MT system.
2. **Word-align English with the target** over the project's whole corpus, excluding test verses, with **eflomal**, which has worked best in the team's experience. Use the Thot implementation in `sil-machine[thot]` (`ThotEflomalWordAlignmentModel`, the same model as silnlp's `eflomal` aligner) with symmetrized alignments. It runs on Windows: a symmetrized model trained on 7,908 NT verses (English BSB → a minority language) in 14 seconds, with sensible alignments for names and terms. The standalone `eflomal` package needs a Unix build toolchain.
3. **Infer the rendering.** For each term, collect the target words aligned to its gloss in all of its verses. The most frequent word or phrase, matched at stem level to handle morphology, is the inferred rendering. The share of the term's verses it appears in is its confidence.
4. **Validate.** On the ~1,100 projects with curated Major renderings in `MT/terms`, measure precision and recall of inferred renderings by term category. That shows which categories (e.g. proper names vs. abstract terms) can be trusted.

**Validation results** (`btmetric-validate-rendering-inference`, October 2026). 30 projects, one per language, with curated renderings and a near-complete NT; English BSB aligned with each extract by symmetrized Thot eflomal.

| | Precision | Recall |
|---|---|---|
| All terms (29 projects, ~20,000 terms) | **0.68** | 0.67 |
| Proper names (69% of terms) | **0.76** (0.84 at confidence > 0.75) | |
| Other categories (AT, RE, BE, FA, MI, RI, FL) | 0.39–0.57 | |

| Confidence threshold | Precision | Inferred terms kept |
|---|---|---|
| ≥ 0.5 | 0.73 | 85% |
| ≥ 0.6 | 0.77 | 75% |
| ≥ 0.75 | 0.77 | 68% |

- **These are lower bounds.** Curated lists often record only some inflected forms, and matching accepts only shared-prefix variants.
- **Proper names are reliable.** Other categories need better matching (stem grouping of aligned words, multi-word renderings) before the key-term feature relies on them; a higher confidence threshold barely helps them.
- **Corpus size matters somewhat** (correlation 0.32 with verse count): projects under ~8,000 verses mostly score 0.37–0.53.
- **Threshold.** The challenge set and the key term features use inferred renderings with confidence ≥ 0.6, after the specificity filter of §2.2.2.
- **Data problems found:** `xyz-XYZ_2026_03_18` is mis-encoded (skipped automatically), and the `bi-ABC_2026_02_16` extract is English text under a Bislama code. Projects with only one or two comparable terms give meaningless per-project precision, but carry no weight in the pooled figures.

**Constraints on auxiliary models.** To match real minority-language conditions:

- The LM, aligner and back-translation model are trained **only on the Bible corpus** for that language.
- They are **not trained on the test data**. Many past experiments test on a random sample of 250 verses rather than whole held-out books (§6.2), so exclude those exact verses. For the strictest setting, also exclude whole chapters, because neighbouring verses share wording.
- They are **independent of every MT system being compared**.

#### 2.2.2 Phase 4 status

**Implemented** (`btmetric/features`). Each feature is prepared per language pair, and models trained on Scripture leave out the verses being scored:

- **`coverage-source`, `coverage-target`.** A symmetrized Thot eflomal aligner is trained on the pair's Bible text only. Each source (target) token counts by the best translation probability among its alignments; the score is the hypothesis's coverage minus the reference's.
  - Counting a token as covered whenever it is aligned at all let untranslated copies of the source look well covered, because the aligner aligns copied words anyway. Weighting by translation probability is the default; the binary version is kept as an option.
  - The two sides share one aligner per language pair.
- **`key-terms`, `key-terms-names`.** The share of the reference's key term renderings the hypothesis also uses. Renderings are inferred (§2.2.1, confidence ≥ 0.6), and matching tolerates suffixes by length: none for renderings of 3 letters or fewer, 2 for 4–5 letters, 3 for longer ones. Verses whose reference has no key term get no score. The two features share one set of inferred renderings.
- **`perplexity`.** The two-sided distance between hypothesis and reference per-character log-perplexity, under a Witten-Bell character 5-gram model of the target Bible. It trains in about 6 seconds per Bible.

**Rendering specificity filter.** Alignment sometimes infers function words as renderings, e.g. Swahili *yake* "his" or *wala* "nor". Three filters were tried against the curated renderings; recall is measured against the same set of findable terms throughout:

| Filter | Precision | Recall | Names | Other terms |
|---|---|---|---|---|
| none | 0.683 | 0.667 | 0.796 | 0.463 |
| rendering in ≤ 2× as many verses as its term | 0.717 | 0.459 | 0.812 | 0.427 |
| ≥ 20% of the rendering's verses are its term's | 0.707 | 0.483 | 0.802 | 0.452 |
| **same, counting namesakes' verses (used)** | **0.729** | 0.622 | **0.820** | **0.495** |

The first two filters lose about a third of the correct renderings for a small precision gain. Most of the lost ones are rare proper names. The Major list treats namesakes (John the Baptist, John the apostle) as separate terms sharing one rendering, so that rendering looks unspecific to each of them. The filter now compares a rendering's verses with those of every term inferred with that rendering.

At the features' confidence threshold of 0.6, it raises precision from 0.766 to **0.803** and lowers recall only from 0.559 to 0.534.

**Fixes found while evaluating:**

- *Extract names.* Experiment configs don't always name an extract as it is stored: some add a suffix (`qac-ABCD-QAC` is `qac-ABCD.txt`), and many extracts are dated. The lookup now tries the exact name, its dated versions, and the name with trailing segments removed, before falling back to the language's selected corpus.
- *Missing scores.* The challenge-set evaluation counted a row with no feature score as a failure. Such rows are now left out, and the number of scored rows is reported. Feature results computed before this fix are superseded.

**Results on the synthetic challenge set** (71,737 rows, 58 test sets; share of rows where the feature strictly prefers the unperturbed output):

| Perturbation | chrF3 | BLEU | coverage-source | coverage-target | perplexity | key-terms | **combined** |
|---|---|---|---|---|---|---|---|
| addition | 0.49 | 0.89 | 0.35 | **0.98** | 0.68 | 0.00 | **0.95** |
| repetition | 0.62 | 0.86 | 0.27 | **0.98** | 0.56 | 0.00 | **0.96** |
| omission | 0.98 | 0.84 | **0.96** | 0.47 | 0.62 | 0.42 | 0.94 |
| truncation | 1.00 | 0.86 | **0.98** | 0.48 | 0.67 | 0.40 | 0.97 |
| key term | 0.96 | 0.83 | 0.91 | 0.93 | 0.59 | **0.70** | **0.98** |
| negation | 0.89 | 0.67 | 0.84 | 0.65 | 0.50 | 0.03 | 0.81 |
| source copy | 0.98 | 0.95 | 0.89 | 0.89 | **0.99** | 0.70 | 0.97 |
| word order | 0.93 | 0.80 | 0.70 | 0.71 | 0.72 | 0.06 | 0.93 |
| wrong verse | 0.98 | 0.95 | 0.97 | 0.97 | 0.69 | 0.73 | **0.99** |
| **mean** | 0.87 | 0.85 | 0.76 | 0.79 | 0.67 | 0.34 | **0.945** |

- **Each sub-score responds to its own error types.** coverage-target to additions and loops, coverage-source to omissions and truncation, perplexity to untranslated text, key terms to key-term errors (more strongly when severe: 0.66 → 0.81). Each is near zero on, or even moves the wrong way on, the other errors. Key terms tie on errors that don't touch terms, which counts as a miss here.
- **Combined.** The pairwise log-linear model (§4.1) is validated by leaving one test set out, so each language is scored by weights fit on other languages. It reaches 0.945 against chrF3's 0.868, the gains coming where chrF3 is blind (additions 0.49 → 0.95, repetition 0.62 → 0.96). It gives up some negation sensitivity (0.89 → 0.81), because BLEU's weight pulls toward its weakness there.
- **Weights** fit on all test sets: chrF3 0.23, BLEU 0.24, coverage-target 0.22, coverage-source 0.16, perplexity 0.09, key terms 0.06. They come from synthetic errors in equal proportions, so they show that the features combine; the final weights must come from human judgments (Phase 7).
- **Key terms' value is not measured yet.** Their likely contribution is severity: a wrong divine name matters more than a wrong common word. The challenge set only asks which side is better, and WMT text has few key terms.

**Results on real errors** (2,793 pairs of real MT outputs):

| Real error | chrF3 | BLEU | coverage-source | coverage-target | perplexity |
|---|---|---|---|---|---|
| over-long output | 0.50 | 0.82 | 0.35 | **0.79** | 0.61 |
| over-short output | **0.95** | 0.84 | 0.77 | 0.25 | 0.58 |
| decoder loop | 0.60 | 0.62 | 0.52 | **0.64** | 0.55 |
| source copy | **0.72** | 0.59 | 0.27 | 0.24 | 0.63 |

Real loops are hard for every metric, because a real loop comes with other errors and the "clean" alternative isn't always clearly better. The real source-copy cases come almost entirely from **closely related language pairs**: Cebuano → Bantayanon, Standard Arabic → Hijazi Arabic, two Quechua varieties, Tibetan under two codes. There an output close to the source can be largely valid, so these labels are unreliable, and coverage's high scores are arguably right. Many projects translate from a related major language, so the metric must stay valid when much of the right answer looks like the source.

**Results against human judgments, perplexity alone** (superseded, see §3.2: the character model was trained on Bible text but scored news):

- System-level pairwise accuracy on WMT MQM: **0.66** (0.63–0.69), against chrF3 0.72 and COMET-22 0.78.
- Segment-level acc_eq on WMT 2021–22 MQM: **0.46**, slightly above chrF3 (0.42–0.44) and BLEU, near COMET-22.
- AfriMTE, Kendall's tau: 0.05. With one system per language pair, there is little variation in fluency for perplexity to pick up.

A naturalness signal trained only on Scripture tracks human judgments of news translation about as well as reference overlap does at segment level, independently of it.

**Results against human judgments, coverage alone** (superseded, see §3.2: the aligners were trained on Bible text but scored news; the in-domain rerun below shows the system-level result is not mainly a domain effect):

| | coverage-source | coverage-target |
|---|---|---|
| System-level pairwise accuracy, WMT MQM | **0.39** (0.36–0.43) | **0.37** (0.34–0.40) |
| Segment-level acc_eq, WMT 2021–22 MQM | 0.45–0.46 | 0.45–0.46 |
| Segment-level acc_eq, WMT DA | 0.38–0.46 | 0.39–0.48 |

At system level both are *below chance*: they tend to rank the human-preferred news system lower. At segment level they slightly beat chrF3 (0.42–0.44) on the expert-annotated MQM sets and trail it on DA.

**Results against human judgments, in-domain training** (`btmetric-fit-combination-human --training-domain news`, October 2026).
- *Setup:* aligners and language models are trained on Bible-sized in-domain news text (§3.2). Every WMT MQM segment of every system is scored, plus sampled DA sets.
- *Combination:* the pairwise log-linear model is fit on within-item hypothesis pairs labelled by human preference. Each collection is scored with weights fit on the others.
- *Segment-level statistic:* the share of within-item pairs ordered as the humans ordered them. Ties count as misses, which understates features that often tie (such as the one-sided perplexity).

| Collection | **combined** | chrF3 | BLEU | perplexity | coverage-target | coverage-reference | coverage-source |
|---|---|---|---|---|---|---|---|
| WMT 2022 MQM | **0.601** | 0.598 | 0.563 | 0.531 | 0.421 | 0.361 | 0.380 |
| WMT 2021 MQM | **0.541** | 0.537 | 0.509 | 0.503 | 0.425 | 0.380 | 0.404 |
| WMT 2020 MQM | 0.579 | 0.579 | 0.553 | 0.550 | 0.422 | 0.419 | 0.401 |
| WMT DA 2017–22 | chrF3 + 0.000–0.005 | 0.47–0.63 | 0.35–0.56 | 0.47–0.54 | 0.39–0.49 | 0.29–0.46 | 0.34–0.46 |

- **System level, WMT MQM:** combined **0.752** (0.72–0.78) vs chrF3 0.747 (0.72–0.78), BLEU 0.740, perplexity 0.661, coverage-reference 0.530, coverage-source 0.406, coverage-target 0.370.
- **Weights fit on all collections:** chrF3 0.74, perplexity-one-sided 0.10, BLEU 0.09, perplexity 0.05, coverage-reference 0.01, coverage-source and coverage-target **0**.
- **Low-resource DA pairs** (gu, ta, kk, is, km, ha, ps): chrF3 is again the strongest single signal (0.50–0.70), and coverage stays near 0.5. These pairs had no News Commentary, so their aligners trained on only ~1,500–2,000 in-domain pairs from the folds. xh–zu and bn–hi had too little in-domain text to train anything.

What this shows:

- **On news, the new features add nothing over chrF3, even trained in-domain.** The combination matches chrF3 within its confidence interval.
- **Coverage's poor system-level showing is not mainly a domain effect.** In-domain training barely changed it (0.39 → 0.41, 0.37 → 0.37). In its normal range, coverage rewards following the source's structure, and stronger news systems (and human references) follow it less closely.
- **WMT can't test what coverage is for.** Strong news systems rarely make the gross errors (dropped clauses, inserted passages, loops) that coverage detects and chrF3 misses, though real minority-language MT output does (§3.3, Phase 3: 3.4% of verses over-long, loops in 1.4%), and that is where coverage wins on the challenge sets.
- **Next:** coverage becomes an *error detector* (`*-detector`). It scores 0 while the hypothesis keeps at least 70% of the reference's coverage and is penalized only below that, so normal paraphrase can't count against a hypothesis. Whether detecting gross errors improves agreement with human judgments of **Bible** MT cannot be settled on news data. That makes Bible-domain human judgments (Phase 10), or judged low-resource MT with gross errors, the deciding evidence.

**Human-annotated omissions and additions** (`btmetric/analysis/mqm_error_categories.py`).
- *Data:* Google's raw WMT MQM release keeps each annotation's category and severity (en-de and zh-en 2020–22; en-ru 2022). The categories are case-normalized, since capitalization differs by year.
- *Pairs:* an output marked with Accuracy/Omission (or Addition) is paired with another system's output for the same source that every annotator marked No-error.
- *Training:* aligners use in-domain News Commentary samples.

| Feature | All omissions (4,823) | Major omissions (3,247) | All additions (1,396) | Major additions (718) |
|---|---|---|---|---|
| **chrF3** | **0.69** | **0.74** | 0.60 | 0.58 |
| BLEU | 0.61 | 0.63 | **0.62** | **0.63** |
| coverage-target | 0.39 | 0.38 | 0.58 | 0.58 |
| coverage-source | 0.45 | 0.45 | 0.32 | 0.35 |
| coverage-reference | 0.43 | 0.43 | 0.31 | 0.32 |
| coverage-reference-detector | 0.23 (65% ties) | 0.24 (63% ties) | 0.12 (76% ties) | 0.08 (83% ties) |
| coverage-target-detector | 0.13 (76% ties) | 0.12 (76% ties) | 0.12 (78% ties) | 0.14 (76% ties) |

Values are the share of pairs where the feature prefers the clean output. The rest split between ties and preferring the faulty one. Restricting to Major errors makes chrF3 *better* (0.69 → 0.74 on omissions) and leaves coverage unchanged. Even major human-marked omissions in fluent MT are usually a phrase within a long sentence, which reference overlap captures better than word alignment trained on 30,000 sentence pairs.

**Decision on coverage.** Across all the evidence:

| Evidence | Coverage beats chrF3? |
|---|---|
| Synthetic perturbations (additions, loops, omissions) | yes, by a wide margin |
| Real minority-language outputs flagged by length (over-long) | yes |
| Human judgments of WMT news (in-domain training) | no; the combination gives it ~0 weight |
| Human-annotated omissions and additions in WMT news | no; chrF3 is clearly better |

Coverage's advantage appears only on errors that are defined by length or by construction: whole clauses dropped, passages inserted, loops. On the errors humans mark in fluent MT, it adds nothing. So:

- Coverage stays **out of the metric's core**.
- `coverage-target-detector` is kept as an **optional guard** against loops and runaway output, which are frequent in low-resource MT and invisible to chrF3. It is excluded from the weighting until Bible-domain human judgments support it.
- The remaining effort goes to the components with clearer prospects:
  - back-translation scoring (§2.3, Phase 5), the only accuracy signal that doesn't rest on surface overlap
  - the Bible-domain human check (Phase 10), which is now the main way to settle what helps on Scripture

**Detector results** (October 2026). The detectors are silent unless the hypothesis keeps under 70% of the reference's coverage, so on mild errors they mostly tie:

| Error (Bible challenge set) | source-detector | target-detector | reference-detector | chrF3 |
|---|---|---|---|---|
| addition | 0.05 | 0.45 | 0.13 | 0.49 |
| repetition | 0.04 | 0.41 | 0.10 | 0.62 |
| omission | 0.54 | 0.09 | 0.77 | 0.98 |
| truncation | 0.64 | 0.11 | 0.89 | 1.00 |
| source copy | 0.70 | 0.71 | 0.85 | 0.98 |
| wrong verse | 0.90 | 0.89 | 0.96 | 0.98 |

- Gross errors (wrong verse, source copy, heavy truncation) are caught; mild ones mostly tie, as a guard's should.
- `coverage-target-detector` catches 41–45% of mild additions and repetitions, where chrF3 is near chance, but far fewer than continuous coverage-target (0.98).
- On WMT human judgments they add nothing: each is at chance at system level (0.47–0.52), segment acc_eq is 0.07–0.11 (almost all ties), and the refit combination gives them 0–0.03 weight (system level 0.753 vs chrF3 0.747, unchanged).

This confirms the decision above: a guard against runaway output, not a source of quality signal.

**Other human translations, Bible text** (§3.3b, Hebrew → English NASB). Share of the 22 MT systems each published translation outscores; a fair metric puts every one near 1.00:

| Translation (style) | chrF3 | COMET-22 | coverage-source | coverage-target | perplexity |
|---|---|---|---|---|---|
| ESV, RSV, NRSV, NKJV (literal) | 1.00 | 1.00 | 1.00 | 1.00 | 0.64–0.68 |
| BSB, CSB, NIV (mediating) | 1.00 | 1.00 | 0.64–0.73 | 1.00 | 0.64 |
| NET | 0.77 | 1.00 | 0.64 | 0.68 | 0.09 |
| NLT (dynamic) | 0.64 | 1.00 | 0.59 | 0.64 | 0.05 |
| GNT (dynamic) | 0.64 | 0.82 | 0.18 | 0.59 | 0.55 |
| CEV (dynamic) | 0.00 | 0.64 | 0.00 | 0.23 | 0.00 |
| KJV, WEB, LSV, YLT (archaic or very literal) | 0.64–1.00 | 0.64–1.00 | 0.64–1.00 | 0.64–1.00 | 0.00 |

- **coverage-source rewards literal translation, even in-domain.** Dynamic translations restructure the source, so fewer source words align, and CEV scores below every MT system. The WMT result is therefore not only a domain mismatch. Rewarding source coverage penalizes meaning-based translation, which many projects aim for. It should not enter the metric as it stands. An omission signal that doesn't reward literalness, such as unaligned content of the *reference*, is worth trying.
- **coverage-target is much fairer.** Only the most dynamic translations drop (NLT 0.64, GNT 0.59, CEV 0.23), and it is the feature that catches additions and loops. Keep it, with this caveat.
- **perplexity measures naturalness within the project's style.** The language model is trained on the target-language Bible, here NASB, the reference itself, so other registers score as unnatural (KJV, WEB, YLT, CEV all 0.00). Style and naturalness are closely tied, and when MT systems are compared for one project, matching the project's register is part of what makes a draft natural *for that project*. In a minority language the only Bible is usually the project's own, so this is how the feature will behave there, and it is largely the right behaviour. The alternative-translation test is a stress test that the real use case rarely meets: another register in the same language. A one-sided variant (`perplexity-one-sided`) penalizes only being *less* predictable than the reference, leaving over-predictable output (loops) to coverage-target. The human-judgment fit decides between the two.
- **This raises the value of Bible-domain human judgments (Phase 10).** WMT news can't show whether a feature that fails there would help on Scripture, and the alternative-translation test only shows bias, not usefulness.

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
info(x)  = [log P(src | x) − log P(src | ∅)] / source tokens     # ∅ = empty input
bt_score = info(hyp) − info(ref)
```

Comparing with the reference cancels per-verse difficulty and the reverse model's overall quality. In a difference the prior cancels, so its role is to flag verses where `info(ref)` is close to zero (memorization dominates); those get no score rather than a misleading one. (The ratio first planned is unstable when `info(ref)` is small.)

**Cost controls:**

- Back-translate or score the **reference once** and cache it.
- Freeze each language's back-translation model.
- Use temperature 0, a fixed example retrieval and a fixed prompt template.
- Prefer **open-weights models**: hosted models get updated or retired, which silently changes scores.

**Memorization gate.** Every back-translation option must pass the perturbation-based memorization test (§3.3a) before it counts as a feature:

- **Error survival rate:** does the injected error appear in the back-translation?
- **Wrong-verse check:** does a wrong verse back-translate to something unlike the correct source?

**Phase 5 status** (October 2026).

*Reverse NLLB scoring* (`bt-nllb`, `btmetric/features/back_translation.py`). NLLB-200 distilled 600M scores the source given the hypothesis by forced decoding; no back-translation is generated. One fp16 forward pass per text, on the GPU.

*Memorization gate* (`btmetric-memorization-gate`). This is a worst case for memorization: the references are German (GCLNR00), French (TOB) and Russian (Synod) Bibles, the kind of text NLLB was trained on, with English BSB as the source. 400 verses per language get each mild perturbation; key terms and negation words are inferred by alignment. Share of verses where the injected error lowers the score:

| Error | German | French | Russian |
|---|---|---|---|
| wrong verse | 1.00 | 1.00 | 1.00 |
| truncation | 0.91 | 0.99 | 0.98 |
| omission | 0.92 | 0.96 | 0.97 |
| key term | 0.93 | 0.92 | 0.90 |
| addition | 0.79 | 0.91 | 0.90 |
| negation | 0.81 | 0.62 | 0.93 |
| word order | 0.75 | 0.83 | 0.82 |
| repetition | 0.72 | 0.81 | 0.80 |
| source copy | 1.00 (0.00 without the copy guard) | 1.00 | 1.00 |

- **Memorization doesn't hide the errors.** No verse was flagged as memorized (`info(ref)` > 0.05 everywhere), and errors survive 72–100% of the time even on Bibles NLLB has seen. French negation is the weakest, plausibly because removing *pas* leaves *ne*.
- **Source copying broke forced decoding.** NLLB finds English given English near certain, so a copied source outscored the reference in almost every verse. A copy guard fixes it: where the hypothesis's chrF against the source exceeds the reference's by more than 20 points, the hypothesis is scored as adding no information.
- The gate passes for reverse NLLB scoring.

*Bible challenge set* (real MT outputs, §3.3a). `bt-nllb` scores only the test sets where NLLB covers both languages, about 13 of 58 (1,300–2,600 rows per error type, fewer for key terms and negation). Share of rows where the unperturbed output scores higher:

| Error | chrF3 | BLEU | bt-nllb |
|---|---|---|---|
| addition | 0.49 | 0.89 | 0.87 |
| repetition | 0.62 | 0.86 | 0.76 |
| negation | 0.89 | 0.67 | 0.88 |
| key term | 0.96 | 0.83 | 0.95 |
| omission | 0.98 | 0.84 | 0.94 |
| word order | 0.93 | 0.80 | 0.86 |
| truncation | 1.00 | 0.86 | 0.98 |
| source copy | 0.98 | 0.95 | **1.00** |
| wrong verse | 0.98 | 0.95 | **1.00** |

It catches the errors chrF3 misses: additions and repetition. It is near perfect on source copies, which confirms the copy guard, and on wrong verses.

*WMT human judgments* (news; NLLB trained on in-domain text via the judged sets' language pairs; at most 20,000 segment pairs per dataset):

| | Segment level (range over collections) | System level, MQM (95% CI) |
|---|---|---|
| chrF3 | 0.47–0.63 | 0.747 (0.717–0.776) |
| bt-nllb | 0.41–0.56, below chrF3 in every collection | **0.423** (0.388–0.457) |
| combined (leave one collection out) | equal to chrF3 within 0.01 | 0.719 (0.688–0.749) |

- On news, back-translation adds nothing. Fitted on all collections, it gets 12% of the weight (chrF3 64%, one-sided perplexity 12%, BLEU 10%, coverage 0), yet the combined metric is no better than chrF3 at segment level and slightly worse at system level.
- At system level `bt-nllb` is **below chance**: on news it tends to prefer the systems humans rated worse. The cause isn't known. One hypothesis is that some WMT systems produce text close to NLLB's own training data, which the reverse model then finds especially informative.

So back-translation is useful as a guard against the Bible-specific failures chrF3 misses (additions, repetition, copies, wrong verses), not as a general quality signal. **Next:** minority languages NLLB doesn't cover, with the fine-tuned and scratch reverse models; check the system-level result on news before giving `bt-nllb` weight in the combined metric; and the generation-based options.

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

**Baselines so far** (`btmetric-meta-evaluate`, October 2026):

- chrF3 system-level pairwise accuracy on WMT MQM 2020–22 is **0.72** (95% CI 0.69–0.75, 830 system pairs); chrF3++ is 0.73. This is close to the ~0.73 the WMT22 findings report for chrF (recalled, not checked), so the harness seems to reproduce published numbers.
- Segment-level acc_eq ranges from 0.41 to 0.62 across WMT sets.
- Kendall's tau-b on AfriMTE is 0.21.

**Gap:** none of the downloaded sets supports **system-level** evaluation in a lower-resource language. DA data with system names, from mt-metrics-eval or the WMT findings repositories, or IndicMT-Eval's 7 systems, would fill it. Until then, system-level evidence for lower-resource languages comes from the Bible-domain tests (§3.3).

**Mapping MQM categories to criteria** gives a separate human label for each sub-score:

- **Accuracy:** mistranslation, omission, addition, terminology
- **Naturalness:** fluency, grammar, unnatural phrasing
- **Clarity:** fluency/coherence, ambiguity (the closest available proxy)

### 3.2 Simulating low-resource conditions on these test sets

The human-judged sets cover languages with plenty of resources. In real use, every auxiliary model is trained on a small amount of text **in the domain it scores**: a Bible-sized corpus, and Bible text when scoring Bible translations. The simulation keeps both properties. Training data is limited to about a Bible's worth (~30k segments), and it comes from the **test set's domain**, which is news for WMT.

- **Aligner and naturalness LM** (`NewsCorpus`, `btmetric-fit-combination-human --training-domain news`):
  - Where OPUS has News Commentary for the pair, a fixed 30k-pair sample, with any sentence that appears in a test set removed: en–de, cs, ru, zh, ja, kk, ar, fr, es. That covers every WMT MQM pair.
  - Otherwise, the pair's own judged source–reference pairs, pooled across years. Each set is scored in 5 folds, so no segment's models are trained on its own source or reference.
- **Back-translation model:** the same principle applies; train it on in-domain, Bible-sized data.
- **Comparison condition:** compute the same features with full-resource models (e.g. NLLB) to measure how much each feature degrades under realistic constraints.

*Correction (October 2026).* Aligners and language models were first trained on Bible text and tested on WMT news. That is out of domain: a word aligner or language model must be trained in the domain it scores. The WMT results for coverage and perplexity from that setup (§2.2.2) are confounded and superseded by the in-domain rebuild. Bible-domain results (challenge sets, alternative human translations) are unaffected.

**Bible coverage** (`btmetric-find-bible-corpora` over `MT/scripture` and `eBible/MT/scripture`, October 2026):

- 32 of the 34 test-set languages have a full or near-full Bible.
- Estonian and Icelandic have essentially only the NT; Lithuanian has about 7,100 verses.
- Xhosa and Zulu have no extract, so the xh–zu and zu–xh DA sets can't be simulated.
- English uses `en-BSB` instead of the most complete extract, which is a back-translation.

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

**Phase 3 status** (October 2026; `btmetric/perturbations`, `btmetric/terms`):

- **Perturbations.** All work in any script, including ones without spaces between words:
  - clause omission, addition (a clause from a nearby verse), word-order shuffle, wrong verse (a nearby verse), source copy, repetition, truncation
  - key-term substitution (another term of the same kind, e.g. one person's name for another)
  - negation removal, using negation words inferred by alignment: target words aligned to English *not/no/never/…* in at least half their occurrences. Languages that negate with affixes yield no markers.
- **Challenge set.** `btmetric-build-challenge-set` applies every perturbation to a randomly chosen real MT output per test set: 60 held-out-book test sets, one per target language, 100 verses each. Key terms are curated where available and otherwise inferred (§2.2.1). Test verses are excluded from the alignment. `--perturb-reference` perturbs the human reference instead, for the memorization gate.
- **First results.** Share of verses where the metric prefers the unperturbed output; 38,714 rows, built before inferred key terms and negation were added:

  | Perturbation | chrF3 | BLEU | spBLEU |
  |---|---|---|---|
  | addition | **0.49** | 0.86 | 0.88 |
  | repetition | **0.72** | 0.93 | 0.95 |
  | omission | 0.98 | 0.80 | 0.87 |
  | key term | 0.93 | 0.75 | 0.88 |
  | word order | 0.94 | 0.82 | 0.89 |
  | truncation | 1.00 | 0.91 | 0.96 |
  | source copy | 0.98 | 0.95 | 0.98 |
  | wrong verse | 0.98 | 0.95 | 0.98 |

  chrF3's recall weighting (beta = 3) makes added content almost free: it is at chance on additions.
- **Calibration against real MT errors.** `btmetric-measure-error-rates` covers 210 outputs from 58 held-out-book test sets. Share of verses with each signature (mean / median):

  | Signature | Mean | Median |
  |---|---|---|
  | output > 150% of reference length (addition) | 3.4% | 1.5% |
  | source copy (concentrated in a few test sets) | 2.3% | 0% |
  | back-to-back repeated spans (decoder loops) | 1.4% | 1.0% |
  | output < 60% of reference length (omission or truncation) | 1.6% | 0.7% |

  Repeated n-grams anywhere in the verse occur in 6.2% of verses, but a spot check found about a third of those are legitimate parallelism, so the back-to-back count is the calibrated one. Weighting accuracy by these rates gives spBLEU 0.92, BLEU 0.90 and **chrF3 0.75**: on the error mix real outputs show, chrF3 does worst of the lexical baselines. Over-long output isn't always an addition, so the addition weight is an upper bound.
- **Data quality.** Some extracts are mis-encoded on the bucket (UTF-8 decoded as cp1252, e.g. `xyz-XYZ_2026_03_18`). Rendering validation detects and skips them; they should be reported and re-extracted.
- **Still to do:**
  - rebuild the challenge set with inferred key terms and negation
  - evaluate the neural baselines on it
  - check that each sub-score responds to its own error types, once the features exist (Phase 4)

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

**Phase 2 results so far** (October 2026; `data/results/`). Still missing: TER (implemented, but sacrebleu's version takes over an hour, so it isn't run by default) and an LLM judge.

*Agreement with human judgments.* System level is pairwise accuracy on WMT MQM 2020–22, 830 system pairs. Segment level is acc_eq averaged over WMT 2021–22 sets, and Kendall's tau-b on AfriMTE.

| Metric | System level (95% CI) | WMT 21–22 segment acc_eq | AfriMTE tau-b |
|---|---|---|---|
| xCOMET-XL | **0.794** (0.767–0.823) | 0.44–0.52 | 0.257 |
| COMET-22 | 0.784 (0.758–0.814) | 0.46–0.54 | 0.255 |
| AfriCOMET | 0.778 (0.752–0.808) | 0.46–0.54 | **0.320** |
| m-chrF3 | 0.747 (0.718–0.776) | 0.41–0.52 | 0.213 |
| chrF3++ | 0.729 (0.699–0.758) | 0.41–0.52 | 0.214 |
| chrF3 | 0.723 (0.693–0.752) | 0.41–0.52 | 0.213 |
| spBLEU | 0.723 (0.693–0.754) | 0.41–0.51 | 0.182 |
| BLEU | 0.718 (0.688–0.749) | 0.41–0.44 | 0.176 |

COMET-22 was trained on WMT 2017–20 direct-assessment data, so its scores on those years are inflated; compare it on WMT 2021–22 and AfriMTE. xCOMET-XL is the strongest system-level metric, significantly better than chrF3 but not than COMET-22. At segment level it leads on the MQM sets it was trained toward (2022 acc_eq 0.50 vs 0.46), trails slightly on DA, and is no better than COMET-22 on African languages, where AfriCOMET stays ahead. The lexical metrics are statistically indistinguishable from one another at system level.

*Baselines on real minority-language system pairs* (`btmetric-score-system-pairs`). All 49,525 usable config pairs from past silnlp experiments; no human labels.

| | chrF3 | m-chrF3 | BLEU | spBLEU |
|---|---|---|---|---|
| chrF3 | — | 0.96 | **0.87** | 0.92 |
| BLEU | 0.87 | 0.88 | — | 0.92 |

- **Choice of metric changes decisions.** chrF3 and BLEU pick different winners on 12.7% of real system comparisons, and chrF3 and spBLEU on 8.5%.
- **Many differences are not significant** (paired t-test on verse-level chrF3, p < 0.05):

  | Test set | Pairs | Significant |
  |---|---|---|
  | ≤ 100 verses | 940 | 36% |
  | 101–250 verses (the usual random test set) | 37,142 | 61% |
  | 1,000–5,000 verses | 6,395 | 84% |
  | > 5,000 verses | 1,303 | 81% |
  | held-out books (any size) | 9,411 | 77% |
  | random verses (any size) | 39,984 | 60% |

  So at the standard 250-verse random test size, about 40% of system comparisons can't be told apart. That makes significance reporting (§5) essential and argues for larger held-out-book test sets.

*Failure mode: valid human translations are scored below MT* (§3.3b).

- **Bible.** Test set: Hebrew OT → English NASB, 22 MT systems, 3,000 held-out verses. The table gives the share of MT systems each published translation outscores:

  | Translation | chrF3 | BLEU | COMET-22 | AfriCOMET | xCOMET-XL |
  |---|---|---|---|---|---|
  | ESV, NKJV, RSV, NRSV, BSB, CSB, WEB, NIV (same tradition as NASB) | 100% | 64–100% | 100% | 100% | 100% |
  | NET | 77% | 64% | 100% | 100% | 100% |
  | NLT | 64% | 64% | 100% | 100% | 100% |
  | GNT | 64% | 50% | 82% | 68% | **100%** |
  | LSV (literal) | 64% | 64% | 64% | 68% | **91%** |
  | YLT (literal, archaic) | 64% | 64% | 64% | 64% | 64% |
  | CEV | **0%** | **0%** | 64% | 64% | **86%** |

  Every lexical metric scores CEV below every MT system. COMET-22 helps with NLT and NET, but about 8 MT systems still outscore CEV, LSV, YLT and partly GNT. **xCOMET-XL comes closest to the desired behaviour.** It puts every modern translation, literal or dynamic, above all MT systems, except CEV, which about 3 MT systems still outscore. Archaic YLT loses to about 8 MT systems under every metric. The new metric needs to fix this.

  This also bears on silver labels (§3.3c): xCOMET-XL is a reasonable silver label for major languages, far better than chrF3 or COMET-22 at not penalizing freer valid translations. It isn't perfect, though, so it should be one signal among several.
- **WMT.** 11 alternative human translations (excluding the one used as reference) were judged with MQM alongside MT systems. MQM raters ranked them 3.5th on average out of about 14 systems; every lexical metric ranked them about 11th. Typical cases: MQM 1st → chrF3 9th of 10; MQM 2nd → 16th of 16.

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
| Key term accuracy | Major Biblical Terms assets (`Major-metadata`, `Major-vrefs`, en/fr/es/id/pt glosses); curated renderings in `MT/terms` for ~1,100 projects; Thot aligners in `silnlp/alignment/machine_aligner.py`; preliminary proper-noun work in `scripts/IDQ/key_terms.ipynb` | Rendering inference by alignment (§2.2.1) for projects without renderings; matching that copes with morphology |
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

**Shared test sets.** A system pair is only valid if both systems were scored on the **same verses against the same reference**. The same series and language don't guarantee that, because data, seeds and test sizes change. `btmetric-index-experiments` (4 levels deep) groups outputs by a hash of `test.vref.txt` + `test.trg.detok.txt`, and `btmetric-find-system-pairs` pairs them (October 2026):

- 15,555 system outputs in 10,205 experiment directories with 6,728 distinct test sets. That is far more than the old index lists, which confirms it's out of date.
- 58,941 pairs of distinct outputs on identical test sets. Pairs with identical predictions are already removed. Each pair is labelled by how the systems differ:

| Pair type | Held-out books | Random verses |
|---|---|---|
| Different configs | **9,587** pairs, 280 test sets, ~122 target languages, median 837 verses | **41,028** pairs, 1,234 test sets, ~412 languages, median 250 verses |
| Checkpoints of one experiment | 900 | 4,520 |
| Reruns of one config | 310 | 2,452 |

- The configs most often differ in source texts, model and learning rate. The differing config keys are recorded for each pair, so results can be broken down by type of difference.
- `btmetric-measure-pair-differences` counts the verses on which each pair's outputs differ.
  - Most pairs differ almost everywhere: the median config pair differs on 99.6% of verses, a rerun pair on 95.6%, a checkpoint pair on 93.9%.
  - 52 pairs differ on under 10% of verses and are flagged `near_identical`.
  - 734 pairs, from 43 test sets, have predictions whose line count doesn't match the test set (outputs with extra line breaks or merged verses). They can't be aligned verse by verse and are left out.
  - That leaves **58,155 usable pairs**: 9,585 config pairs on held-out books and 40,393 on random verses, plus checkpoint and rerun pairs.

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
- **Key term coverage:** most projects have no curated renderings, so the feature depends on how accurately alignment infers them, which varies by term category and corpus size. §2.2.1's validation against curated renderings measures this.
- **Goodhart effects:** if the metric drives system selection, systems may overfit to its features. Keep the perturbation suite and the human-translation test as regression checks.
- **Compute scheduling:** only one remote ClearML job at a time, so large training sweeps take longer.
- **Test-set design in past experiments:** random 250-verse test sets give optimistic scores and limited statistical power, and they complicate keeping test data out of auxiliary-model training.

---

## 9. Open Questions

- How accurately does alignment infer key term renderings, by term category and corpus size, compared with curated renderings?
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
