"""Back-translation scoring (PLAN.md 2.3): how much a hypothesis tells a reverse model about the source.

No back-translation is generated. A reverse (target -> source) NLLB model scores the source given the hypothesis
by forced decoding, and the probability it gives the source with no input at all is subtracted:

    info(x) = [log P(src | x) - log P(src | empty)] / source tokens

The subtracted term is what the model already "knows" about the source; NLLB was trained on Bible translations.
It cancels out of the feature, info(hyp) - info(ref), but it identifies memorized verses: where info(ref) is near
zero, the reference tells the model nothing it didn't expect, and the verse gets no score.

A hypothesis that copies the source makes the source near certain, so it would outscore any translation. Where a
hypothesis overlaps the source much more than the reference does, it is scored as telling the model nothing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from sacrebleu.metrics import CHRF

from .base import Feature

LOGGER = logging.getLogger(__name__)


class NllbLanguages:
    """NLLB-200 language codes for the ISO 639-1/639-3 codes used by the test sets and Bible extracts."""

    # Two-letter codes, and macrolanguages whose NLLB code is one member language.
    _ALIASES = {
        "af": "afr", "am": "amh", "ar": "arb", "ara": "arb", "as": "asm", "az": "azj", "aze": "azj", "be": "bel",
        "bg": "bul", "bn": "ben", "bo": "bod", "bs": "bos", "ca": "cat", "cs": "ces", "cy": "cym", "da": "dan",
        "de": "deu", "el": "ell", "en": "eng", "eo": "epo", "es": "spa", "et": "est", "ekk": "est", "eu": "eus",
        "fa": "pes", "fas": "pes", "fi": "fin", "fj": "fij", "fr": "fra", "ga": "gle", "gd": "gla", "gl": "glg",
        "gn": "grn", "gu": "guj", "ha": "hau", "he": "heb", "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun",
        "hy": "hye", "id": "ind", "ig": "ibo", "is": "isl", "it": "ita", "ja": "jpn", "jv": "jav", "ka": "kat",
        "kk": "kaz", "km": "khm", "kn": "kan", "ko": "kor", "ky": "kir", "lg": "lug", "ln": "lin", "lo": "lao",
        "lt": "lit", "lv": "lvs", "lav": "lvs", "mg": "plt", "mlg": "plt", "mi": "mri", "mk": "mkd", "ml": "mal",
        "mn": "khk", "mon": "khk", "mr": "mar", "ms": "zsm", "msa": "zsm", "mt": "mlt", "my": "mya", "ne": "npi",
        "nep": "npi", "nl": "nld", "nb": "nob", "no": "nob", "ny": "nya", "om": "gaz", "orm": "gaz", "or": "ory",
        "ori": "ory", "pa": "pan", "pl": "pol", "ps": "pbt", "pus": "pbt", "pt": "por", "qu": "quy", "que": "quy",
        "ro": "ron", "ru": "rus", "rw": "kin", "sd": "snd", "si": "sin", "sk": "slk", "sl": "slv", "sm": "smo",
        "sn": "sna", "so": "som", "sq": "als", "sqi": "als", "sr": "srp", "st": "sot", "su": "sun", "sv": "swe",
        "sw": "swh", "swa": "swh", "ta": "tam", "te": "tel", "tg": "tgk", "th": "tha", "ti": "tir", "tk": "tuk",
        "tl": "tgl", "tn": "tsn", "tr": "tur", "ts": "tso", "tw": "twi", "uk": "ukr", "ur": "urd", "uz": "uzn",
        "uzb": "uzn", "vi": "vie", "wo": "wol", "xh": "xho", "yi": "ydd", "yid": "ydd", "yo": "yor", "zh": "zho",
        "cmn": "zho", "zu": "zul",
    }  # fmt: skip

    def __init__(self) -> None:
        from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

        self._codes: Dict[str, str] = {}
        # A few languages come in several scripts. Prefer Latin, which most of their Bibles use, and Simplified Chinese.
        for code in sorted(FAIRSEQ_LANGUAGE_CODES, key=lambda c: not c.endswith(("_Latn", "_Hans"))):
            self._codes.setdefault(code.split("_")[0], code)

    def code(self, language: str) -> Optional[str]:
        """The NLLB code for an ISO code or extract name ("de-GCLNR00"), or None if NLLB doesn't cover it."""
        iso = language.split("-")[0].lower()
        return self._codes.get(self._ALIASES.get(iso, iso))


class ForcedDecodingScorer(ABC):
    """Sums of token log probabilities of output texts given input texts, for one reverse model and direction."""

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._memo: Dict[Tuple[str, str], Tuple[float, int]] = {}

    def log_likelihoods(self, inputs: Sequence[str], outputs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Per pair: log P(output | input) and the number of output tokens scored."""
        keys = list(zip(inputs, outputs))
        missing = list(dict.fromkeys(key for key in keys if key not in self._memo))
        for start in range(0, len(missing), self._batch_size):
            batch = missing[start : start + self._batch_size]
            self._memo.update(zip(batch, self._score_batch([i for i, _ in batch], [o for _, o in batch])))
        totals = np.array([self._memo[key][0] for key in keys])
        lengths = np.array([self._memo[key][1] for key in keys])
        return totals, lengths

    @abstractmethod
    def _score_batch(self, inputs: List[str], outputs: List[str]) -> List[Tuple[float, int]]:
        """(log P(output | input), output tokens) per pair."""

    @staticmethod
    def _forced_decode(model, encoded: dict, labels, pad_id: int, skip_first: bool) -> List[Tuple[float, int]]:
        import torch

        mask = labels != pad_id
        with torch.no_grad():
            logits = model(**encoded, labels=labels.masked_fill(~mask, -100)).logits
        token_scores = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
        lengths = mask.sum(dim=1)
        if skip_first:
            # NLLB's first label token is the language tag, which is forced and says nothing about the content.
            token_scores[:, 0] = 0
            lengths = lengths - 1
        return [(float(total), int(length)) for total, length in zip(token_scores.sum(dim=1), lengths)]


class NllbScorer(ForcedDecodingScorer):
    def __init__(self, model, tokenizer, input_code: str, output_code: str, batch_size: int) -> None:
        super().__init__(batch_size)
        self._model = model
        self._tokenizer = tokenizer
        self._input_code = input_code
        self._output_code = output_code

    def _score_batch(self, inputs: List[str], outputs: List[str]) -> List[Tuple[float, int]]:
        self._tokenizer.src_lang = self._input_code
        self._tokenizer.tgt_lang = self._output_code
        encoded = self._tokenizer(
            inputs, text_target=outputs, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self._model.device)
        labels = encoded.pop("labels")
        return self._forced_decode(self._model, encoded, labels, self._tokenizer.pad_token_id, skip_first=True)


def load_seq2seq(path: str):
    """A seq2seq model for scoring: fp16 on the GPU if there is one."""
    import torch
    from transformers import AutoModelForSeq2SeqLM

    # device_count rather than is_available: with CUDA_VISIBLE_DEVICES empty, PyTorch on Windows reports CUDA as
    # available but has no device to use.
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    LOGGER.info(f"Loading {path} on {device}")
    return AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).to(device).eval()


class ReverseModels(ABC):
    """Where the reverse (target -> source) model for a language pair comes from."""

    name: str = ""

    @abstractmethod
    def scorer(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str]
    ) -> Optional[ForcedDecodingScorer]:
        """A scorer of source text given target text, or None if there is no model for the pair."""


class PretrainedNllb(ReverseModels):
    name = "nllb"

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._languages = NllbLanguages()
        self._loaded: Optional[Tuple[object, object]] = None
        self._scorers: Dict[Tuple[str, str], NllbScorer] = {}

    def scorer(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str]
    ) -> Optional[ForcedDecodingScorer]:
        source, target = self._languages.code(source_language), self._languages.code(target_language)
        if source is None or target is None:
            LOGGER.warning(f"NLLB doesn't cover {source_language} or {target_language}")
            return None
        if self._loaded is None:
            from transformers import AutoTokenizer

            self._loaded = (load_seq2seq(self._model_name), AutoTokenizer.from_pretrained(self._model_name))
        if (target, source) not in self._scorers:
            model, tokenizer = self._loaded
            self._scorers[(target, source)] = NllbScorer(model, tokenizer, target, source, self._batch_size)
        return self._scorers[(target, source)]


class BackTranslationInfo(Feature):
    """info(hyp) - info(ref) under a reverse model; see the module docstring."""

    criterion = "accuracy"

    def __init__(
        self,
        models: Optional[ReverseModels] = None,
        min_reference_info: float = 0.05,
        max_copy_margin: float = 20.0,
    ) -> None:
        self._models = models or PretrainedNllb()
        self.name = f"bt-{self._models.name}"
        self._min_reference_info = min_reference_info
        self._max_copy_margin = max_copy_margin
        self._chrf = CHRF()
        self._scorer: Optional[ForcedDecodingScorer] = None

    def prepare(self, source_language: str, target_language: str, excluded_segments=frozenset()) -> None:
        self._scorer = self._models.scorer(source_language, target_language, frozenset(excluded_segments))
        if self._scorer is None:
            LOGGER.warning(f"No reverse model for {source_language}-{target_language}; {self.name} scores are NaN")

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._scorer is None:
            return np.full(len(hypotheses), np.nan)
        info_hypothesis, info_reference = self.information(sources, hypotheses), self.information(sources, references)
        info_hypothesis[self._copies_source(sources, hypotheses, references)] = 0.0
        scores = info_hypothesis - info_reference
        scores[info_reference < self._min_reference_info] = np.nan
        return scores

    def _copies_source(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        # Measured against the reference's own overlap, so names and loanwords shared with the source don't count.
        return np.array(
            [
                self._chrf.sentence_score(hyp, [src]).score - self._chrf.sentence_score(ref, [src]).score
                > self._max_copy_margin
                for src, hyp, ref in zip(sources, hypotheses, references)
            ]
        )

    def information(self, sources: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        """Per-token log-likelihood of the source given each text, less that given no text."""
        assert self._scorer is not None
        given, lengths = self._scorer.log_likelihoods(texts, sources)
        prior, _ = self._scorer.log_likelihoods([""] * len(sources), sources)
        return (given - prior) / np.maximum(lengths, 1)
