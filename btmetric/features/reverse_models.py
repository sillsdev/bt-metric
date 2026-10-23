"""Reverse (target -> source) models trained per project, for back-translation scoring of languages NLLB lacks.

Two kinds, trained on the project's own Bible minus the verses about to be scored:

- NLLB fine-tuned in the reverse direction. Languages NLLB doesn't know get a new language tag, as silnlp does.
  Its English decoder has memorized the Bible, which the memorization gate measures.
- A small transformer trained from scratch, with its own SentencePiece vocabulary. It has never seen the scored
  verses in any language, so it can't have memorized them; the cost is a weaker model.

Trained models are cached under data/models/reverse, keyed by the training text and settings.
"""

import hashlib
import json
import logging
import tempfile
from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .back_translation import ForcedDecodingScorer, NllbLanguages, NllbScorer, ReverseModels, load_seq2seq
from .corpora import ParallelText, TrainingCorpus

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSettings:
    steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    label_smoothing: float = 0.1
    max_length: int = 200
    seed: int = 111


def train_seq2seq(
    model,
    encode: Callable[[List[str], List[str]], dict],
    inputs: List[str],
    outputs: List[str],
    settings: TrainingSettings,
) -> None:
    """Trains in bf16 autocast with AdamW and linear warmup and decay. `encode` returns model inputs whose labels
    mark padding with -100."""
    import torch

    rng = np.random.default_rng(settings.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / settings.warmup_steps, max(0.0, (settings.steps - step) / settings.steps)),
    )
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=settings.label_smoothing)
    model.train()
    lengths = np.array([len(text) for text in inputs])
    step, running = 0, []
    while step < settings.steps:
        # Batches of similar length waste less on padding; chunks keep the order random between epochs.
        order = rng.permutation(len(inputs))
        chunks = [order[i : i + settings.batch_size * 50] for i in range(0, len(order), settings.batch_size * 50)]
        batches = [
            sorted_chunk[i : i + settings.batch_size]
            for chunk in chunks
            for sorted_chunk in [chunk[np.argsort(lengths[chunk])]]
            for i in range(0, len(sorted_chunk), settings.batch_size)
        ]
        for b in rng.permutation(len(batches)):
            batch = encode([inputs[i] for i in batches[b]], [outputs[i] for i in batches[b]])
            labels = batch.pop("labels")
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                logits = model(**batch, labels=labels).logits
            loss = loss_function(logits.float().view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running.append(loss.item())
            step += 1
            if step % 500 == 0 or step == settings.steps:
                LOGGER.info(f"Step {step}/{settings.steps}: loss {np.mean(running):.3f}")
                running = []
            if step >= settings.steps:
                break
    model.eval()


class TrainedReverseModels(ReverseModels):
    """Trains one reverse model per (training text, settings), caches it on disk, and keeps the last in memory."""

    def __init__(self, corpus: TrainingCorpus, cache_dir: Path, settings: TrainingSettings, batch_size: int) -> None:
        self._corpus = corpus
        self._cache_dir = cache_dir / self.name
        self._settings = settings
        self._batch_size = batch_size
        self._current: Optional[Tuple[str, ForcedDecodingScorer]] = None

    def scorer(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str]
    ) -> Optional[ForcedDecodingScorer]:
        text = self._corpus.parallel(source_language, target_language, excluded_segments)
        if text is None or len(text.source) < 1000:
            LOGGER.warning(f"Too little parallel text to train a reverse model for {source_language}-{target_language}")
            return None
        key = hashlib.sha1(json.dumps([text.key, asdict(self._settings)]).encode("utf-8")).hexdigest()[:16]
        if self._current is None or self._current[0] != key:
            directory = self._cache_dir / key
            if not (directory / "done").is_file():
                LOGGER.info(f"Training {self.name} reverse model on {len(text.source)} verses ({text.key})")
                self._train(text, source_language, target_language, directory)
                (directory / "done").write_text(text.key, encoding="utf-8")
            self._current = (key, self._load(directory))
        return self._current[1]

    @abstractmethod
    def _train(self, text: ParallelText, source_language: str, target_language: str, directory: Path) -> None:
        """Trains a model on the text, in the target -> source direction, and saves it to the directory."""

    @abstractmethod
    def _load(self, directory: Path) -> ForcedDecodingScorer:
        """A scorer for a saved model."""


class FineTunedNllb(TrainedReverseModels):
    name = "nllb-finetuned"

    def __init__(
        self,
        corpus: TrainingCorpus,
        base_model: str = "facebook/nllb-200-distilled-600M",
        cache_dir: Path = Path("data/models/reverse"),
        settings: TrainingSettings = TrainingSettings(steps=3000, batch_size=16, learning_rate=1e-4, warmup_steps=300),
        batch_size: int = 32,
    ) -> None:
        super().__init__(corpus, cache_dir, settings, batch_size)
        self._base_model = base_model
        self._languages = NllbLanguages()

    def _code(self, language: str) -> str:
        # Languages NLLB doesn't know get a tag of their own, which fine-tuning learns.
        return self._languages.code(language) or f"{language.split('-')[0].lower()}_New"

    def _train(self, text: ParallelText, source_language: str, target_language: str, directory: Path) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self._base_model, dtype=torch.float32)
        input_code, output_code = self._code(target_language), self._code(source_language)
        new_codes = [code for code in (input_code, output_code) if code not in tokenizer.get_vocab()]
        if new_codes:
            from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

            known_ids = tokenizer.convert_tokens_to_ids(FAIRSEQ_LANGUAGE_CODES)
            tokenizer.add_special_tokens(
                {"additional_special_tokens": new_codes}, replace_additional_special_tokens=False
            )
            model.resize_token_embeddings(len(tokenizer))
            embeddings = model.get_input_embeddings().weight.data
            # A new tag starts as the average language tag rather than at random.
            embeddings[tokenizer.convert_tokens_to_ids(new_codes)] = embeddings[known_ids].mean(dim=0)
        model.gradient_checkpointing_enable()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer.src_lang, tokenizer.tgt_lang = input_code, output_code

        def encode(inputs: List[str], outputs: List[str]) -> dict:
            encoded = tokenizer(
                inputs,
                text_target=outputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._settings.max_length,
            ).to(model.device)
            encoded["labels"] = encoded["labels"].masked_fill(encoded["labels"] == tokenizer.pad_token_id, -100)
            return encoded

        train_seq2seq(model, encode, text.target, text.source, self._settings)
        directory.mkdir(parents=True, exist_ok=True)
        model.half().save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        (directory / "codes.json").write_text(json.dumps([input_code, output_code]), encoding="utf-8")

    def _load(self, directory: Path) -> ForcedDecodingScorer:
        from transformers import AutoTokenizer

        input_code, output_code = json.loads((directory / "codes.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(directory)
        return NllbScorer(load_seq2seq(str(directory)), tokenizer, input_code, output_code, self._batch_size)


@dataclass(frozen=True)
class ScratchArchitecture:
    vocab_size: int = 8000
    layers: int = 4
    d_model: int = 512
    heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.3


class SentencePieceScorer(ForcedDecodingScorer):
    def __init__(self, model, processor, batch_size: int, max_length: int = 256) -> None:
        super().__init__(batch_size)
        self._model = model
        self._processor = processor
        self._max_length = max_length

    def _score_batch(self, inputs: List[str], outputs: List[str]) -> List[Tuple[float, int]]:
        encoded = encode_pieces(self._processor, inputs, outputs, self._max_length, self._model.device)
        labels = encoded.pop("labels")
        return self._forced_decode(self._model, encoded, labels, self._processor.pad_id(), skip_first=False)


def encode_pieces(processor, inputs: List[str], outputs: List[str], max_length: int, device) -> Dict[str, object]:
    """Input ids, attention mask and labels (each text's pieces plus end of sentence), padded with the pad id."""
    import torch

    def padded(texts: List[str]) -> "torch.Tensor":
        ids = [pieces[: max_length - 1] + [processor.eos_id()] for pieces in processor.encode(texts)]
        width = max(len(row) for row in ids)
        return torch.tensor([row + [processor.pad_id()] * (width - len(row)) for row in ids], device=device)

    input_ids = padded(inputs)
    return {"input_ids": input_ids, "attention_mask": input_ids != processor.pad_id(), "labels": padded(outputs)}


class ScratchReverseModel(TrainedReverseModels):
    name = "scratch"

    def __init__(
        self,
        corpus: TrainingCorpus,
        cache_dir: Path = Path("data/models/reverse"),
        settings: TrainingSettings = TrainingSettings(
            steps=12000, batch_size=64, learning_rate=5e-4, warmup_steps=1000
        ),
        architecture: ScratchArchitecture = ScratchArchitecture(),
        batch_size: int = 64,
    ) -> None:
        super().__init__(corpus, cache_dir, settings, batch_size)
        self._architecture = architecture

    def _train(self, text: ParallelText, source_language: str, target_language: str, directory: Path) -> None:
        import sentencepiece as spm
        import torch
        from transformers import MarianConfig, MarianMTModel

        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            corpus_path = Path(temp) / "corpus.txt"
            corpus_path.write_text("\n".join(text.source + text.target) + "\n", encoding="utf-8")
            # One vocabulary for both sides; full character coverage, since minority-language orthographies
            # often use rare characters.
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(directory / "sp"),
                vocab_size=self._architecture.vocab_size,
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=-1,
                eos_id=2,
                hard_vocab_limit=False,
                minloglevel=2,
            )
        processor = spm.SentencePieceProcessor(model_file=str(directory / "sp.model"))
        a = self._architecture
        config = MarianConfig(
            vocab_size=processor.get_piece_size(),
            decoder_vocab_size=processor.get_piece_size(),
            d_model=a.d_model,
            encoder_layers=a.layers,
            decoder_layers=a.layers,
            encoder_attention_heads=a.heads,
            decoder_attention_heads=a.heads,
            encoder_ffn_dim=a.ffn_dim,
            decoder_ffn_dim=a.ffn_dim,
            dropout=a.dropout,
            max_position_embeddings=512,
            pad_token_id=0,
            eos_token_id=2,
            decoder_start_token_id=0,
            share_encoder_decoder_embeddings=True,
        )
        torch.manual_seed(self._settings.seed)
        model = MarianMTModel(config).to("cuda" if torch.cuda.is_available() else "cpu")

        def encode(inputs: List[str], outputs: List[str]) -> dict:
            encoded = encode_pieces(processor, inputs, outputs, self._settings.max_length, model.device)
            encoded["labels"] = encoded["labels"].masked_fill(encoded["labels"] == processor.pad_id(), -100)
            return encoded

        train_seq2seq(model, encode, text.target, text.source, self._settings)
        model.save_pretrained(directory)

    def _load(self, directory: Path) -> ForcedDecodingScorer:
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_file=str(directory / "sp.model"))
        return SentencePieceScorer(load_seq2seq(str(directory)), processor, self._batch_size)
