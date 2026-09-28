"""Subword tokenizers: English keeps Auto-AVSR's unigram5000 (the decoder
was trained on it); a new language gets its own SentencePiece unigram model
trained on its clip texts, with the vocabulary sized to the data (small
data, small vocabulary), written the way Auto-AVSR's spm/train.sh writes
it: `<unk> 1`, then every piece numbered from 2.

    spm/<lang>/unigram<N>.model, unigram<N>_units.txt
"""
from __future__ import annotations

import os
from typing import Tuple

from .common import DIRS, say
from .manifest import Tokenizer, texts_for_tokenizer


def auto_avsr_dir() -> str:
    return os.environ.get("THELIP_AUTO_AVSR") or os.path.join(DIRS["tools"], "auto_avsr")


def english_tokenizer() -> Tuple[str, str]:
    base = os.path.join(auto_avsr_dir(), "spm", "unigram")
    return os.path.join(base, "unigram5000.model"), os.path.join(base, "unigram5000_units.txt")


def vocab_size_for(n_sentences: int, language: str) -> int:
    if language == "zh":
        return min(3000, max(500, n_sentences // 20))
    return min(2000, max(300, n_sentences // 30))


def train_tokenizer(language: str) -> Tuple[str, str]:
    """Trains (once) and returns (model path, units path) for the language."""
    if language == "en":
        return english_tokenizer()
    import sentencepiece as spm

    corpus = texts_for_tokenizer(language)
    n = sum(1 for _ in open(corpus, encoding="utf-8"))
    vocab = vocab_size_for(n, language)
    out_dir = os.path.join(DIRS["spm"], language)
    prefix = os.path.join(out_dir, f"unigram{vocab}")
    model, units = prefix + ".model", prefix + "_units.txt"
    if os.path.isfile(model) and os.path.isfile(units):
        return model, units
    say(f"{language}: training a unigram tokenizer with {vocab} pieces on {n} sentences")
    spm.SentencePieceTrainer.Train(
        input=corpus, model_prefix=prefix, vocab_size=vocab, model_type="unigram",
        character_coverage=1.0, input_sentence_size=10_000_000, shuffle_input_sentence=True,
        bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1)
    sp = spm.SentencePieceProcessor(model_file=model)
    pieces = sorted({p for line in open(corpus, encoding="utf-8") for p in sp.encode(line.strip(), out_type=str)})
    with open(units, "w", encoding="utf-8") as f:
        f.write("<unk> 1\n")
        for i, piece in enumerate(pieces, start=2):
            f.write(f"{piece} {i}\n")
    return model, units


def tokenizer_for(language: str) -> Tokenizer:
    model, units = train_tokenizer(language)
    return Tokenizer(model, units)
