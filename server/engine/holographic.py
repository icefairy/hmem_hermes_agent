"""Holographic Reduced Representations (HRR) with phase encoding.

HRRs are a vector symbolic architecture for encoding compositional structure
into fixed-width distributed representations. This module uses *phase vectors*:
each concept is a vector of angles in [0, 2π). The algebraic operations are:

  bind   — circular convolution (phase addition)    — associates two concepts
  unbind — circular correlation (phase subtraction) — retrieves a bound value
  bundle — superposition (circular mean)            — merges multiple concepts

Phase encoding is numerically stable, avoids the magnitude collapse of
traditional complex-number HRRs, and maps cleanly to cosine similarity.

Atoms are generated deterministically from SHA-256 so representations are
identical across processes, machines, and runs.

Chinese support: tokenization uses jieba (same tokenizer as the FTS5 index),
so Chinese text is encoded word-by-word instead of as one opaque blob.

References:
  Plate (1995) — Holographic Reduced Representations
  NousResearch/hermes-agent plugins/memory/holographic (port adapted)
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import Any

import jieba
import numpy as np

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi
_FLOAT32_BLOB_PREFIX = b"HRR1"

# numpy is a hard requirement (declared in requirements.txt); annotations
# use Any so static checkers are not confused by the module-level import.
Array = Any


def hrr_available() -> bool:
    """HRR runs fully locally — no model API, no key. Always True here."""
    return True


# ---------------------------------------------------------------------------
# Atom & text encoding
# ---------------------------------------------------------------------------


def encode_atom(word: str, dim: int = 1024) -> Array:
    """Deterministic phase vector via SHA-256 counter blocks.

    Same scheme as Hermes: hash f"{word}:{i}" for i=0,1,... to fill uint16
    values, then scale to [0, 2π). Identical across processes and machines.
    """
    values_per_block = 16  # each SHA-256 digest = 16 uint16 values
    blocks_needed = math.ceil(dim / values_per_block)
    uint16_values: list[int] = []
    for i in range(blocks_needed):
        digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
        uint16_values.extend(struct.unpack("<16H", digest))
    phases = np.asarray(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)
    return phases


def bind(a: Array, b: Array) -> Array:
    """Circular convolution = element-wise phase addition."""
    return (a + b) % _TWO_PI


def unbind(memory: Array, key: Array) -> Array:
    """Circular correlation = element-wise phase subtraction.

    unbind(bind(a, b), a) ≈ b (modulo superposition noise).
    """
    return (memory - key) % _TWO_PI


def bundle(*vectors: Array) -> Array:
    """Superposition via circular mean of complex exponentials.

    The result is similar to each input. Capacity ~O(sqrt(dim)) before the
    signal-to-noise ratio degrades noticeably.
    """
    complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
    return np.angle(complex_sum) % _TWO_PI


def similarity(a: Array, b: Array) -> float:
    """Phase cosine similarity in [-1, 1].  1=same, ~0=unrelated."""
    return float(np.mean(np.cos(a - b)))


def _tokenize(text: str) -> list[str]:
    """Word tokens for HRR: jieba segment + strip punctuation.

    Keeps tokens of length >= 2 (drops single punctuation tokens that jieba
    occasionally emits). English words stay intact, Chinese words are cut.
    """
    if not text or not text.strip():
        return []
    tokens: list[str] = []
    for w in jieba.lcut(text.lower()):
        w = w.strip().strip(".,;:!?\"'()[]{}#@<>《》【】「」")
        if len(w) >= 2:
            tokens.append(w)
    return tokens


def encode_text(text: str, dim: int = 1024) -> Array:
    """Bag-of-words: bundle of atom vectors for each HRR token.

    Returns encode_atom("__hrr_empty__") for empty text so the vector is never
    a zero/NaN vector.
    """
    tokens = _tokenize(text)
    if not tokens:
        return encode_atom("__hrr_empty__", dim)
    atom_vectors = [encode_atom(t, dim) for t in tokens]
    return bundle(*atom_vectors)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def phases_to_bytes(phases: Array, dim: int | None = None) -> bytes:
    """Serialize phase vectors as float32 blobs (prefixed HRR1)."""
    if dim is None:
        dim = int(phases.shape[0])
    float32_blob_bytes = len(_FLOAT32_BLOB_PREFIX) + dim * np.dtype(np.float32).itemsize
    float64_bytes = dim * np.dtype(np.float64).itemsize
    if float32_blob_bytes == float64_bytes:
        # dim=1: sizes collide, write legacy float64 to stay unambiguous
        return np.asarray(phases, dtype=np.float64).tobytes()
    payload = np.asarray(phases, dtype=np.float32).tobytes()
    return _FLOAT32_BLOB_PREFIX + payload


def bytes_to_phases(data: bytes, dim: int | None = None) -> Array:
    """Deserialize phase vectors from prefixed float32 or legacy float64 blobs."""
    if not data:
        raise ValueError("empty HRR blob")
    if dim is not None:
        f32_payload = dim * np.dtype(np.float32).itemsize
        f32_bytes = len(_FLOAT32_BLOB_PREFIX) + f32_payload
        f64_bytes = dim * np.dtype(np.float64).itemsize
        if f32_bytes == f64_bytes:
            return np.frombuffer(data, dtype=np.float64).copy()
        if data.startswith(_FLOAT32_BLOB_PREFIX) and len(data) == f32_bytes:
            return np.frombuffer(
                data[len(_FLOAT32_BLOB_PREFIX) :], dtype=np.float32
            ).astype(np.float64)
        if len(data) == f64_bytes:
            return np.frombuffer(data, dtype=np.float64).copy()
        raise ValueError(f"HRR blob length {len(data)} mismatch for dim={dim}")
    if data.startswith(_FLOAT32_BLOB_PREFIX):
        payload = data[len(_FLOAT32_BLOB_PREFIX) :]
        if len(payload) % np.dtype(np.float32).itemsize != 0:
            raise ValueError("HRR float32 blob has invalid length")
        return np.frombuffer(payload, dtype=np.float32).astype(np.float64)
    if len(data) % np.dtype(np.float64).itemsize != 0:
        raise ValueError("HRR legacy blob has invalid length")
    return np.frombuffer(data, dtype=np.float64).copy()


def snr_estimate(dim: int, n_items: int) -> float:
    """SNR ≈ sqrt(dim / n_items). Warn when it drops below 2 (errors likely)."""
    if n_items <= 0:
        return math.inf
    snr = math.sqrt(dim / n_items)
    if snr < 2.0:
        logger.warning(
            "HRR storage near capacity: SNR=%.2f (dim=%d, n_items=%d)",
            snr,
            dim,
            n_items,
        )
    return snr
