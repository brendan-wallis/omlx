# SPDX-License-Identifier: Apache-2.0
"""Bounded KV storage for Inkling 2's sliding-window layers.

Sliding layers only read the last ``sliding_window_size`` keys, so storing
the whole history wastes memory that grows with the context. These caches
keep a window plus a little slack, always in temporal order:

* The attention kernels index keys by backward distance from the queries
  (``queries are the last L keys``), with no positional encoding baked into
  the keys. A ring buffer would hand them keys out of order; a temporal
  buffer hands them exactly the slice a full-history cache would have, so
  the result is the same, bit for bit.
* The buffer's first row stays aligned to ``ALIGN`` absolute positions, so
  the flash prefill kernel tiles the window the same way it does over a
  full-history cache.
* Rows are dropped only when the buffer is full, and at least
  ``max_size - 1 + slack`` rows are kept, so speculative verify can trim
  rejected drafts and the next query still sees a full window.

``Inkling2WindowKVCache`` subclasses ``RotatingKVCache`` so oMLX's prefix
cache, SSD cache, block alignment and memory estimator treat it as a
sliding-window layer with window ``max_size``. Restored snapshots come back
as stock rotating caches; ``adopt`` converts them (and stock batch rotating
caches) before the model reads them.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, RotatingKVCache

# Multiple of the flash kernel's 16-key tile.
ALIGN = 64
# Rows kept beyond the window, so a speculative verify (or a one-token
# re-feed) can be trimmed without losing keys the next query reads.
DEFAULT_SLACK = 64


def _round_up(n: int, step: int) -> int:
    return ((n + step - 1) // step) * step


class Inkling2WindowKVCache(RotatingKVCache):
    """Temporal-order sliding-window KV cache for one sequence."""

    step = 256

    def __init__(self, max_size: int, slack: int = DEFAULT_SLACK):
        super().__init__(max_size=max_size, keep=0)
        self.slack = int(slack)

    # ``_idx`` is the number of rows in use; row 0 holds absolute position
    # ``start_position``. Rows before the first real key (alignment padding)
    # are zeros that every query sees as out of window.
    @property
    def start_position(self) -> int:
        return int(self.offset) - int(self._idx)

    @classmethod
    def from_rows(
        cls,
        keys: mx.array | None,
        values: mx.array | None,
        offset: int,
        max_size: int,
        slack: int = DEFAULT_SLACK,
    ) -> "Inkling2WindowKVCache":
        """A cache whose ``keys``/``values`` hold the latest positions up to
        ``offset`` in temporal order."""
        cache = cls(max_size, slack)
        cache.offset = int(offset)
        if keys is None or keys.shape[2] == 0:
            return cache
        rows = int(keys.shape[2])
        start = int(offset) - rows
        if start < 0:
            raise ValueError(f"{rows} rows exceed offset {offset}")
        pad = start % ALIGN
        if pad and start > 0:
            zk = mx.zeros((*keys.shape[:2], pad, keys.shape[3]), keys.dtype)
            zv = mx.zeros((*values.shape[:2], pad, values.shape[3]), values.dtype)
            keys = mx.concatenate([zk, keys], axis=2)
            values = mx.concatenate([zv, values], axis=2)
            rows += pad
        cap = _round_up(rows + cls.step, cls.step)
        cache.keys = mx.zeros((*keys.shape[:2], cap, keys.shape[3]), keys.dtype)
        cache.values = mx.zeros(
            (*values.shape[:2], cap, values.shape[3]), values.dtype
        )
        cache.keys[..., :rows, :] = keys
        cache.values[..., :rows, :] = values
        cache._idx = rows
        return cache

    @classmethod
    def from_cache(cls, other: Any, max_size: int | None = None):
        """Convert a stock ``RotatingKVCache`` (e.g. an SSD restore)."""
        max_size = int(max_size or other.max_size)
        if other.keys is None:
            return cls.from_rows(None, None, int(other.offset), max_size)
        if getattr(other, "keep", 0):
            raise ValueError("Inkling 2 sliding layers keep no sink tokens")
        keys = other._temporal_order(other.keys)
        values = other._temporal_order(other.values)
        rows = min(int(keys.shape[2]), int(other.offset))
        return cls.from_rows(
            keys[..., keys.shape[2] - rows :, :],
            values[..., values.shape[2] - rows :, :],
            int(other.offset),
            max_size,
        )

    def _history_floor(self) -> int:
        """Rows that must stay behind the next query."""
        return self.max_size - 1 + self.slack

    def _drop_and_reserve(self, keys: mx.array, values: mx.array) -> None:
        incoming = int(keys.shape[2])
        if self.keys is not None and self._idx + incoming <= self.keys.shape[2]:
            # A prefill chunk leaves the buffer chunk-sized; shrink it once
            # decode starts instead of carrying it for the whole request.
            bloated = self.keys.shape[2] > (
                self._history_floor() + ALIGN + incoming + 2 * self.step
            )
            if not (bloated and incoming < self.step):
                return
        keep_from = 0
        if self.keys is not None:
            target = max(0, int(self.offset) - self._history_floor())
            target -= target % ALIGN
            keep_from = max(0, target - self.start_position)
        kept = self._idx - keep_from
        cap = _round_up(kept + max(incoming, self.step), self.step)
        new_k = mx.zeros((*keys.shape[:2], cap, keys.shape[3]), keys.dtype)
        new_v = mx.zeros((*values.shape[:2], cap, values.shape[3]), values.dtype)
        if kept > 0:
            new_k[..., :kept, :] = self.keys[..., keep_from : self._idx, :]
            new_v[..., :kept, :] = self.values[..., keep_from : self._idx, :]
        self.keys, self.values = new_k, new_v
        self._idx = kept

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._drop_and_reserve(keys, values)
        n = int(keys.shape[2])
        prev = self._idx
        self.keys[..., prev : prev + n, :] = keys
        self.values[..., prev : prev + n, :] = values
        self._idx = prev + n
        self.offset += n
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def keys_and_values(self):
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def _temporal_order(self, v: mx.array) -> mx.array:
        return v[..., : self._idx, :] if v.shape[2] > self._idx else v

    def size(self) -> int:
        if self.keys is None:
            return 0
        return min(int(self.offset), self.max_size, self._idx)

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.offset, self.keep, self.max_size, 0
        k, v = self.keys_and_values()
        return k, v, self.offset, self.keep, self.max_size, self._idx

    @state.setter
    def state(self, v):
        if len(v) == 2:
            keys, values = v
            offset = 0 if keys is None else int(keys.shape[2])
            if not hasattr(self, "max_size"):
                raise ValueError("A two-part state needs an existing window size")
        else:
            keys, values, offset, _keep, max_size, _idx = v
            self.max_size = int(max_size)
        if not hasattr(self, "slack"):
            self.slack = DEFAULT_SLACK
        self.keep = 0
        fresh = Inkling2WindowKVCache.from_rows(
            keys, values, int(offset), self.max_size, self.slack
        )
        self.keys, self.values = fresh.keys, fresh.values
        self.offset, self._idx = fresh.offset, fresh._idx

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        # With history dropped, the next query still needs max_size - 1 rows.
        floor = 0 if self.start_position <= 0 else self.max_size - 1
        n = max(0, min(int(n), self._idx - floor))
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        # The Inkling 2 kernels build their own band from key positions.
        return None

    @classmethod
    def merge(cls, caches):
        return Inkling2BatchWindowKVCache.merge(caches)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError("Inkling 2 window KV stays in BF16")


class Inkling2BatchWindowKVCache(BatchKVCache):
    """Left-padded batch of sliding-window rows in temporal order.

    Same layout as ``BatchKVCache`` (``offset`` is each row's position,
    ``left_padding`` its first valid buffer row), except that rows older
    than the window are dropped from the front when the buffer fills.
    ``left_padding`` may go negative once a row's oldest keys are gone.
    """

    def __init__(self, left_padding, max_size: int, slack: int = DEFAULT_SLACK):
        super().__init__(left_padding)
        self.max_size = int(max_size)
        self.slack = int(slack)
        # True once any row is missing positions before the window.
        self.lost_history = False

    def _history_floor(self) -> int:
        return self.max_size - 1 + self.slack

    def update_and_fetch(self, keys, values):
        incoming = int(keys.shape[2])
        if self.keys is not None and self._idx + incoming > self.keys.shape[2]:
            # A right-padded multi-row prefill keeps every row until
            # finalize() aligns them.
            drop = 0
            if self._right_padding is None:
                drop = max(0, self._idx - self._history_floor())
            kept = self._idx - drop
            cap = _round_up(kept + max(incoming, self.step), self.step)
            new_k = mx.zeros((*keys.shape[:2], cap, keys.shape[3]), keys.dtype)
            new_v = mx.zeros((*values.shape[:2], cap, values.shape[3]), values.dtype)
            if kept > 0:
                new_k[..., :kept, :] = self.keys[..., drop : self._idx, :]
                new_v[..., :kept, :] = self.values[..., drop : self._idx, :]
            self.keys, self.values = new_k, new_v
            self._idx = kept
            if drop:
                self.left_padding = self.left_padding - drop
                self.lost_history = True
        return super().update_and_fetch(keys, values)

    def trim(self, n):
        floor = self.max_size - 1 if self.lost_history else 0
        n = max(0, min(int(n), self._idx - floor))
        self._idx -= n
        self.offset = self.offset - n
        return n

    def make_mask(self, *args, **kwargs):
        return None

    def extend(self, other):
        lost = self.lost_history or getattr(other, "lost_history", False)
        super().extend(other)
        self.lost_history = lost

    def extract(self, idx):
        mx.eval(self.left_padding, self.offset)
        padding = max(0, int(self.left_padding.tolist()[idx]))
        offset = int(self.offset.tolist()[idx])
        keys = self.keys[idx : idx + 1, :, padding : self._idx]
        values = self.values[idx : idx + 1, :, padding : self._idx]
        return Inkling2WindowKVCache.from_rows(
            keys, values, offset, self.max_size, self.slack
        )

    @classmethod
    def merge(cls, caches):
        caches = [as_window_cache(c) for c in caches]
        max_size = caches[0].max_size
        slack = caches[0].slack
        lengths = [0 if c.keys is None else c._idx for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            out = cls([0] * len(caches), max_size, slack)
            out.offset = mx.array([int(c.offset) for c in caches])
            return out
        padding = [max_length - n for n in lengths]
        ref = next(c for c in caches if c.keys is not None)
        b = len(caches)
        h, dk = ref.keys.shape[1], ref.keys.shape[3]
        dv = ref.values.shape[3]
        keys = mx.zeros((b, h, max_length, dk), dtype=ref.keys.dtype)
        values = mx.zeros((b, h, max_length, dv), dtype=ref.values.dtype)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            k, v = c.keys_and_values()
            keys[i : i + 1, :, p:] = k
            values[i : i + 1, :, p:] = v
        out = cls(padding, max_size, slack)
        out.keys, out.values = keys, values
        out.offset = mx.array([int(c.offset) for c in caches])
        out._idx = max_length
        out.lost_history = any(c.start_position > 0 for c in caches)
        return out

    def size(self):
        return self._idx

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError("Inkling 2 window KV stays in BF16")


def as_window_cache(cache: Any, max_size: int | None = None) -> Inkling2WindowKVCache:
    """``cache`` as an ``Inkling2WindowKVCache`` (converting a stock one)."""
    if isinstance(cache, Inkling2WindowKVCache):
        return cache
    if isinstance(cache, RotatingKVCache):
        return Inkling2WindowKVCache.from_cache(cache, max_size)
    raise TypeError(f"Not a sliding-window cache: {type(cache).__name__}")


def _from_batch_rotating(cache: BatchRotatingKVCache) -> Inkling2BatchWindowKVCache:
    mx.eval(cache.left_padding, cache.offset)
    out = Inkling2BatchWindowKVCache(
        [0] * int(cache.offset.shape[0]), cache.max_size
    )
    out.offset = cache.offset
    if cache.keys is None:
        out.left_padding = cache.left_padding
        return out
    # Stock batch rotating caches reach Inkling 2 from merging restored
    # snapshots, still in temporal order with their left padding.
    cache._temporal_order()
    keys, values = cache.keys[..., : cache._idx, :], cache.values[..., : cache._idx, :]
    out.left_padding = cache.left_padding
    n = int(keys.shape[2])
    cap = _round_up(n + out.step, out.step)
    out.keys = mx.zeros((*keys.shape[:2], cap, keys.shape[3]), keys.dtype)
    out.values = mx.zeros((*values.shape[:2], cap, values.shape[3]), values.dtype)
    out.keys[..., :n, :] = keys
    out.values[..., :n, :] = values
    out._idx = n
    out.lost_history = bool(mx.any(cache.offset > n).item())
    return out


def adopt(entry: Any) -> None:
    """Swap a stock rotating member of an Inkling 2 ``CacheList`` entry for
    the temporal-order window cache, in place."""
    members = getattr(entry, "caches", None)
    if not members:
        return
    kv = members[0]
    if isinstance(kv, (Inkling2WindowKVCache, Inkling2BatchWindowKVCache)):
        return
    if isinstance(kv, BatchRotatingKVCache):
        new = _from_batch_rotating(kv)
    elif isinstance(kv, RotatingKVCache):
        new = Inkling2WindowKVCache.from_cache(kv)
    else:
        return
    entry.caches = (new, *tuple(members[1:]))


__all__ = [
    "ALIGN",
    "Inkling2BatchWindowKVCache",
    "Inkling2WindowKVCache",
    "adopt",
    "as_window_cache",
]
