# SPDX-License-Identifier: Apache-2.0
"""Bounded sliding-window KV storage for Inkling 2.

The window cache must hand the attention kernels exactly the keys a
full-history cache would, so every path (decode, masked verify, flash
prefill, batched rows) gives identical outputs while storage stays bounded.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402

from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache  # noqa: E402
from omlx.patches.inkling2 import kernels  # noqa: E402
from omlx.patches.inkling2.cache import (  # noqa: E402
    ALIGN,
    Inkling2BatchWindowKVCache,
    Inkling2WindowKVCache,
    adopt,
)

HQ, HKV, D = 4, 2, 32


def _qkv(length, b=1, seed=0, r=64):
    mx.random.seed(seed)
    q = mx.random.normal((b, HQ, length, D)).astype(mx.bfloat16) * 0.3
    k = mx.random.normal((b, HKV, length, D)).astype(mx.bfloat16) * 0.3
    v = mx.random.normal((b, HKV, length, D)).astype(mx.bfloat16)
    rel = (mx.random.normal((b, length, HQ, r)) * 0.5).astype(mx.bfloat16)
    sink = mx.random.normal((b, length, HQ)).astype(mx.bfloat16)
    return q, k, v, rel, sink


def _attend(cache, q, k, v, rel, sink, window):
    keys, values = cache.update_and_fetch(k, v)
    lp = getattr(cache, "left_padding", None)
    return kernels.banded_sink_attention(
        q, keys, values, rel, sink, scale=D**-0.5, window=window, left_padding=lp
    )


# Prefill chunks cover the flash (>= 32 queries) and masked paths; the
# 8-token blocks are speculative verifies followed by a rollback.
_SCHEDULE = [100, 37, 64, 5] + [1] * 300 + [8, ("trim", 5)] * 6 + [1] * 200 + [300]


@pytest.mark.parametrize("window", [64, 128])
def test_window_cache_matches_full_history_bit_for_bit(window):
    full, bounded = KVCache(), Inkling2WindowKVCache(window)
    seed = 0
    for step in _SCHEDULE:
        if isinstance(step, tuple):
            assert bounded.trim(step[1]) == full.trim(step[1]) == step[1]
            continue
        seed += 1
        args = _qkv(step, seed=seed, r=window)
        want = _attend(full, *args, window)
        got = _attend(bounded, *args, window)
        assert mx.array_equal(want, got), f"diverged at offset {full.offset}"
    assert bounded.offset == full.offset
    # Storage stays bounded: window + slack, alignment and one growth step.
    assert bounded.keys.shape[2] <= window + bounded.slack + ALIGN + 2 * 300
    assert bounded.start_position % ALIGN == 0
    assert bounded.start_position > 0


def test_decode_storage_stays_bounded():
    cache = Inkling2WindowKVCache(1024)
    k = mx.zeros((1, HKV, 1, D), dtype=mx.bfloat16)
    biggest = 0
    for _ in range(4000):
        cache.update_and_fetch(k, k)
        biggest = max(biggest, cache.keys.shape[2])
    mx.eval(cache.keys)
    assert cache.offset == 4000
    # Kept rows plus one growth step, rounded up to the step.
    assert biggest <= 1024 + cache.slack + ALIGN + 2 * cache.step
    assert cache._idx >= 1024 - 1 + cache.slack


def test_trim_never_drops_keys_the_next_query_reads():
    window = 64
    cache = Inkling2WindowKVCache(window)
    k = mx.zeros((1, HKV, 400, D), dtype=mx.bfloat16)
    cache.update_and_fetch(k, k)
    # Outgrowing the buffer drops rows older than the window plus slack.
    cache.update_and_fetch(k[:, :, :200], k[:, :, :200])
    assert cache.start_position > 0
    trimmed = cache.trim(10_000)
    assert trimmed == cache._idx + trimmed - (window - 1)
    assert cache._idx == window - 1
    # Before any history is dropped, trimming is unrestricted.
    early = Inkling2WindowKVCache(window)
    early.update_and_fetch(k[:, :, :20], k[:, :, :20])
    assert early.trim(20) == 20 and early.offset == 0


def _restored(cache):
    """What the SSD path rebuilds: the last max_size rows, stock rotating."""
    keys, values = cache.keys_and_values()
    rows = min(cache.max_size, keys.shape[2])
    out = PrefillReadyRotatingKVCache(max_size=cache.max_size)
    out.keys = mx.contiguous(keys[..., -rows:, :])
    out.values = mx.contiguous(values[..., -rows:, :])
    out.offset = cache.offset
    out._idx = rows
    return out


@pytest.mark.parametrize("prompt", [40, 333])
def test_restored_rotating_snapshot_is_adopted_and_matches(prompt):
    window = 64
    full, bounded = KVCache(), Inkling2WindowKVCache(window)
    args = _qkv(prompt, seed=1, r=window)
    _attend(full, *args, window)
    _attend(bounded, *args, window)
    entry = CacheList(_restored(bounded), ArraysCache(4))
    adopt(entry)
    restored = entry[0]
    assert isinstance(restored, Inkling2WindowKVCache)
    assert restored.offset == full.offset
    for seed, n in enumerate([1, 1, 8, 50, 1], start=2):
        a = _qkv(n, seed=seed, r=window)
        assert mx.array_equal(_attend(full, *a, window), _attend(restored, *a, window))


def test_batched_rows_match_full_history_batch():
    window = 64
    lengths = (300, 90)
    singles_full, singles_win = [], []
    for i, n in enumerate(lengths):
        f, w = KVCache(), Inkling2WindowKVCache(window)
        a = _qkv(n, seed=10 + i, r=window)
        _attend(f, *a, window)
        _attend(w, *a, window)
        singles_full.append(f)
        singles_win.append(w)
    full = KVCache.merge(singles_full)
    win = Inkling2WindowKVCache.merge(singles_win)
    assert isinstance(win, Inkling2BatchWindowKVCache)
    assert win.offset.tolist() == list(lengths)
    for seed in range(400):
        a = _qkv(1, b=2, seed=100 + seed, r=window)
        want = _attend(full, *a, window)
        got = _attend(win, *a, window)
        assert mx.allclose(want, got, atol=1e-6), f"step {seed}"
    assert win.keys.shape[2] <= window + win.slack + 2 * win.step
    # A row leaves the batch as a single-request window cache.
    row = win.extract(1)
    assert isinstance(row, Inkling2WindowKVCache)
    assert row.offset == lengths[1] + 400
    ref = full.extract(1)
    for seed in range(3):
        a = _qkv(1, seed=999 + seed, r=window)
        assert mx.array_equal(_attend(ref, *a, window), _attend(row, *a, window))


def test_stock_batch_rotating_member_is_adopted():
    from mlx_lm.models.cache import BatchRotatingKVCache

    window = 64
    snaps = []
    for i, n in enumerate((200, 70)):
        w = Inkling2WindowKVCache(window)
        _attend(w, *_qkv(n, seed=20 + i, r=window), window)
        snaps.append(_restored(w))
    stock = BatchRotatingKVCache.merge(snaps)
    entry = CacheList(stock, ArraysCache(4))
    adopt(entry)
    assert isinstance(entry[0], Inkling2BatchWindowKVCache)
    assert entry[0].offset.tolist() == [200, 70]
    out = _attend(entry[0], *_qkv(1, b=2, seed=30, r=window), window)
    assert out.shape == (2, HQ, 1, D)


def _tiny(**dims):
    from tests.test_inkling_vlm_mtp import _TINY_LOOP, _inkling2_tiny

    if dims.pop("looped", False):
        dims = {**_TINY_LOOP, **dims}
    return _inkling2_tiny(**dims)


@pytest.mark.parametrize("looped", [False, True], ids=["plain", "looped"])
def test_model_bounded_cache_matches_full_history(looped, monkeypatch):
    from omlx.patches.inkling2 import model as inkling2

    model = _tiny(looped=looped)
    ids = mx.random.randint(0, 300, (1, 900))

    def run(bounded):
        monkeypatch.setattr(inkling2, "BOUNDED_SLIDING_KV", bounded)
        cache = model.make_cache()
        outs = [model(ids[:, :600], cache=cache)[:, -1]]
        # The second chunk outgrows the buffer, so old rows are dropped.
        outs.append(model(ids[:, 600:800], cache=cache)[:, -1])
        for t in range(800, 840):
            outs.append(model(ids[:, t : t + 1], cache=cache)[:, -1])
        # Verify 8, keep 3 drafts, then continue.
        model(ids[:, 840:848], cache=cache, return_hidden=True, capture_layer_ids=[5])
        assert model.mtp_partial_rollback(cache, 3, 7)
        for t in range(844, 900):
            outs.append(model(ids[:, t : t + 1], cache=cache)[:, -1])
        return mx.stack(outs), cache

    want, full_cache = run(False)
    got, bounded_cache = run(True)
    assert mx.array_equal(want, got)
    sliding = [c[0] for c in bounded_cache if isinstance(c[0], Inkling2WindowKVCache)]
    assert sliding and all(c.start_position > 0 for c in sliding)
    assert all(c._idx <= 8 + c.slack + ALIGN + c.step for c in sliding)
    assert all(type(c[0]) is KVCache for c in full_cache)


def test_window_cache_is_rotating_family_for_storage():
    from omlx.cache.type_registry import CacheTypeRegistry
    from omlx.memory_monitor import collect_kv_layer_specs

    assert CacheTypeRegistry.is_rotating_family("Inkling2WindowKVCache")
    model = _tiny(looped=True)
    full, rotating, _ = collect_kv_layer_specs(model.make_cache())
    window = model.args.text.sliding_window_size
    visits = model.model.num_visits
    assert rotating == [(visits - full, window)]


def test_buffer_shrinks_when_decode_follows_a_long_chunk():
    cache = Inkling2WindowKVCache(1024)
    chunk = mx.zeros((1, HKV, 4096, D), dtype=mx.bfloat16)
    cache.update_and_fetch(chunk, chunk)
    cache.update_and_fetch(chunk, chunk)
    assert cache.keys.shape[2] > 4096
    one = chunk[:, :, :1]
    cache.update_and_fetch(one, one)
    assert cache.keys.shape[2] <= 1024 + cache.slack + ALIGN + 2 * cache.step
    for _ in range(50):
        cache.update_and_fetch(one, one)
    assert cache.offset == 8192 + 51
    assert cache._idx >= 1024 - 1 + cache.slack


def test_scheduler_snapshot_of_window_member_is_canonical_rotating(
    mock_model, mock_tokenizer
):
    """Prefix/SSD extraction stores the latest max_size rows, and the
    restored rotating cache adopts back into an identical window."""
    from omlx.cache.type_handlers import RotatingKVCacheHandler
    from omlx.scheduler import Scheduler

    window = 64
    full, live = KVCache(), Inkling2WindowKVCache(window)
    for seed, n in enumerate([300, 300, 1, 77]):
        a = _qkv(n, seed=40 + seed, r=window)
        _attend(full, *a, window)
        _attend(live, *a, window)
    assert live.start_position > 0

    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    extracted, _ = scheduler._extract_cache_states([CacheList(live, ArraysCache(4))])
    sub_names, sub_meta = extracted[0]["meta_state"]
    keys, values = extracted[0]["state"][0]
    assert sub_names[0] == "Inkling2WindowKVCache"
    assert tuple(sub_meta[0]) == ("0", str(window), str(live.offset), str(window))
    assert keys.shape[2] == window

    restored = RotatingKVCacheHandler().reconstruct_cache(
        {"keys": keys, "values": values}, tuple(sub_meta[0])
    )
    entry = CacheList(restored, ArraysCache(4))
    adopt(entry)
    for seed in range(5):
        a = _qkv(1 if seed % 2 else 8, seed=60 + seed, r=window)
        assert mx.array_equal(_attend(full, *a, window), _attend(entry[0], *a, window))


def test_prefix_store_reads_sequence_length_from_full_member():
    """The store slices new blocks by the full-history length; a sliding
    member in the first layer must not be mistaken for it (that saved new
    blocks' full-attention KV shifted by the restored prefix)."""
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    def layer(name, rows):
        kv = mx.zeros((1, HKV, rows, D))
        conv = (mx.zeros((1, 3, 8)),) * 4
        return {
            "class_name": "CacheList",
            "cache_type": "CacheList",
            "state": [(kv, kv), conv],
            "meta_state": ([name, "ArraysCache"], [(), ()]),
        }

    data = [layer("Inkling2WindowKVCache", 1024), layer("KVCache", 32792)]
    assert BlockAwarePrefixCache._get_cache_seq_len(None, data) == 32792
    only_windows = [layer("Inkling2WindowKVCache", 1024)]
    assert BlockAwarePrefixCache._get_cache_seq_len(None, only_windows) == 1024
