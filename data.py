"""Deterministic data access over uint16 token shards.

Train: the token stream is cut into non-overlapping (seq_len+1)-token windows (stride seq_len);
a permutation seeded by `data_seed` fixes the order, and step s uses windows perm[s*B:(s+1)*B].
So the batch at a given step depends only on (data_seed, step, B): runs and continuations are
reproducible and share data order.
Val: val_000.bin = [0, 8M) evaluation tokens, [8M, 10M) held-out metric tokens.
"""
import glob
import os

import numpy as np
import torch

VAL_EVAL_TOKENS = 8_000_000


class TrainData:
    def __init__(self, data_dir, seq_len, batch_size, data_seed=0, device="cuda"):
        files = sorted(glob.glob(os.path.join(data_dir, "train_*.bin")))
        assert files, f"no train shards in {data_dir}"
        self.mms = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.T, self.B, self.device = seq_len, batch_size, device
        self.win_per_shard = [(len(m) - 1) // seq_len for m in self.mms]
        self.cum = np.cumsum([0] + self.win_per_shard)
        rng = np.random.default_rng(data_seed)
        self.perm = rng.permutation(self.cum[-1])

    def n_windows(self):
        return int(self.cum[-1])

    def get(self, step, micro=0, n_micro=1):
        B = self.B
        mb = B // n_micro
        ids = self.perm[(step * B + micro * mb) % len(self.perm):][:mb]
        if len(ids) < mb:  # wrap-around (only if the budget exceeds the dataset)
            ids = np.concatenate([ids, self.perm[: mb - len(ids)]])
        buf = np.empty((mb, self.T + 1), dtype=np.int64)
        for i, w in enumerate(ids):
            s = int(np.searchsorted(self.cum, w, side="right") - 1)
            off = int(w - self.cum[s]) * self.T
            buf[i] = self.mms[s][off: off + self.T + 1]
        t = torch.from_numpy(buf)
        if self.device == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


class ValData:
    def __init__(self, data_dir, seq_len, device="cuda"):
        self.mm = np.memmap(os.path.join(data_dir, "val_000.bin"), dtype=np.uint16, mode="r")
        self.T, self.device = seq_len, device

    def _windows(self, start, end, n=None):
        nw = (end - start - 1) // self.T
        if n is not None:
            nw = min(nw, n)
        return [(start + i * self.T) for i in range(nw)]

    def eval_batches(self, n_tokens=VAL_EVAL_TOKENS, bs=64):
        starts = self._windows(0, VAL_EVAL_TOKENS, n_tokens // self.T)
        for i in range(0, len(starts), bs):
            yield self._make(starts[i:i + bs])

    def metric_batches(self, n_batches, bs, offset=0):
        """Held-out batches from [8M, 10M); `offset` rotates through the pool."""
        starts = self._windows(VAL_EVAL_TOKENS, len(self.mm))
        out = []
        for j in range(n_batches):
            k = (offset + j) * bs
            sel = [starts[(k + i) % len(starts)] for i in range(bs)]
            out.append(self._make(sel))
        return out

    def _make(self, starts):
        buf = np.stack([self.mm[s: s + self.T + 1].astype(np.int64) for s in starts])
        t = torch.from_numpy(buf).to(self.device)
        return t[:, :-1], t[:, 1:]
