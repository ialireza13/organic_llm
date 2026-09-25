"""Download FineWeb sample-10BT parquet shards and tokenize to uint16 .bin files (nanoGPT style).

Output (in --out):
  val_000.bin            ~10M tokens (first docs of the first parquet file)
  train_XXX.bin          100M-token shards, ~1.5B tokens total
Each document is prefixed by the GPT-2 <|endoftext|> token (50256).
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

ENC = None
EOT = 50256


def _init():
    global ENC
    ENC = tiktoken.get_encoding("gpt2")


def _tok(texts):
    out = ENC.encode_ordinary_batch(texts, num_threads=1)
    arrs = []
    for t in out:
        a = np.empty(len(t) + 1, dtype=np.uint16)
        a[0] = EOT
        a[1:] = t
        arrs.append(a)
    return np.concatenate(arrs) if arrs else np.empty(0, dtype=np.uint16)


def text_batches(path, bs=512):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        col = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        for i in range(0, len(col), bs):
            yield col[i:i + bs]


class ShardWriter:
    def __init__(self, out, prefix, shard_tokens):
        self.out, self.prefix, self.shard_tokens = out, prefix, shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.n = 0
        self.idx = 0
        self.total = 0

    def add(self, arr):
        while len(arr):
            k = min(len(arr), self.shard_tokens - self.n)
            self.buf[self.n:self.n + k] = arr[:k]
            self.n += k
            arr = arr[k:]
            if self.n == self.shard_tokens:
                self.flush()

    def flush(self):
        if self.n == 0:
            return
        fn = os.path.join(self.out, f"{self.prefix}_{self.idx:03d}.bin")
        self.buf[:self.n].tofile(fn)
        print(f"wrote {fn} ({self.n} tokens)", flush=True)
        self.total += self.n
        self.idx += 1
        self.n = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb")
    ap.add_argument("--repo", default="HuggingFaceFW/fineweb")
    ap.add_argument("--subdir", default="sample/10BT")
    ap.add_argument("--nfiles", type=int, default=3)
    ap.add_argument("--train_tokens", type=int, default=1_500_000_000)
    ap.add_argument("--val_tokens", type=int, default=10_000_000)
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    val = ShardWriter(args.out, "val", args.val_tokens)
    train = ShardWriter(args.out, "train", args.shard_tokens)
    val_done = False
    train_count = 0
    with Pool(args.workers, initializer=_init) as pool:
        for fi in range(args.nfiles):
            fname = f"{args.subdir}/{fi:03d}_00000.parquet"
            path = hf_hub_download(args.repo, fname, repo_type="dataset")
            print(f"downloaded {fname} in {time.time() - t0:.0f}s", flush=True)
            for arr in pool.imap(_tok, text_batches(path), chunksize=4):
                if not val_done:
                    need = args.val_tokens - val.n
                    val.add(arr[:need])  # val is filled exactly; remainder of batch goes to train
                    arr = arr[need:]
                    if val.idx == 1:
                        val_done = True
                if len(arr) and val_done:
                    k = min(len(arr), args.train_tokens - train_count)
                    train.add(arr[:k])
                    train_count += k
                if train_count >= args.train_tokens:
                    break
            if train_count >= args.train_tokens:
                break
    train.flush()
    print(f"done: train {train.total} tokens, val {val.total} tokens, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
