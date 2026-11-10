#!/usr/bin/env python3
"""
Prepare a tokenised dataset for bonsai_lm.py (GPT-2 BPE tokenizer, uint16 .bin files).

Default: stream FineWeb-Edu (high-quality educational web text) from Hugging Face and stop
once --tokens training tokens are collected. Nothing else is downloaded.
    pip3 install --user datasets tiktoken numpy
    python3 prepare_data.py --tokens 3e9 --out data/fineweb_edu        # ~6 GB on disk

Your own text instead (a .txt file, or a folder of .txt / .md / .tex files, searched recursively):
    python3 prepare_data.py --local ~/my_corpus --out data/my_corpus

Output in --out: train.bin, val.bin (uint16 token ids, documents separated by <|endoftext|>),
meta.json (tokenizer, vocab size, token counts).
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import tiktoken

ENC = None


def _init_worker():
    global ENC
    ENC = tiktoken.get_encoding("gpt2")


def encode(text):
    ids = [ENC.eot_token] + ENC.encode_ordinary(text)
    return np.asarray(ids, dtype=np.uint16)


def local_docs(path):
    exts = (".txt", ".md", ".tex", ".rst")
    files = [path] if os.path.isfile(path) else sorted(
        os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.lower().endswith(exts))
    if not files:
        raise SystemExit(f"no text files found under {path}")
    print(f"found {len(files):,} files")
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # split very long files into paragraphs-sized documents so they shuffle into val/train
        for chunk in text.split("\n\n\n"):
            if chunk.strip():
                yield chunk


def hf_docs(dataset, name, split, field):
    from datasets import load_dataset
    ds = load_dataset(dataset, name=name, split=split, streaming=True)
    for row in ds:
        text = row.get(field)
        if text:
            yield text


def main():
    ap = argparse.ArgumentParser(description="Tokenise text for bonsai_lm.py")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT", help="dataset config (FineWeb-Edu: sample-10BT)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="text")
    ap.add_argument("--local", default=None, help="local .txt file or folder instead of Hugging Face")
    ap.add_argument("--tokens", type=float, default=3e9, help="training tokens to collect")
    ap.add_argument("--val-tokens", type=float, default=10e6)
    ap.add_argument("--out", default="data/fineweb_edu")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = {k: os.path.join(args.out, f"{k}.bin") for k in ("train", "val")}
    for p in paths.values():
        if os.path.exists(p):
            raise SystemExit(f"{p} already exists - delete it or choose another --out")

    docs = local_docs(args.local) if args.local else hf_docs(args.dataset, args.name, args.split,
                                                              args.field)
    targets = {"val": int(args.val_tokens), "train": int(args.tokens)}
    counts = {"val": 0, "train": 0}
    files = {k: open(p, "wb") for k, p in paths.items()}
    t0, next_report, n_docs = time.time(), 50_000_000, 0
    split = "val"  # fill the validation set first, then training

    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for ids in pool.imap(encode, docs, chunksize=64):
            n_docs += 1
            files[split].write(ids.tobytes())
            counts[split] += len(ids)
            if split == "val" and counts["val"] >= targets["val"]:
                split = "train"
            total = counts["val"] + counts["train"]
            if total >= next_report:
                rate = total / (time.time() - t0) / 1e6
                print(f"  {total / 1e6:8,.0f}M tokens  ({rate:.2f}M tok/s, {n_docs:,} docs)", flush=True)
                next_report += 50_000_000
            if counts["train"] >= targets["train"]:
                pool.terminate()
                break

    for f in files.values():
        f.close()
    meta = dict(tokenizer="gpt2", vocab_size=50257, eot=50256, dtype="uint16",
                train_tokens=counts["train"], val_tokens=counts["val"], docs=n_docs,
                source=args.local or f"{args.dataset}/{args.name}")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: train {counts['train']:,} tokens, "
          f"val {counts['val']:,} tokens -> {args.out}")
    if counts["train"] < targets["train"]:
        print("note: the source ran out before reaching --tokens")


if __name__ == "__main__":
    main()
