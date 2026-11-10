#!/usr/bin/env python3
"""
BONSAI LM - train a Bonsai-style ternary language model seriously, on real text.

Architecture: modern decoder-only transformer (RMSNorm, rotary position embeddings, SwiGLU MLP,
FlashAttention via PyTorch SDPA). Every linear layer inside the blocks is ternary: t * s with
t in {-1, 0, +1} and one scale per group of 128 weights (g128), optional blockwise Hadamard
rotation. As in Bonsai 2, the embedding (tied with the output head) and the norms stay in
high precision. --precision fp trains the same model in normal precision as a baseline.

1) Data:     python3 prepare_data.py --tokens 3e9 --out data/fineweb_edu
2) Smoke:    python3 bonsai_lm.py --data data/fineweb_edu --config tiny --tokens 20e6 --out runs/smoke
3) Serious:  python3 bonsai_lm.py --data data/fineweb_edu --config base --tokens 2.5e9 --out runs/base_t
             (about 1.5-2 days on an RTX 3060; Ctrl+C saves a checkpoint, --resume continues)
   Baseline: python3 bonsai_lm.py --data data/fineweb_edu --config base --tokens 2.5e9 \
                 --precision fp --out runs/base_fp
Sample:      python3 bonsai_lm.py --sample runs/base_t/bonsai_ternary.pt --prompt "The universe"

Run long jobs inside tmux so closing the terminal doesn't stop them:  tmux new -s bonsai
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

G = 128
REAL_VOCAB = 50257  # GPT-2 tokens; the embedding is padded to 50304 for speed  # ternary group size and Hadamard block size

CONFIGS = {  # non-embedding parameters in brackets (embedding adds ~vocab * d)
    "tiny":   dict(d=384,  layers=6,  heads=6,  ctx=512),    # ~11M
    "small":  dict(d=512,  layers=8,  heads=8,  ctx=1024),   # ~25M
    "base":   dict(d=768,  layers=12, heads=12, ctx=1024),   # ~85M  (GPT-2 small size, 124M total)
    "medium": dict(d=1024, layers=24, heads=16, ctx=1024),   # ~300M (use --grad-ckpt on 12 GB)
}


# ------------------------------------------------------------------ ternary layer
def hadamard_matrix(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def quantize_ternary(w):
    g = w.reshape(-1, G)
    s = g.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    return (g / s).round().clamp(-1, 1), s


class TernaryLinear(nn.Linear):
    def __init__(self, fan_in, fan_out, ternary=True, hadamard=False):
        assert fan_in % G == 0, f"fan_in {fan_in} must be a multiple of {G}"
        super().__init__(fan_in, fan_out, bias=False)
        self.ternary_on, self.use_h = ternary, ternary and hadamard
        self.register_buffer("H", hadamard_matrix(G), persistent=False)
        self.frozen = None  # (t, s) when loaded from a packed export

    def rotated_weight(self):
        with torch.autocast(self.weight.device.type, enabled=False):  # quantize in full fp32
            w = self.weight.float()
            if self.use_h:
                w = (w.reshape(-1, G) @ self.H).reshape(w.shape)
        return w

    @torch.no_grad()
    def ternary(self):
        if self.frozen is not None:
            return self.frozen
        return quantize_ternary(self.rotated_weight())

    def forward(self, x):
        if not self.ternary_on:
            return F.linear(x, self.weight)
        if self.use_h:
            shp = x.shape
            x = (x.reshape(*shp[:-1], -1, G) @ self.H.to(x.dtype)).reshape(shp)
        if self.frozen is not None:
            t, s = self.frozen
            w = (t * s).reshape(self.weight.shape)
        else:
            w = self.rotated_weight()
            t, s = quantize_ternary(w)
            w = w + ((t * s).reshape(w.shape) - w).detach()  # straight-through estimator
        return F.linear(x, w.to(x.dtype))


# ------------------------------------------------------------------ model
def rope_cache(ctx, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(ctx).float(), inv)
    return torch.cos(f), torch.sin(f)  # (ctx, head_dim/2)


def apply_rope(x, cos, sin):  # x: (B, H, T, hd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[2]].to(x.dtype), sin[: x.shape[2]].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, ternary, hadamard, dropout):
        super().__init__()
        self.heads, self.p = heads, dropout
        hidden = int(math.ceil(8 * d / 3 / G) * G)  # SwiGLU width, multiple of 128
        L = lambda i, o: TernaryLinear(i, o, ternary, hadamard)
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.proj = L(d, 3 * d), L(d, d)
        self.gate, self.up, self.down = L(d, hidden), L(d, hidden), L(hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split(C, dim=2)
        q, k, v = (z.view(B, T, self.heads, C // self.heads).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        h = self.n2(x)
        return x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))


class BonsaiLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx, ternary=True, hadamard=False, dropout=0.0,
                 grad_ckpt=False):
        super().__init__()
        assert d % heads == 0 and (d // heads) % 2 == 0
        self.ctx, self.grad_ckpt = ctx, grad_ckpt
        self.tok = nn.Embedding(vocab, d)  # high precision, tied with the output head
        self.blocks = nn.ModuleList(Block(d, heads, ternary, hadamard, dropout) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        cos, sin = rope_cache(ctx, d // heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx):
        x = self.tok(idx)
        for b in self.blocks:
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = b(x, self.cos, self.sin)
        return F.linear(self.norm(x), self.tok.weight)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total, total - model.tok.weight.numel()


# ------------------------------------------------------------------ data
class TokenData:
    def __init__(self, folder, ctx, micro_batch, dev):
        with open(os.path.join(folder, "meta.json")) as f:
            self.meta = json.load(f)
        self.train = np.memmap(os.path.join(folder, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(folder, "val.bin"), dtype=np.uint16, mode="r")
        self.ctx, self.mb, self.dev = ctx, micro_batch, dev

    def batch(self, split, gen):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.ctx - 1, (self.mb,), generator=gen)
        x = torch.from_numpy(np.stack([data[i:i + self.ctx] for i in ix.tolist()]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + self.ctx] for i in ix.tolist()]).astype(np.int64))
        if self.dev.type == "cuda":
            return x.pin_memory().to(self.dev, non_blocking=True), y.pin_memory().to(self.dev, non_blocking=True)
        return x.to(self.dev), y.to(self.dev)


# ------------------------------------------------------------------ packing / loading
def pack_base3(t):
    c = (t.flatten() + 1).to(torch.int32)
    c = F.pad(c, (0, (-c.numel()) % 5)).view(-1, 5)
    return (c[:, 0] + 3 * c[:, 1] + 9 * c[:, 2] + 27 * c[:, 3] + 81 * c[:, 4]).to(torch.uint8)


def unpack_base3(p, numel):
    p, digits = p.to(torch.int32), []
    for _ in range(5):
        digits.append(p % 3)
        p = p // 3
    return (torch.stack(digits, 1).flatten()[:numel] - 1).to(torch.float32)


def export_packed(model, config, path):
    ternary, skip, n_w, n_bytes = {}, set(), 0, 0
    for name, m in model.named_modules():
        if isinstance(m, TernaryLinear) and m.ternary_on:
            t, s = (z.cpu() for z in m.ternary())
            p = pack_base3(t)
            ternary[name] = dict(trits=p, numel=t.numel(), scales=s.squeeze(1).half())
            skip.add(name + ".weight")
            n_w += t.numel()
            n_bytes += p.numel() + 2 * s.numel()
    other = {k: (v.half() if v.is_floating_point() else v).cpu()
             for k, v in model.state_dict().items() if k not in skip}
    torch.save(dict(config=config, ternary=ternary, other=other), path)
    if n_w:
        print(f"  packed export -> {path}: {n_w:,} ternary weights at {n_bytes * 8 / n_w:.3f} "
              f"bits/weight; file {os.path.getsize(path) / 2**20:,.1f} MB "
              f"(FP16 would be {n_w * 2 / 2**20:,.1f} MB for the ternary part)")


def load_for_sampling(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.pop("grad_ckpt", None)
    model = BonsaiLM(**cfg)
    if "ternary" in ck:
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in ck["other"].items()}, strict=False)
        mods = dict(model.named_modules())
        for name, e in ck["ternary"].items():
            t = unpack_base3(e["trits"], e["numel"]).view(-1, G)
            mods[name].frozen = (t.to(dev), e["scales"].float().unsqueeze(1).to(dev))
    else:
        model.load_state_dict(ck["model"])
    return model.to(dev).eval()


@torch.no_grad()
def generate(model, enc, prompt, n, temp, top_k, dev):
    model.eval()
    idx = torch.tensor([enc.encode_ordinary(prompt) or [enc.eot_token]], device=dev)
    for _ in range(n):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(idx[:, -model.ctx:])[:, -1, :].float() / temp
        logits[:, REAL_VOCAB:] = -float("inf")  # never sample the padding ids
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        if nxt.item() == enc.eot_token:
            break
        idx = torch.cat([idx, nxt], dim=1)
    return enc.decode(idx[0].tolist())


# ------------------------------------------------------------------ training
def train(args):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    last_path, best_path = (os.path.join(args.out, n) for n in ("last.pt", "best.pt"))
    ck = torch.load(last_path, map_location="cpu", weights_only=False) \
        if args.resume and os.path.exists(last_path) else None

    if ck:
        config, a = ck["config"], ck["args"]
        for k in ("tokens", "batch_tokens", "lr", "warmup", "wd", "precision"):
            setattr(args, k, a[k])  # keep the schedule identical when resuming
    else:
        base = dict(CONFIGS[args.config])
        for k in ("d", "layers", "heads", "ctx"):
            if getattr(args, k) is not None:
                base[k] = getattr(args, k)
        config = dict(vocab=50304, **base, ternary=args.precision == "ternary",
                      hadamard=args.hadamard, dropout=args.dropout, grad_ckpt=args.grad_ckpt)
    if args.lr is None:
        args.lr = 1.5e-3 if config["ternary"] else 6e-4

    data = TokenData(args.data, config["ctx"], args.micro_batch, dev)
    tokens_per_micro = args.micro_batch * config["ctx"]
    accum = max(1, round(args.batch_tokens / tokens_per_micro))
    tokens_per_step = accum * tokens_per_micro
    steps = int(args.tokens // tokens_per_step)
    if steps < 1:
        raise SystemExit("--tokens is smaller than one optimiser step; increase --tokens")
    if args.tokens > len(data.train):
        print(f"warning: --tokens ({args.tokens:.3g}) exceeds the dataset ({len(data.train):.3g}); "
              f"data will repeat")
    warmup = args.warmup if args.warmup is not None else max(1, min(max(100, steps // 50), steps // 5))

    model = BonsaiLM(**config).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=dev.type == "cuda")
    step, best_val, gen = 0, float("inf"), torch.Generator().manual_seed(1234)
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best_val = ck["step"], ck["best_val"]
        gen.set_state(ck["gen"])
    fwd = torch.compile(model) if args.compile else model

    def lr_at(s):
        if s < warmup:
            return args.lr * (s + 1) / warmup
        p = min(1.0, (s - warmup) / max(1, steps - warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def loss_on(x, y):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = fwd(x)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))

    @torch.no_grad()
    def evaluate(n):
        model.eval()
        vgen = torch.Generator().manual_seed(42)  # same val batches every time
        v = sum(loss_on(*data.batch("val", vgen)).item() for _ in range(n)) / n
        model.train()
        return v

    def save(path):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best_val=best_val,
                        config=config, args=vars(args), gen=gen.get_state()), path)

    total, non_emb = count_params(model)
    print(f"device: {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}")
    print(f"model: {total / 1e6:.1f}M params ({non_emb / 1e6:.1f}M non-embedding), d={config['d']}, "
          f"{config['layers']} layers, {config['heads']} heads, ctx {config['ctx']}, "
          f"{'TERNARY g128' if config['ternary'] else 'full precision'}"
          f"{' + Hadamard' if config['hadamard'] else ''}, dropout {config['dropout']}")
    print(f"training: {args.tokens / 1e9:.2f}B tokens = {steps:,} steps x {tokens_per_step:,} tokens "
          f"(micro-batch {args.micro_batch} x accum {accum}), lr {args.lr:g}, warmup {warmup}")
    print(f"data: {len(data.train) / 1e9:.2f}B train tokens from {data.meta.get('source')}\n")

    log = open(os.path.join(args.out, "log.csv"), "a", newline="")
    logw = csv.writer(log)
    if not ck:
        logw.writerow(["step", "tokens", "train_loss", "val_loss", "lr", "tok_per_s", "hours"])
    model.train()
    t0, tok_seen, train_loss = time.time(), 0, float("nan")
    try:
        while step < steps:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss_acc = torch.zeros((), device=dev)
            for _ in range(accum):
                loss = loss_on(*data.batch("train", gen)) / accum
                loss.backward()
                loss_acc += loss.detach()  # no GPU sync per micro-batch
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tok_seen += tokens_per_step
            if step % args.log_every == 0 or step % args.eval_every == 0 or step == steps:
                train_loss = loss_acc.item()
            if step % args.log_every == 0:
                el = time.time() - t0
                tps = tok_seen / el
                eta = (steps - step) * tokens_per_step / tps / 3600
                print(f"step {step:7,}/{steps:,}  loss {train_loss:.3f}  lr {lr_at(step):.2e}  "
                      f"{tps / 1e3:6.1f}k tok/s  ETA {eta:5.1f} h", flush=True)
            if step % args.eval_every == 0 or step == steps:
                val = evaluate(args.eval_batches)
                flag = ""
                if val < best_val:
                    best_val, flag = val, "  * best"
                    save(best_path)
                save(last_path)
                el = time.time() - t0
                logw.writerow([step, step * tokens_per_step, f"{train_loss:.4f}", f"{val:.4f}",
                               f"{lr_at(step):.3e}", f"{tok_seen / el:.0f}", f"{el / 3600:.3f}"])
                log.flush()
                print(f"  == eval step {step:,}: val loss {val:.3f} (ppl {math.exp(val):.1f}){flag}")
                if step % (args.eval_every * 4) == 0 or step == steps:
                    print("   | " + generate(model, enc, args.prompt, 80, 0.8, 50, dev).replace("\n", "\n   | "))
                    model.train()
    except KeyboardInterrupt:
        print("\ninterrupted - saving last.pt (continue with --resume)")
        save(last_path)
        log.close()
        return
    log.close()
    print(f"\ntraining done: best val loss {best_val:.3f} (ppl {math.exp(best_val):.1f})")
    best = load_for_sampling(best_path, dev)
    if config["ternary"]:
        export_packed(best, config, os.path.join(args.out, "bonsai_ternary.pt"))


def main():
    ap = argparse.ArgumentParser(description="Train a Bonsai-style ternary LM on tokenised text")
    ap.add_argument("--data", default="data/fineweb_edu", help="folder from prepare_data.py")
    ap.add_argument("--out", default="runs/bonsai")
    ap.add_argument("--config", choices=CONFIGS, default="small")
    ap.add_argument("--d", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--precision", choices=["ternary", "fp"], default="ternary")
    ap.add_argument("--hadamard", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0, help="0 is right for large single-pass data")
    ap.add_argument("--tokens", type=float, default=500e6, help="total training tokens")
    ap.add_argument("--batch-tokens", type=int, default=262144, help="tokens per optimiser step")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="sequences per forward pass; lower it (4, 2) if you run out of VRAM")
    ap.add_argument("--lr", type=float, default=None, help="default 1.5e-3 ternary, 6e-4 fp")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true", help="save memory for big models")
    ap.add_argument("--compile", action="store_true", help="torch.compile: ~1.3-2x faster")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--sample", default=None, help="best.pt / last.pt / bonsai_ternary.pt to sample")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    if args.sample:
        import tiktoken
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_for_sampling(args.sample, dev)
        print(generate(model, tiktoken.get_encoding("gpt2"), args.prompt, args.max_new,
                       args.temp, args.top_k, dev))
        return
    train(args)


if __name__ == "__main__":
    main()
