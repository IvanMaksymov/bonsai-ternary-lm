#!/usr/bin/env python3
"""
LITTLE BONSAI - train a small Bonsai-style ternary language model from scratch.

Every linear layer stores its weights as t * s, with t in {-1, 0, +1} and one FP16 scale s
per group of 128 weights (g128), exactly the format described in the Bonsai 2 paper.
With --hadamard, each layer also applies a blockwise Hadamard rotation (128 x 128 blocks)
to its weights before quantizing, and the same rotation to its inputs, so the maths is
unchanged:  (x H)(W H)^T = x H H^T W^T = x W^T.

Train (Tiny Shakespeare by default, or --data your_text.txt):
    python little_bonsai.py --out run_plain
    python little_bonsai.py --out run_hadamard --hadamard        # compare the two!
Dropout (default 0.2) fights overfitting on small texts:
    python little_bonsai.py --out run_drop --steps 5000                 # dropout 0.2
    python little_bonsai.py --out run_nodrop --dropout 0                # old behaviour
Resume an interrupted run:      python little_bonsai.py --out run_hadamard --hadamard --resume
Generate from the packed file:  python little_bonsai.py --sample run_hadamard/bonsai_ternary.pt
Generate from a checkpoint:     python little_bonsai.py --sample run_hadamard/best.pt --prompt "JULIET:"

Outputs in --out: last.pt (resume), best.pt (best validation), bonsai_ternary.pt (the
ternary weights packed 5 per byte in base 3 + FP16 scales: ~1.73 bits per weight).
"""
import argparse
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

G = 128  # group size for scales, and the Hadamard block size


# ------------------------------------------------------------------ ternary maths
def hadamard_matrix(n):
    """Sylvester construction, normalised so that H @ H.T = I (and H is symmetric)."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def quantize_ternary(w):
    """w: fp32 tensor whose last dim is a multiple of G. Returns t (rows, G) and s (rows, 1)."""
    g = w.reshape(-1, G)
    s = g.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)  # absmean scale per group
    t = (g / s).round().clamp(-1, 1)
    return t, s


class TernaryLinear(nn.Linear):
    """Linear layer with ternary g128 weights, optional blockwise Hadamard rotation.
    Training keeps full-precision 'latent' weights; the forward pass uses their ternary
    version and gradients pass straight through the rounding (straight-through estimator)."""

    def __init__(self, fan_in, fan_out, hadamard=False):
        assert fan_in % G == 0, f"fan_in ({fan_in}) must be a multiple of {G}"
        super().__init__(fan_in, fan_out, bias=False)
        self.use_h = hadamard
        self.register_buffer("H", hadamard_matrix(G), persistent=False)
        self.frozen = None  # (t, s) when loaded from a packed export

    def rotated_weight(self):
        with torch.autocast(self.weight.device.type, enabled=False):
            w = self.weight.float()
            if self.use_h:
                w = (w.reshape(-1, G) @ self.H).reshape(w.shape)  # W -> W H (per 128-block)
        return w

    @torch.no_grad()
    def ternary(self):
        if self.frozen is not None:
            return self.frozen
        return quantize_ternary(self.rotated_weight())

    def forward(self, x):
        if self.use_h:  # x -> x H (per 128-block), matching the rotated weights
            shape = x.shape
            x = (x.reshape(*shape[:-1], -1, G) @ self.H.to(x.dtype)).reshape(shape)
        if self.frozen is not None:
            t, s = self.frozen
            w = (t * s).reshape(self.weight.shape)
        else:
            w = self.rotated_weight()
            t, s = quantize_ternary(w)
            w = w + ((t * s).reshape(w.shape) - w).detach()  # straight-through estimator
        return F.linear(x, w)


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, heads, hadamard, dropout=0.0):
        super().__init__()
        self.heads, self.p = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = TernaryLinear(d, 3 * d, hadamard)
        self.proj = TernaryLinear(d, d, hadamard)
        self.fc = TernaryLinear(d, 4 * d, hadamard)
        self.out = TernaryLinear(4 * d, d, hadamard)
        self.drop = nn.Dropout(dropout)  # active only in model.train(), off for eval/sampling

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (z.view(B, T, self.heads, C // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class LittleBonsai(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx, hadamard, dropout=0.0):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d, heads, hadamard, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = TernaryLinear(d, vocab, hadamard)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


# ------------------------------------------------------------------ data
def load_text(path):
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    cache = "tinyshakespeare.txt"
    if not os.path.exists(cache):
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
               "data/tinyshakespeare/input.txt")
        print(f"downloading Tiny Shakespeare -> {cache}")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(urllib.request.urlopen(url, timeout=30).read().decode("utf-8"))
    with open(cache, encoding="utf-8") as f:
        return f.read()


def amp_dtype_for(dev):
    if dev.type == "cuda" and torch.cuda.get_device_capability(dev) < (8, 0):
        return torch.float16  # e.g. GTX 1660 Super: no native bf16
    return torch.bfloat16


@torch.no_grad()
def generate(model, chars, prompt, n, temp, dev):
    stoi = {c: i for i, c in enumerate(chars)}
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in (prompt or "\n")]], device=dev)
    for _ in range(n):
        with torch.autocast(dev.type, dtype=amp_dtype_for(dev), enabled=dev.type == "cuda"):
            logits = model(idx[:, -model.ctx:])[:, -1, :].float() / temp
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
    return "".join(chars[i] for i in idx[0].tolist())


# ------------------------------------------------------------------ packing (export)
def pack_base3(t):
    """{-1,0,+1} -> 5 trits per byte (3^5 = 243 <= 256): 1.6 bits per weight."""
    c = (t.flatten() + 1).to(torch.int32)
    c = F.pad(c, (0, (-c.numel()) % 5)).view(-1, 5)
    return (c[:, 0] + 3 * c[:, 1] + 9 * c[:, 2] + 27 * c[:, 3] + 81 * c[:, 4]).to(torch.uint8)


def unpack_base3(p, numel):
    p = p.to(torch.int32)
    digits = []
    for _ in range(5):
        digits.append(p % 3)
        p = p // 3
    return (torch.stack(digits, 1).flatten()[:numel] - 1).to(torch.float32)


def export_packed(model, config, chars, path):
    ternary, skip = {}, set()
    n_weights = n_groups = packed_bytes = 0
    for name, mod in model.named_modules():
        if isinstance(mod, TernaryLinear):
            t, s = mod.ternary()
            t, s = t.cpu(), s.cpu()
            p = pack_base3(t)
            assert torch.equal(unpack_base3(p, t.numel()), t.flatten()), "packing round-trip failed"
            ternary[name] = dict(trits=p, numel=t.numel(), scales=s.squeeze(1).half())
            skip.add(name + ".weight")
            n_weights += t.numel()
            n_groups += t.shape[0]
            packed_bytes += p.numel() + 2 * s.numel()
    other = {k: (v.half() if v.is_floating_point() else v).cpu()
             for k, v in model.state_dict().items() if k not in skip}
    torch.save(dict(config=config, chars=chars, ternary=ternary, other=other), path)

    ideal_bits = n_weights * math.log2(3) + n_groups * 16
    kb = 1024
    print(f"\n  packed export -> {path}")
    print(f"  ternary weights: {n_weights:,} in {n_groups:,} groups of {G}")
    print(f"    FP16 storage            {n_weights * 2 / kb:9.1f} KB   16.000 bits/weight")
    print(f"    ideal ternary (paper)   {ideal_bits / 8 / kb:9.1f} KB   "
          f"{ideal_bits / n_weights:6.3f} bits/weight  (log2(3) + 16/128)")
    print(f"    base-3 packed (actual)  {packed_bytes / kb:9.1f} KB   "
          f"{packed_bytes * 8 / n_weights:6.3f} bits/weight  (5 trits per byte + FP16 scales)")
    print(f"  file on disk (incl. FP16 embeddings and norms): {os.path.getsize(path) / kb:,.1f} KB")


def load_packed(path, dev):
    ck = torch.load(path, map_location="cpu")
    model = LittleBonsai(**ck["config"])
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                           for k, v in ck["other"].items()}, strict=False)
    mods = dict(model.named_modules())
    for name, e in ck["ternary"].items():
        t = unpack_base3(e["trits"], e["numel"]).view(-1, G)
        s = e["scales"].float().unsqueeze(1)
        mods[name].frozen = (t.to(dev), s.to(dev))
    return model.to(dev), ck["chars"]


def load_checkpoint_model(path, dev):
    ck = torch.load(path, map_location="cpu")
    model = LittleBonsai(**ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(dev), ck["chars"]


# ------------------------------------------------------------------ training
def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")

    text = load_text(args.data)
    ck = torch.load(last_path, map_location="cpu") if args.resume and os.path.exists(last_path) else None
    chars = ck["chars"] if ck else sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long, device=dev)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    config = ck["config"] if ck else dict(vocab=len(chars), d=args.d, layers=args.layers,
                                          heads=args.heads, ctx=args.ctx, hadamard=args.hadamard,
                                          dropout=args.dropout)
    model = LittleBonsai(**config).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    step0, best_val = 0, float("inf")
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0, best_val = ck["step"], ck["best_val"]
        print(f"resumed from {last_path} at step {step0} (best val {best_val:.3f})")

    amp = amp_dtype_for(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and amp == torch.float16))
    fwd = torch.compile(model) if args.compile else model
    ctx, batch = config["ctx"], args.batch
    offs = torch.arange(ctx + 1, device=dev)

    def get_batch(src):
        ix = torch.randint(len(src) - ctx - 1, (batch,), device=dev)
        chunk = src[ix[:, None] + offs]
        return chunk[:, :-1], chunk[:, 1:]

    def loss_on(x, y):
        with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
            logits = fwd(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))

    @torch.no_grad()
    def evaluate(n=40):
        model.eval()
        v = sum(loss_on(*get_batch(val_data)).item() for _ in range(n)) / n
        model.train()
        return v

    def lr_at(step):  # linear warm-up, then cosine decay to 10%
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def save(path, step):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                        best_val=best_val, config=config, chars=chars), path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {dev} ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}), "
          f"autocast {str(amp).split('.')[-1]}")
    print(f"model: {n_params / 1e6:.2f}M params, d={config['d']}, {config['layers']} layers, "
          f"{config['heads']} heads, ctx {ctx}, Hadamard {'ON' if config['hadamard'] else 'off'}, "
          f"dropout {config.get('dropout', 0.0)}")
    print(f"data: {len(data):,} chars, vocab {len(chars)}; {args.steps:,} steps x "
          f"{batch * ctx:,} tokens/step\n")

    model.train()
    t0, tokens = time.perf_counter(), 0
    for step in range(step0, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = loss_on(*get_batch(train_data))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tokens += batch * ctx

        last = step + 1 == args.steps
        if (step + 1) % args.eval_every == 0 or last:
            val = evaluate()
            elapsed = time.perf_counter() - t0
            flag = ""
            if val < best_val:
                best_val = val
                save(best_path, step + 1)
                flag = "  * best"
            save(last_path, step + 1)
            print(f"step {step + 1:6d}/{args.steps}  train {loss.item():.3f}  val {val:.3f}  "
                  f"lr {lr_at(step):.2e}  {tokens / elapsed / 1e3:6.1f}k tok/s  "
                  f"{elapsed / 60:5.1f} min{flag}", flush=True)
            if (step + 1) % (args.eval_every * 5) == 0 or last:
                snippet = generate(model, chars, args.prompt, 160, 0.8, dev)
                print("   | " + snippet.replace("\n", "\n   | "))
                model.train()

    print(f"\ntraining done, best val loss {best_val:.3f}")
    best_model, _ = load_checkpoint_model(best_path, dev)
    export_packed(best_model, config, chars, os.path.join(args.out, "bonsai_ternary.pt"))


def main():
    ap = argparse.ArgumentParser(description="Train a little Bonsai-style ternary LM")
    ap.add_argument("--data", default=None, help="UTF-8 text file (default: Tiny Shakespeare)")
    ap.add_argument("--out", default="little_bonsai_run")
    ap.add_argument("--hadamard", action="store_true", help="blockwise Hadamard rotation (Bonsai 2)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--d", type=int, default=384, help="width, multiple of 128")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="dropout rate (0.2 like nanoGPT; 0 = off). Fixed for a run, kept on --resume")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--compile", action="store_true", help="torch.compile (faster on Linux)")
    ap.add_argument("--sample", default=None, help="generate from best.pt or bonsai_ternary.pt")
    ap.add_argument("--prompt", default="ROMEO:\n")
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    if args.sample:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_packed = "ternary" in torch.load(args.sample, map_location="cpu")
        model, chars = (load_packed if is_packed else load_checkpoint_model)(args.sample, dev)
        print(f"[{'packed ternary file' if is_packed else 'training checkpoint'}: {args.sample}]\n")
        print(generate(model, chars, args.prompt, args.tokens, args.temp, dev))
        return
    if args.d % G or args.d % args.heads:
        ap.error(f"--d must be a multiple of {G} and divisible by --heads")
    train(args)


if __name__ == "__main__":
    main()
