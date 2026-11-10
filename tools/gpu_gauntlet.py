#!/usr/bin/env python3
"""
GPU ARRIVAL GAUNTLET  -  a welcome test for a (used) RTX 3060 12GB

Six stages:
  0. Identity   - is it really a 12 GB Ampere GA106 (sm_86, 28 SMs)? (catches fakes/rebadges)
  1. VRAM       - fills every free byte with a known pattern, waits, verifies (catches bad memory)
  2. Compute    - matmul TFLOPS in FP32 / TF32 / FP16 / BF16
  3. Bandwidth  - device memory copy speed (what drives LLM token speed)
  4. Ternary LM - trains a tiny BitNet/Bonsai-style ternary transformer {-1,0,+1} with g128
                  scales on Tiny Shakespeare, then makes it write, and reports its bits/weight
  5. Soak       - sustained load while logging temperature, power, clocks (catches tired coolers)

Before/after mode:
  Today, on the GTX 1660 Super:   python gpu_gauntlet.py --out gtx1660s.json
  After the swap, on the 3060:    python gpu_gauntlet.py --out rtx3060.json --compare gtx1660s.json

Options: --quick (about 2 min), --train-seconds N, --soak-seconds N, --skip-vram,
         --cpu-smoke (tiny CPU run of stage 4 only, to check your Python setup)
Requires: PyTorch with CUDA (2.3 or newer), nvidia-smi on PATH (for stage 5 sensors).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

GB = 1024 ** 3


def hdr(title):
    print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}", flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------- stage 0
def flash_attention_works(dev):
    """FlashAttention kernels in PyTorch need Ampere (sm_80) or newer."""
    import warnings
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.randn(1, 8, 256, 64, device=dev, dtype=torch.float16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # older GPUs print a page of "kernel not used" notes
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
        sync()
        return True
    except Exception:
        return False


def stage_identity(dev, m):
    hdr("Stage 0 - Identity check")
    p = torch.cuda.get_device_properties(dev)
    cc = (p.major, p.minor)
    m["gpu_name"] = p.name
    m["compute_capability"] = f"{p.major}.{p.minor}"
    m["sm_count"] = p.multi_processor_count
    m["vram_total_gb"] = round(p.total_memory / GB, 2)
    m["native_bf16"] = cc >= (8, 0)
    m["flash_attention"] = flash_attention_works(dev)
    for k in ("gpu_name", "compute_capability", "sm_count", "vram_total_gb",
              "native_bf16", "flash_attention"):
        print(f"  {k:22s} {m[k]}")
    print(f"  {'pytorch / cuda':22s} {torch.__version__} / {torch.version.cuda}")
    genuine = cc == (8, 6) and p.multi_processor_count == 28 and p.total_memory > 11 * GB
    m["looks_like_rtx3060_12gb"] = genuine
    print("  ->", "PASS: genuine RTX 3060 12GB profile (GA106, sm_86, 28 SMs, 12 GB)"
          if genuine else "INFO: not an RTX 3060 12GB profile (fine if this is your baseline card)")


# ---------------------------------------------------------------- stage 1
def stage_vram(dev, m, chunk_mb=256, hold_seconds=5):
    hdr("Stage 1 - VRAM fill & integrity")
    torch.cuda.empty_cache()
    n = chunk_mb * 1024 * 1024 // 4  # int32 elements per chunk
    # Stop at the real free VRAM: on Windows the driver may otherwise spill into system RAM
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    max_chunks = free_bytes // (chunk_mb * 1024 * 1024)
    chunks, seeds = [], []
    t0 = time.perf_counter()
    while len(chunks) < max_chunks:
        try:
            i = len(chunks)
            seed = (i * 1_000_003) % (2 ** 30)
            t = torch.arange(n, device=dev, dtype=torch.int32).add_(seed)
            chunks.append(t)
            seeds.append(seed)
        except torch.cuda.OutOfMemoryError:
            break
    sync()
    if chunks:  # free one chunk so the checksums have working room
        chunks.pop()
        seeds.pop()
    filled_gb = len(chunks) * chunk_mb / 1024
    print(f"  filled {filled_gb:.2f} GB in {time.perf_counter() - t0:.1f}s, "
          f"holding for {hold_seconds}s ...", flush=True)
    time.sleep(hold_seconds)
    bad = 0
    for t, seed in zip(chunks, seeds):
        expected = n * (n - 1) // 2 + n * seed
        # sum in 1M-element slices so the int64 temporaries stay tiny (~8 MB)
        total = sum(s.sum(dtype=torch.int64) for s in t.split(1 << 20))
        if int(total.item()) != expected:
            bad += 1
    del chunks
    torch.cuda.empty_cache()
    m["vram_usable_gb"] = round(filled_gb, 2)
    m["vram_bad_chunks"] = bad
    print(f"  usable VRAM ~{filled_gb:.2f} GB, corrupted {chunk_mb} MB chunks: {bad}")
    print("  ->", "PASS: memory holds its data" if bad == 0 else "FAIL: memory errors found!")


# ---------------------------------------------------------------- stage 2
def matmul_tflops(dev, dtype, n=8192, iters=20, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    c = torch.empty(n, n, device=dev, dtype=dtype)
    for _ in range(3):
        torch.matmul(a, b, out=c)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b, out=c)
    sync()
    dt = time.perf_counter() - t0
    torch.backends.cuda.matmul.allow_tf32 = False
    del a, b, c
    torch.cuda.empty_cache()
    return 2 * n ** 3 * iters / dt / 1e12


def stage_compute(dev, m, quick):
    hdr("Stage 2 - Matmul throughput (TFLOPS)")
    iters = 8 if quick else 20
    for label, dtype, tf32 in (("fp32", torch.float32, False), ("tf32", torch.float32, True),
                               ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)):
        try:
            v = matmul_tflops(dev, dtype, iters=iters, tf32=tf32)
        except Exception as e:  # e.g. bf16 not supported on very old cards
            print(f"  {label:5s} failed: {e}")
            continue
        m[f"tflops_{label}"] = round(v, 2)
        print(f"  {label:5s} {v:6.2f} TFLOPS", flush=True)
    print("  (RTX 3060 typically: fp32 ~10-12, tf32 ~12, fp16/bf16 ~18-24)")


# ---------------------------------------------------------------- stage 3
def stage_bandwidth(dev, m):
    hdr("Stage 3 - Memory bandwidth")
    n = (1 * GB) // 4
    x = torch.empty(n, device=dev, dtype=torch.float32).normal_()
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    sync()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    sync()
    gbs = 2 * x.numel() * 4 * iters / (time.perf_counter() - t0) / 1e9
    del x, y
    torch.cuda.empty_cache()
    m["bandwidth_gbs"] = round(gbs, 1)
    print(f"  device copy: {gbs:.0f} GB/s   (RTX 3060 spec 360 GB/s, expect ~290-330)")


# ---------------------------------------------------------------- stage 4
class TernaryLinear(nn.Linear):
    """BitNet b1.58 / Bonsai-style layer: weights become s_g * t, t in {-1,0,+1},
    one scale per group of 128 weights. Full-precision 'latent' weights are kept for
    training and gradients pass straight through the rounding (STE)."""
    group = 128

    def __init__(self, fan_in, fan_out, bias=False):
        assert fan_in % self.group == 0, "fan_in must be a multiple of 128"
        super().__init__(fan_in, fan_out, bias=bias)

    def ternary(self):
        g = self.weight.reshape(-1, self.group)
        s = g.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        t = (g / s).round().clamp(-1, 1)
        return t, s

    def forward(self, x):
        t, s = self.ternary()
        wq = (t * s).reshape_as(self.weight)
        w = self.weight + (wq - self.weight).detach()  # straight-through estimator
        return F.linear(x, w, self.bias)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = TernaryLinear(d, 3 * d), TernaryLinear(d, d)
        self.fc, self.out = TernaryLinear(d, 4 * d), TernaryLinear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (z.view(B, T, self.heads, C // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class TinyTernaryGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = TernaryLinear(d, vocab)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def get_corpus():
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
           "data/tinyshakespeare/input.txt")
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyshakespeare.txt")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return f.read(), "Tiny Shakespeare (cached)"
    try:
        txt = urllib.request.urlopen(url, timeout=15).read().decode("utf-8")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(txt)
        return txt, "Tiny Shakespeare"
    except Exception:
        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            return f.read() * 20, "this script's own source code (offline fallback)"


@torch.no_grad()
def ternary_stats(model):
    neg = zero = pos = groups = 0
    for mod in model.modules():
        if isinstance(mod, TernaryLinear):
            t, _ = mod.ternary()
            neg += int((t == -1).sum())
            zero += int((t == 0).sum())
            pos += int((t == 1).sum())
            groups += t.shape[0]
    total = neg + zero + pos
    bits = (total * math.log2(3) + groups * 16) / total
    return dict(neg=neg / total, zero=zero / total, pos=pos / total, total=total,
                bits_per_weight=bits, fp16_kb=total * 2 / 1024, packed_kb=total * bits / 8 / 1024)


def stage_ternary(dev, m, seconds, cfg):
    hdr("Stage 4 - Train a tiny ternary LLM (Bonsai-style {-1,0,+1}, g128)")
    text, src = get_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=dev)
    split = int(0.95 * len(data))
    train, val = data[:split], data[split:]
    ctx, batch = cfg["ctx"], cfg["batch"]

    native_bf16 = dev.type == "cpu" or torch.cuda.get_device_capability(dev) >= (8, 0)
    amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and amp_dtype == torch.float16))
    model = TinyTernaryGPT(len(chars), cfg["d"], cfg["layers"], cfg["heads"], ctx).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.0)
    offs = torch.arange(ctx + 1, device=dev)

    def batch_from(src_data):
        ix = torch.randint(len(src_data) - ctx - 1, (batch,), device=dev)
        chunk = src_data[ix[:, None] + offs]
        return chunk[:, :-1], chunk[:, 1:]

    def loss_of(x, y):
        with torch.autocast(dev.type, dtype=amp_dtype):
            logits = model(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))

    nparams = sum(p.numel() for p in model.parameters())
    print(f"  corpus: {src}, {len(text):,} chars, vocab {len(chars)}")
    print(f"  model: {nparams / 1e6:.1f}M params, d={cfg['d']}, {cfg['layers']} layers, "
          f"ctx {ctx}, batch {batch}, autocast {str(amp_dtype).split('.')[-1]}")
    print(f"  training for {seconds}s ...", flush=True)

    model.train()
    step, tokens = 0, 0
    t_start = last_log = time.perf_counter()
    loss = torch.tensor(float("nan"))
    while time.perf_counter() - t_start < seconds:
        lr = 1.5e-3 * min(1.0, (step + 1) / 100)  # short warm-up
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch_from(train)
        loss = loss_of(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens += batch * ctx
        now = time.perf_counter()
        if now - last_log > 10:
            sync()
            print(f"  step {step:5d}  loss {loss.item():.3f}  "
                  f"{tokens / (now - t_start) / 1e3:7.1f}k tok/s", flush=True)
            last_log = now
    sync()
    elapsed = time.perf_counter() - t_start

    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_of(*batch_from(val)).item() for _ in range(10)) / 10
        idx = torch.tensor([[stoi.get(c, 0) for c in "ROMEO:\n"]], device=dev)
        torch.manual_seed(3060)
        for _ in range(400):
            with torch.autocast(dev.type, dtype=amp_dtype):
                logits = model(idx[:, -ctx:])[:, -1, :].float() / 0.8
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
    sample = "".join(chars[i] for i in idx[0].tolist())

    st = ternary_stats(model)
    m["train_tokens_per_s"] = round(tokens / elapsed)
    m["train_steps"] = step
    m["train_final_loss"] = round(loss.item(), 3)
    m["val_loss"] = round(val_loss, 3)
    m["ternary_bits_per_weight"] = round(st["bits_per_weight"], 3)
    print(f"\n  done: {step} steps, {tokens / elapsed / 1e3:.1f}k tokens/s, "
          f"val loss {val_loss:.3f} (random guessing would be {math.log(len(chars)):.2f})")
    print(f"  ternary weights: -1 {st['neg']:.1%} | 0 {st['zero']:.1%} | +1 {st['pos']:.1%}"
          f"  over {st['total']:,} weights")
    print(f"  storage: {st['bits_per_weight']:.3f} bits/weight "
          f"(log2(3) + 16/128 - the same formula as the Bonsai 2 paper)")
    print(f"           FP16 {st['fp16_kb']:,.0f} KB  ->  ternary {st['packed_kb']:,.0f} KB "
          f"({st['fp16_kb'] / st['packed_kb']:.1f}x smaller)")
    print("\n  --- the model writes ---------------------------------------------")
    print("  " + sample.replace("\n", "\n  "))
    print("  ----------------------------------------------------------------")


# ---------------------------------------------------------------- stage 5
def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


class SensorPoller(threading.Thread):
    QUERY = "temperature.gpu,power.draw,clocks.sm,fan.speed"

    def __init__(self, gpu_index=0, every=1.0):
        super().__init__(daemon=True)
        self.gpu_index, self.every = gpu_index, every
        self.samples, self.stop_event = [], threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}", f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                vals = [_num(v.strip()) for v in out.split(",")]
                if len(vals) == 4:
                    self.samples.append((time.time(), *vals))
            except Exception:
                pass
            self.stop_event.wait(self.every)


def stage_soak(dev, m, seconds):
    hdr(f"Stage 5 - Sustained load for {seconds}s (temperature, power, clocks)")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(dev) >= (8, 0) else torch.float16
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    c = torch.empty(n, n, device=dev, dtype=dtype)
    poller = SensorPoller(gpu_index=dev.index or 0)
    poller.start()
    windows = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        t0, k = time.perf_counter(), 0
        while time.perf_counter() - t0 < 5:
            for _ in range(10):
                torch.matmul(a, b, out=c)
            sync()
            k += 10
        windows.append(2 * n ** 3 * k / (time.perf_counter() - t0) / 1e12)
        s = poller.samples[-1] if poller.samples else None
        if s:
            print(f"  t={len(windows) * 5:4d}s  {windows[-1]:5.1f} TFLOPS  "
                  f"{s[1] or 0:3.0f} C  {s[2] or 0:5.1f} W  {s[3] or 0:5.0f} MHz  fan {s[4] or 0:3.0f}%",
                  flush=True)
    poller.stop_event.set()
    poller.join(timeout=6)
    del a, b, c
    torch.cuda.empty_cache()

    k = min(3, max(1, len(windows) // 2))
    start, end = sum(windows[:k]) / k, sum(windows[-k:]) / k
    drop = max(0.0, 1 - end / start)
    m["soak_tflops_start"], m["soak_tflops_end"] = round(start, 2), round(end, 2)
    m["soak_throttle_pct"] = round(drop * 100, 1)
    samples = poller.samples
    if samples:
        temps = [s[1] for s in samples if s[1] is not None]
        half = samples[len(samples) // 2:]
        powers = [s[2] for s in half if s[2] is not None]
        if temps:
            m["soak_max_temp_c"] = max(temps)
        if powers:
            m["soak_avg_power_w"] = round(sum(powers) / len(powers), 1)
    else:
        print("  (nvidia-smi not found - no sensor data, throughput only)")
    print(f"\n  throughput: start {start:.1f} -> end {end:.1f} TFLOPS ({drop:.0%} drop)")
    if "soak_max_temp_c" in m:
        print(f"  max temperature {m['soak_max_temp_c']:.0f} C, "
              f"average power (2nd half) {m.get('soak_avg_power_w', 0):.0f} W")
    hot = m.get("soak_max_temp_c", 0) > 83
    print("  ->", "WARN: running hot or throttling - check airflow/fan curve, or cap power at 150 W"
          if hot or drop > 0.10 else "PASS: cooler keeps up under sustained load")


# ---------------------------------------------------------------- report
def scorecard(m):
    hdr("SCORECARD (targets for an RTX 3060 12GB)")
    checks = [
        ("Genuine RTX 3060 12GB", m.get("looks_like_rtx3060_12gb")),
        ("FlashAttention available", m.get("flash_attention")),
        ("VRAM integrity", m.get("vram_bad_chunks") == 0 if "vram_bad_chunks" in m else None),
        ("BF16 matmul >= 15 TFLOPS", m["tflops_bf16"] >= 15 if "tflops_bf16" in m else None),
        ("Bandwidth >= 270 GB/s", m["bandwidth_gbs"] >= 270 if "bandwidth_gbs" in m else None),
        ("Ternary LM learned (val loss < 2.2)", m["val_loss"] < 2.2 if "val_loss" in m else None),
        ("Thermals: <= 83 C and < 10% throttle",
         (m.get("soak_max_temp_c", 0) <= 83 and m["soak_throttle_pct"] < 10)
         if "soak_throttle_pct" in m else None),
    ]
    for name, ok in checks:
        mark = "skipped" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {mark:8s} {name}")


def compare(m, old_path):
    with open(old_path, encoding="utf-8") as f:
        old = json.load(f)
    hdr(f"BEFORE vs AFTER  ({old.get('gpu_name', 'old')}  ->  {m.get('gpu_name', 'new')})")
    keys = ["vram_usable_gb", "tflops_fp32", "tflops_fp16", "tflops_bf16", "bandwidth_gbs",
            "train_tokens_per_s", "val_loss", "soak_max_temp_c", "soak_avg_power_w"]
    print(f"  {'metric':22s} {'before':>12s} {'after':>12s} {'change':>9s}")
    for k in keys:
        a, b = old.get(k), m.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ratio = f"{b / a:.2f}x" if a else "-"
            print(f"  {k:22s} {a:12,.2f} {b:12,.2f} {ratio:>9s}")
    for k in ("native_bf16", "flash_attention"):
        if k in old or k in m:
            print(f"  {k:22s} {str(old.get(k)):>12s} {str(m.get(k)):>12s}")


def main():
    ap = argparse.ArgumentParser(description="GPU Arrival Gauntlet")
    ap.add_argument("--quick", action="store_true", help="shorter run (~2 minutes)")
    ap.add_argument("--train-seconds", type=int, default=None)
    ap.add_argument("--soak-seconds", type=int, default=None)
    ap.add_argument("--skip-vram", action="store_true")
    ap.add_argument("--out", default="gauntlet_results.json")
    ap.add_argument("--compare", default=None, help="JSON from an earlier run to compare against")
    ap.add_argument("--cpu-smoke", action="store_true", help="tiny CPU run of stage 4 only")
    args = ap.parse_args()

    m = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.cpu_smoke:
        dev = torch.device("cpu")
        stage_ternary(dev, m, args.train_seconds or 15,
                      dict(d=128, layers=2, heads=4, ctx=64, batch=16))
        print("\nCPU smoke test finished - your Python setup works.")
        return

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible to PyTorch. Install a CUDA build of PyTorch from pytorch.org.")
    dev = torch.device("cuda", 0)
    train_s = args.train_seconds or (45 if args.quick else 120)
    soak_s = args.soak_seconds or (30 if args.quick else 180)

    stage_identity(dev, m)
    if not args.skip_vram:
        stage_vram(dev, m)
    stage_compute(dev, m, args.quick)
    stage_bandwidth(dev, m)
    stage_ternary(dev, m, train_s, dict(d=384, layers=6, heads=6, ctx=256, batch=64))
    stage_soak(dev, m, soak_s)
    scorecard(m)
    if args.compare:
        compare(m, args.compare)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
