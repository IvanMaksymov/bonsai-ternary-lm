#!/usr/bin/env python3
"""
Compare ternary vs full-precision training runs from their log.csv files.

    python3 compare.py                      # uses logs/{small,base}_{ternary,fp}.csv
    python3 compare.py --pair "base 124M" runs/base_t/log.csv runs/base_fp/log.csv

Prints a step-by-step table of validation loss and the ternary - FP gap, and writes
light and dark versions of two figures to figures/:
  loss_curves[-dark].png  validation loss vs training tokens, one panel per model size
  gap[-dark].png          ternary minus full-precision validation loss vs training tokens
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light / dark steps of the same hues)
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ternary="#2a78d6", fp="#eb6834",
                  sizes=["#1baf7a", "#6250d6"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", ternary="#3987e5", fp="#d95926",
                 sizes=["#199e70", "#9085e9"]),
}


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [(int(r["step"]), float(r["tokens"]) / 1e9, float(r["val_loss"])) for r in rows]


def style_axes(ax, th):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=9, length=0, pad=6)
    ax.grid(True, color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th["ink2"])
    ax.yaxis.label.set_color(th["ink2"])


def end_label(ax, x, y, text, color, th, dy=0.0):
    ax.plot([x], [y], "o", ms=5, color=color, mec=th["surface"], mew=1.5, zorder=4)
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center",
                fontsize=9, color=th["ink"])


def plot_loss(pairs, mode, outdir):
    th = THEMES[mode]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.0), facecolor=th["surface"])
    axes = [axes] if len(pairs) == 1 else axes
    for ax, (name, t, f) in zip(axes, pairs):
        style_axes(ax, th)
        for data, key, label in ((t, "ternary", "Ternary"), (f, "fp", "Full precision")):
            ax.plot([d[1] for d in data], [d[2] for d in data], color=th[key], lw=2,
                    solid_capstyle="round", label=label)
        lt, lf = t[-1], f[-1]
        up = lt[2] >= lf[2]
        end_label(ax, lt[1], lt[2], f"Ternary {lt[2]:.3f}", th["ternary"], th, 7 if up else -7)
        end_label(ax, lf[1], lf[2], f"FP {lf[2]:.3f}", th["fp"], th, -7 if up else 7)
        ax.set_title(name, loc="left", fontsize=11, color=th["ink"], pad=10)
        ax.set_xlabel("Training tokens (billions)")
        ax.set_ylabel("Validation loss")
        ax.set_xlim(0, max(lt[1], lf[1]) * 1.32)
        lo = min(lt[2], lf[2])
        ax.set_ylim(lo - 0.12, lo + 1.15)
    leg = axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(th["ink2"])
    fig.tight_layout()
    path = os.path.join(outdir, f"loss_curves{'-dark' if mode == 'dark' else ''}.png")
    fig.savefig(path, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return path


def plot_gap(pairs, mode, outdir):
    th = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor=th["surface"])
    style_axes(ax, th)
    ax.axhline(0, color=th["axis"], lw=1.2, zorder=1)
    xmax = 0
    for (name, t, f), color in zip(pairs, th["sizes"]):
        fv = {s: v for s, _, v in f}
        pts = [(x, v - fv[s]) for s, x, v in t if s in fv]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2,
                solid_capstyle="round", label=name, zorder=3)
        end_label(ax, pts[-1][0], pts[-1][1], f"{name}: {pts[-1][1]:+.3f}", color, th)
        xmax = max(xmax, pts[-1][0])
    ax.set_xlim(0, xmax * 1.35)
    ax.set_title("Cost of ternary weights (ternary − full-precision validation loss)", loc="left",
                 fontsize=11, color=th["ink"], pad=10)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Loss gap (nats)")
    ax.annotate("below 0: ternary ahead", (xmax * 1.33, 0), xytext=(0, -10), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=th["muted"])
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(th["ink2"])
    fig.tight_layout()
    path = os.path.join(outdir, f"gap{'-dark' if mode == 'dark' else ''}.png")
    fig.savefig(path, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "TERNARY_CSV", "FP_CSV"))
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    specs = args.pair or [["small · 51M params", "logs/small_ternary.csv", "logs/small_fp.csv"],
                          ["base · 124M params", "logs/base_ternary.csv", "logs/base_fp.csv"]]
    pairs = [(n, read_log(t), read_log(f)) for n, t, f in specs]
    os.makedirs(args.out, exist_ok=True)

    for name, t, f in pairs:
        fv = {s: v for s, _, v in f}
        print(f"\n{name}\n  {'step':>6} {'tokens':>8} {'ternary':>8} {'fp':>8} {'gap':>7}")
        for s, x, v in t:
            if s in fv:
                print(f"  {s:6d} {x:7.2f}B {v:8.4f} {fv[s]:8.4f} {v - fv[s]:+7.4f}")
        lt, lf = t[-1][2], f[-1][2]
        print(f"  final: ternary {lt:.4f} (ppl {math.exp(lt):.2f}), fp {lf:.4f} (ppl {math.exp(lf):.2f}), "
              f"gap {lt - lf:+.4f} nats, perplexity +{(math.exp(lt - lf) - 1) * 100:.1f}%")
    for mode in ("light", "dark"):
        print("wrote", plot_loss(pairs, mode, args.out), "and", plot_gap(pairs, mode, args.out))


if __name__ == "__main__":
    main()
