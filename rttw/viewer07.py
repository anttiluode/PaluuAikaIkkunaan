"""Stage 0.7 figure.  Run:  python -m rttw.viewer07   (after rttw.stage07)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .composite import CompParams
from . import stage06 as S
from .stage07 import LAMS, GBS
from .viewer import SURF, INK, INK2, MUTED, GRID, S1, S2, S3
from .viewer06 import _episode

FIG = os.path.join(os.path.dirname(S.OUT), "figures")
YEL = "#eda100"
TESTS = (("fwd_novel", "forward, novel", S2), ("backward", "backward", S1),
         ("gaps", "gap filling (4 of 10)", S3), ("noise_episode_pairs", "noisy recitation", YEL))


def _val(o, key):
    return 0.5 * (o["gaps_familiar"] + o["gaps_novel"]) if key == "gaps" else o[key]


def figure(path):
    R = json.load(open(os.path.join(S.OUT, "stage07_results.json")))
    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.38, wspace=0.28,
                          left=0.05, right=0.99, top=0.9, bottom=0.04)
    fig.suptitle("Stage 0.7: who speaks when — one global schema gain vs a local gate",
                 x=0.05, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.05, 0.925, "Lines: the chain's content pathways scaled by a constant λ. "
             "Squares at the right: the gate (chain muted only while the ring delivers a bound word). "
             "Everything else identical.", fontsize=9, color=INK2)
    for j, gB in enumerate(GBS):
        ax = fig.add_subplot(gs[0, j])
        D = R[f"g_B={gB:g}"]
        for i, (key, lab, c) in enumerate(TESTS):
            ys = [_val(D[f"global {l:g}"], key) for l in LAMS]
            ax.plot(LAMS, ys, color=c, lw=2, marker="o", ms=4, label=lab)
            ax.plot([1.105 + 0.03 * i], [_val(D["gate"], key)], marker="s", ms=7, color=c,
                    mec=INK, mew=0.6)
        ax.axvline(1.075, color=GRID, lw=1)
        ax.set_xticks(list(LAMS) + [1.15])
        ax.set_xticklabels([f"{l:g}" for l in LAMS] + ["gate"], fontsize=8)
        ax.set_xlabel("global schema gain λ")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{'ab'[j]}  episode drive g_B = {gB:g}", loc="left", fontsize=10, fontweight="bold")
        if j == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[0, 2])
    for gB, c in zip(GBS, (S2, S1)):
        D = R[f"g_B={gB:g}"]
        ys = [D[f"global {l:g}"]["worst_of_four"] for l in LAMS]
        ax.plot(LAMS, ys, color=c, lw=2, marker="o", ms=4, label=f"global λ, g_B = {gB:g}")
        ax.axhline(D["gate"]["worst_of_four"], color=c, lw=1.5, ls="--")
        ax.text(0.1, D["gate"]["worst_of_four"] + 0.015, f"gate, g_B = {gB:g}", color=c, fontsize=8)
    ax.set_xlabel("global schema gain λ")
    ax.set_ylabel("worst of the four scores")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("c  The falsifier: can one global λ match the gate?", loc="left",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    W = S.weights()[0]
    sent = S.NOV[0]
    sub = gs[1, :].subgridspec(1, 2, wspace=0.12)
    for j, (comp, title) in enumerate((
            (CompParams(g_B=4.0), "d  backward, g_B = 4, chain always on (Stage 0.6)"),
            (CompParams(g_B=4.0, schema="gate"), "e  backward, g_B = 4, gate: the chain waits while the episode speaks"))):
        ax = fig.add_subplot(sub[0, j])
        C = S.make(W, comp, seed=3)
        C.hear(sent)
        outs = C.recall(11, direction=-1, start=10, record=True)
        _episode(ax, outs, title, target=sent[::-1])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    figure(os.path.join(FIG, "stage07_results.png"))
    print("figure written to", FIG)


if __name__ == "__main__":
    main()
