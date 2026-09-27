"""Stage 0.6 figures.  Run:  python -m rttw.viewer06   (after rttw.stage06)"""
import json, os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .grammar import VOCAB, IDX, SLOT
from .composite import CompParams
from . import stage06 as S
from .viewer import SURF, INK, INK2, MUTED, GRID, S1, S2, S3, SEQ, ROW_ORDER

FIG = os.path.join(os.path.dirname(S.OUT), "figures")
YEL = "#eda100"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})


def _episode(ax, outs, title, target=None, gaps=()):
    ring = np.concatenate([o["ring_r"] for o in outs], axis=0)      # time x 11
    words = np.concatenate([o["r"] for o in outs], axis=0)          # time x V
    rows = [IDX[w] for w in ROW_ORDER]
    img = np.concatenate([ring.T, np.full((1, ring.shape[0]), np.nan), words[:, rows].T], axis=0)
    n = len(outs)
    T = outs[0]["r"].shape[0]
    cmap = SEQ.copy()
    cmap.set_bad(SURF)
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=(0, n * T, img.shape[0] - 0.5, -0.5))
    labels = [f"place {k}" for k in range(11)] + [""] + ROW_ORDER
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for c in range(n):
        ax.axvspan(c * T + (T - 18), (c + 1) * T, color="#ecebe7", alpha=0.45, lw=0)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    y = img.shape[0] + 0.6
    for c, o in enumerate(outs):
        w = o["published"] if o["word"] is None else o["word"]
        col = INK
        if target is not None and o["word"] is None:
            col = S1 if w == target[c] else S2
        ax.text(c * T + T / 2, y, w or "–", ha="center", va="top", fontsize=8, color=col,
                fontweight="bold" if c in gaps else "normal")
        if c in gaps:
            ax.text(c * T + T / 2, y + 1.4, "gap", ha="center", va="top", fontsize=7, color=INK2)


def composite_viewer(path):
    W = S.weights()[0]
    sent = S.NOV[0]
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    fig.subplots_adjust(hspace=0.28, wspace=0.14, left=0.06, right=0.99, top=0.9, bottom=0.05)
    fig.suptitle("The composite: a ring of places bound to the chain, heard once", x=0.06,
                 ha="left", fontsize=12, fontweight="bold")
    fig.text(0.06, 0.925, f"Heard once: “{' '.join(sent)}” — a sentence the chain's "
             "training never contained (winter stories go to the hill). Top rows: the 11 ring places; "
             "bottom rows: word assemblies. Blue word = matches the heard sentence, orange = does not.",
             fontsize=9, color=INK2, wrap=True)

    C = S.make(W, CompParams(g_B=S.G_MAIN), seed=3)
    C.reset()
    C.B[:] = 0
    C.ring.schedule_kick(0, +1)
    outs = []
    for w in sent:
        o = C.cycle(w, drive=False, record=True)
        if o["ring_region"] is not None:
            C.B[o["ring_region"], IDX[w]] = 1.0
        outs.append(o)
    _episode(axs[0, 0], outs, "a  hearing it once: each place binds the word current with it")

    outs = C.recall(11, record=True)
    _episode(axs[0, 1], outs, "b  eyes closed, ring forward: the episode wins over the schema", target=sent)

    B_full = C.B.copy()
    gaps = (4, 7)
    for g in gaps:
        C.B[g] = 0.0
    outs = C.recall(11, record=True)
    _episode(axs[1, 0], outs, "c  places 4 and 7 never bound: the chain fills the gaps with its schema",
             target=sent, gaps=gaps)
    C.B = B_full

    outs = C.recall(11, direction=-1, start=10, record=True)
    _episode(axs[1, 1], outs, "d  ring started at the end going backward", target=sent[::-1])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def results(path):
    R = json.load(open(os.path.join(S.OUT, "stage06_results.json")))
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    fig.subplots_adjust(hspace=0.55, wspace=0.3, left=0.05, right=0.99, top=0.9, bottom=0.1)
    fig.suptitle("Stage 0.6 results", x=0.05, ha="left", fontsize=12, fontweight="bold")
    xs = [ps * 400 for ps in S.RATES]
    cols = {1: INK2, 4: YEL, 8: S3, 16: S1}

    for ax, key, title in ((axs[0, 0], "free", "a  Free ring: more units per place"),
                           (axs[0, 1], "locked", "b  Ring locked to the chain's rhythm")):
        for P in S.PS:
            ys = [R["C1"][key][f"P={P} {x:g}/s"] for x in xs]
            ax.plot(xs, ys, color=cols[P], lw=2, marker="o", ms=4, label=f"{P} unit{'s' if P > 1 else ''} per place")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel("spurious ignitions per unit per second")
        ax.set_ylabel("fraction of run before the wave is lost")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8, loc="center left")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")

    # (c) forward / backward / jump bars
    ax = axs[0, 2]
    groups = [("forward", "familiar"), ("forward", "novel"), ("backward", "familiar"),
              ("backward", "novel"), ("jump", "familiar"), ("jump", "novel")]
    modes = [("chain only", INK2), ("ring only", S3), (f"composite g_B={S.G_MAIN:g}", S1)]
    x = np.arange(len(groups))
    wdt = 0.26
    for j, (m, c) in enumerate(modes):
        vals = []
        for task, kind in groups:
            if task == "jump":
                mm = "composite" if m.startswith("composite") else m
                vals.append(R["C4"][mm][kind])
            else:
                vals.append(R["C23"][task].get(m, {}).get(kind, np.nan))
        ax.bar(x + (j - 1) * wdt, np.nan_to_num(vals), wdt * 0.9, color=c, edgecolor=SURF,
               label=m.replace(f" g_B={S.G_MAIN:g}", ""))
        for xi, v in zip(x, vals):
            if np.isnan(v):
                ax.text(xi + (j - 1) * wdt, 0.02, "can't", rotation=90, fontsize=6.5, color=INK2,
                        ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{k}" for t, k in groups], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("words correct")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_title("c  Recall after hearing once", loc="left", fontsize=10, fontweight="bold", pad=26)

    # (d) g_B sweep
    ax = axs[1, 0]
    for kind, c in (("familiar", S3), ("novel", S2)):
        for task, ls in (("forward", "-"), ("backward", ":")):
            ys = [R["C23"][task][f"composite g_B={g:g}"][kind] for g in S.GAINS]
            ax.plot(S.GAINS, ys, color=c, ls=ls, lw=2, marker="o", ms=4, label=f"{task}, {kind}")
    ax.axhline(R["C23"]["forward"]["chain only"]["novel"], color=INK2, lw=1, ls="--")
    ax.text(S.GAINS[-1], R["C23"]["forward"]["chain only"]["novel"] + 0.02, "chain only, novel",
            ha="right", fontsize=7.5, color=INK2)
    ax.set_xlabel("episode drive g_B (place → its word)")
    ax.set_ylabel("words correct")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("d  How strong must the episode be?", loc="left", fontsize=10, fontweight="bold")

    # (e) partial episodes
    ax = axs[1, 1]
    ks = ["0 missing", "2 missing", "4 missing"]
    for mode, c in (("ring only", S3), ("composite", S1)):
        for kind, ls in (("familiar", "-"), ("novel", ":")):
            ys = [R["C5"][k][mode][kind]["accuracy"] for k in ks]
            if mode == "ring only" and kind == "novel":
                continue          # identical to familiar: the ring doesn't know the schema
            lab = "ring only (familiar = novel)" if mode == "ring only" else f"{mode}, {kind}"
            ax.plot([0, 2, 4], ys, color=c, ls=ls, lw=2, marker="o", ms=4, label=lab)
    ax.set_xticks([0, 2, 4])
    ax.set_xlabel("places never bound (of 10)")
    ax.set_ylabel("words correct")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("e  Gaps in the episode: the chain fills them", loc="left", fontsize=10, fontweight="bold")

    # (f) noise + restart
    ax = axs[1, 2]
    conds = ["ring only", "ring only + restart", "composite", "composite + restart"]
    keys = [k for k in R["C6"]]
    x = np.arange(len(keys))
    wdt = 0.2
    ccol = [S3, "#0b6b4a", S1, "#1c5cab"]
    for j, cnd in enumerate(conds):
        ep = [R["C6"][k][cnd]["episode_pairs"] for k in keys]
        ax.bar(x + (j - 1.5) * wdt, ep, wdt * 0.9, color=ccol[j], edgecolor=SURF, label=cnd)
        intr = [R["C6"][k][cnd]["plausible_intrusions"] for k in keys]
        ax.scatter(x + (j - 1.5) * wdt, intr, marker="_", s=90, color=S2, zorder=3,
                   label="plausible but wrong" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(" ", "\n") for k in keys], fontsize=8)
    ax.set_ylabel("consecutive words that follow the episode")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    ax.set_title("f  Noisy ring, 4 recitations", loc="left", fontsize=10, fontweight="bold", pad=34)

    for a in axs.flat:
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    composite_viewer(os.path.join(FIG, "composite_viewer.png"))
    results(os.path.join(FIG, "stage06_results.png"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
