"""Stage 0.5 figures.  Run:  python -m rttw.viewer05   (after rttw.stage05)"""
import json, os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ring import Ring, SINGLE, PAIR_OR, PAIR_AND, onsets_of
from .stage05 import OUT, WORDS, TAUS, NS, RATES, LAGS
from .viewer import SURF, INK, INK2, MUTED, GRID, S1, S2, S3, SEQ

FIG = os.path.join(os.path.dirname(OUT), "figures")
YEL = "#eda100"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})


def _panel(ax, rec, title, mark=None):
    M = np.array([r[2][0] for r in rec])            # time x N (ring 0)
    T = len(M)
    t0 = rec[0][0]
    ax.imshow(M.T, aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest",
              extent=(0, T, 7.5, -0.5))
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"{k}  {w}" for k, w in enumerate(WORDS)], fontsize=8)
    ax.set_xlabel("time (ms)", fontsize=8)
    ticks = np.arange(0, T + 1, 100)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t * 2.5)}" for t in ticks], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ons = onsets_of(rec, 0)
    if mark is None:
        text = " ".join(WORDS[k] for _, k in ons[:18])
    else:
        t_op = mark[0]
        before = [WORDS[k] for t, k in ons if t < t_op][-7:]
        after = [WORDS[k] for t, k in ons if t >= t_op][:11]
        text = "… " + " ".join(before) + "   ‖   " + " ".join(after) + " …"
        ax.axvline(t_op - t0, color=S2, lw=1.4)
        ax.text(t_op - t0 + 4, 7.3, mark[1], color=S2, fontsize=8, va="bottom")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.text(0, -0.36, "published:  " + text, transform=ax.transAxes, fontsize=9, color=INK,
            va="top")


def ring_viewer(path):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10.5))
    fig.subplots_adjust(hspace=0.95, left=0.1, right=0.98, top=0.88, bottom=0.08)
    fig.suptitle("The route door: one ring, symmetric wiring, direction is a state",
                 x=0.1, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.1, 0.935, "Region k stores word k. Each region excites both neighbours equally; its own "
             "burst recruits local PV (dead time). Where the start is blocked decides the direction.",
             fontsize=9, color=INK2)
    R = Ring(SINGLE, seed=1, words=WORDS)
    _panel(axs[0], R.kick(0, +1) + R.run(460), "a  kicked at 'winter', back side blocked on the left: reads forward")
    R = Ring(SINGLE, seed=1, words=WORDS)
    _panel(axs[1], R.kick(0, -1) + R.run(460), "b  same weights, same words, blocked on the right: reads backward")
    R = Ring(SINGLE, seed=1, words=WORDS)
    rec = R.kick(0, +1) + R.run(226)
    here = onsets_of(rec, 0)[-1][1]
    t_op = R.t
    rec += R.silence(20)
    rec += R.kick(here, -1) + R.run(200)
    _panel(axs[2], rec, f"c  'reverse from here': silence 50 ms, restart at '{WORDS[here]}' blocked on the other side",
           mark=(t_op, "reverse"))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results(path):
    R = json.load(open(os.path.join(OUT, "stage05_results.json")))
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.4))
    fig.subplots_adjust(hspace=0.5, wspace=0.28, left=0.07, right=0.98, top=0.9, bottom=0.08)
    fig.suptitle("Stage 0.5 results", x=0.07, ha="left", fontsize=12, fontweight="bold")

    # (a) band
    ax = axs[0, 0]
    col = {"split": S2, "one-way": S1, "dies": MUTED}
    for i, N in enumerate(NS):
        for j, tp in enumerate(TAUS):
            d = R["R1"]["grid"][f"N={N} tau_p={tp}"]
            o = max(d, key=d.get)
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=col[o], lw=0))
    ax.set_xlim(-0.6, len(TAUS) - 0.4)
    ax.set_ylim(len(NS) - 0.5, -0.5)
    ax.set_xticks(range(len(TAUS)))
    ax.set_xticklabels([f"{R['R1']['refractory'][str(tp)]}" for tp in TAUS])
    ax.set_xlabel("refractory period after a burst (steps of 2.5 ms)")
    ax.set_yticks(range(len(NS)))
    ax.set_yticklabels([f"{N} regions\nlap {R['R1']['lap'][str(N)]:.0f}" for N in NS], fontsize=8)
    for o, c in col.items():
        ax.scatter([], [], marker="s", s=60, color=c, label=o)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_title("a  Dead time must be long enough and short enough", loc="left",
                 fontsize=10, fontweight="bold", pad=24)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    # (b) survival under noise
    ax = axs[0, 1]
    styles = {"single ring": (INK2, "-"), "OR pair, private": (S2, "-"), "OR pair, shared": (S2, ":"),
              "AND pair, private": (S1, "-"), "AND pair, shared": (S1, ":")}
    xs = [ps * 400 for ps in RATES]
    for name, (c, ls) in styles.items():
        ys = [R["R45"][name][f"{x:.2f}/s"]["survival"] for x in xs]
        ax.plot(xs, ys, color=c, ls=ls, lw=2, marker="o", ms=4, label=name)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("spurious ignitions per region per second")
    ax.set_ylabel("fraction of run before the wave is lost")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("b  Spurious ignitions kill the loop, in every wiring", loc="left",
                 fontsize=10, fontweight="bold")

    # (c) clean bursts + reversal
    ax = axs[1, 0]
    for name, (c, ls) in styles.items():
        ys = [R["R45"][name][f"{x:.2f}/s"]["clean"] for x in xs]
        ax.plot(xs, ys, color=c, ls=ls, lw=2, marker="o", ms=4, label=name)
    rev = max(R["R45"][n][f"{x:.2f}/s"]["reversed"] for n in styles for x in xs)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("spurious ignitions per region per second")
    ax.set_ylabel("bursts that are part of the + wave")
    ax.set_ylim(0, 1.02)
    ax.text(0.98, 0.95, f"laps running backward: at most {rev:.0%} in any condition",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("c  AND stops noise spreading, but not killing", loc="left",
                 fontsize=10, fontweight="bold")

    # (d) lag
    ax = axs[1, 1]
    for name, c in (("OR pair", S2), ("AND pair", S1)):
        ys = [R["R6"][name][str(l)]["alive"] for l in LAGS]
        ax.plot([l * 2.5 for l in LAGS], ys, color=c, lw=2, marker="o", ms=5, label=name)
    ax.set_xlabel("start offset between the two rings (ms)")
    ax.set_ylabel("laps with one + wave")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fl = [R["R6"]["AND pair"][str(l)]["final_lag_steps"] for l in LAGS]
    ok = [l for l, f in zip(LAGS, fl) if f is not None]
    if ok:
        ax.text(0.98, 0.55, f"surviving runs end at zero lag\n(|final lag| < 1 step) from starts up to {max(ok) * 2.5:g} ms",
                transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.set_title("d  Two rings lock at zero lag, if they start within one burst", loc="left",
                 fontsize=10, fontweight="bold")
    for a in (axs[0, 1], axs[1, 0], axs[1, 1]):
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    ring_viewer(os.path.join(FIG, "ring_viewer.png"))
    results(os.path.join(FIG, "stage05_results.png"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
