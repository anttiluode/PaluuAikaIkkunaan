"""Window viewer and result figures.  Run:  python -m rttw.viewer  (after experiments)"""
import json, os, pickle
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .grammar import VOCAB, IDX, SLOT, TOPICS, WORLD as WORLD_OF, story, valid_transition
from .machine import Machine, Params
from .experiments import OUT, fresh, warm

FIG = os.path.join(os.path.dirname(OUT), "figures")

# ---- palette (reference data-viz palette, light surface)
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
DEAD = "#ecebe7"
SEQ = LinearSegmentedColormap.from_list(
    "seqblue", [SURF, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TOPIC_COLORS = {"winter": S1, "beach": S2, "night": S3}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})

ROW_ORDER = sorted(VOCAB, key=lambda w: (SLOT[w], VOCAB.index(w)))


# ------------------------------------------------------------------ episodes
def run_episode(m, script):
    """script: list of dicts with keys word (None = eyes closed), ais (bool),
    and optional reject=(route, item, base). Returns per-cycle records."""
    recs = []
    for step in script:
        ctx0 = m.ctx.copy()
        veto0 = m.veto.copy()
        o = m.cycle(step.get("word"), ais_open=step.get("ais", True), record=True)
        o["ctx0"], o["veto0"] = ctx0, veto0
        o["eyes"] = "open" if step.get("word") is not None else "closed"
        o["rejected"] = None
        if step.get("reject"):
            route, item, base = step["reject"]
            m.reject(o["published"], base=base, route=route, item=item)
            o["rejected"] = o["published"]
        recs.append(o)
    return recs


def plot_episode(recs, title, path, subtitle=None, show_pred=False, mark=None):
    T = recs[0]["r"].shape[0]
    n = len(recs)
    R = np.concatenate([o["r"] for o in recs], axis=0)          # time x V
    rows = [IDX[w] for w in ROW_ORDER]
    img = R[:, rows].T

    nrow_txt = 4 if show_pred else 3
    fw = max(9, 0.62 * n + 2.2)
    fig = plt.figure(figsize=(fw, 8.4))
    L0 = 1.25 / fw   # fixed 1.25 in left margin for row labels
    gs = fig.add_gridspec(4, 1, height_ratios=[6.2, 0.35 * nrow_txt + 0.4, 1.1, 1.1],
                          hspace=0.12, left=L0, right=0.985, top=0.87, bottom=0.04)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img, aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest",
              extent=(0, n * T, len(rows) - 0.5, -0.5))
    p = Params()
    for c in range(n):
        ax.axvspan(c * T + (T - p.dead), (c + 1) * T, color=DEAD, alpha=0.55, lw=0)
        ax.axvline(c * T, color=GRID, lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ROW_ORDER, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(0, n * T)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.suptitle(title, x=L0, y=0.985, ha="left", fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        import textwrap
        width = int(fig.get_figwidth() * 13)
        fig.text(L0, 0.935, "\n".join(textwrap.wrap(subtitle, width)), ha="left",
                 va="top", fontsize=9, color=INK2)

    # ---- text rows
    tx = fig.add_subplot(gs[1], sharex=ax)
    tx.set_ylim(-0.5, nrow_txt - 0.5)
    tx.invert_yaxis()
    labels = ["input (eyes)", "internal", "published (AIS)"] + (["prediction"] if show_pred else [])
    tx.set_yticks(range(nrow_txt))
    tx.set_yticklabels(labels, fontsize=8)
    tx.tick_params(length=0)
    tx.set_xticks([])
    for sp in tx.spines.values():
        sp.set_visible(False)
    for c, o in enumerate(recs):
        x = c * T + T / 2
        inp = o["word"] if o["word"] is not None else "·"
        tx.text(x, 0, inp, ha="center", va="center", fontsize=8,
                color=INK if o["word"] is not None else MUTED)
        if o["word"] is None:
            tx.axvspan(c * T, (c + 1) * T, ymin=2 / 3 + 0.05 if False else 0.0, ymax=0,
                       color="none")
        cur = o["confirmed"] or "–"
        tx.text(x, 1, cur, ha="center", va="center", fontsize=8, color=INK)
        if o["ais_open"]:
            pub = o["published"] or "–"
            col = INK
            if o["rejected"]:
                pub = f"{pub} ✕"
                col = S2
            tx.text(x, 2, pub, ha="center", va="center", fontsize=8, color=col,
                    fontweight="bold")
        else:
            tx.add_patch(plt.Rectangle((c * T + 2, 1.6), T - 4, 0.8, color=DEAD, lw=0))
            tx.text(x, 2, "AIS shut", ha="center", va="center", fontsize=7, color=INK2)
        if show_pred:
            tx.text(x, 3, o["pred"] or "–", ha="center", va="center", fontsize=8,
                    color=INK2)
    if mark:
        for c, txt in mark.items():
            tx.text(c * T + T / 2, nrow_txt - 0.5 + 0.02, txt, ha="center", va="top",
                    fontsize=7, color=S2)

    # ---- context and veto rows (topic units)
    cyc = np.arange(n) * T + T / 2
    for gi, key, lab in ((2, "ctx0", "context\n(topic units)"), (3, "veto0", "SST veto\n(topic routes)")):
        a2 = fig.add_subplot(gs[gi], sharex=ax)
        for w, col in TOPIC_COLORS.items():
            y = np.array([o[key][IDX[w]] for o in recs])
            a2.plot(cyc, y, color=col, lw=2, marker="o", ms=3.5)
            if y.max() > 0.05:
                j = int(np.argmax(y))
                a2.text(cyc[j] + T * 0.15, y[j], w, color=INK2, fontsize=7, va="bottom")
        a2.text(-0.005, 0.5, lab, transform=a2.transAxes, fontsize=8, color=INK2,
                ha="right", va="center")
        a2.set_yticks([])
        a2.set_xticks([])
        a2.grid(False)
        top = max(1.0, max(o[key].max() for o in recs) * 1.15) if key == "ctx0" else 1.15
        a2.set_ylim(-0.05, top)
        for sp in ("left", "bottom"):
            a2.spines[sp].set_color(GRID)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ episode scripts
def episode_window(W, seed=0):
    m = fresh(W, seed=21)
    rng = np.random.default_rng(21)
    warm(m, rng)
    script = [dict(word=w) for w in ["winter", "girl", "walks", "to"]]
    free = [dict(word=None) for _ in range(11)]
    free[3]["ais"] = False
    free[4]["ais"] = False
    recs = run_episode(m, script + free)
    plot_episode(recs, "Reading, then eyes closed",
                 os.path.join(FIG, "window_viewer.png"),
                 subtitle="Four words are read; then input stops and the carried prediction keeps the story going. "
                          "Two cycles have the AIS shut: the sequence runs on unpublished. Grey = PV dead time.")
    return recs


def _garden_trial(W, sst, seed_i, topic_gain, topic="winter", A="sea"):
    m = fresh(W, seed=seed_i, sst=sst, topic_gain=topic_gain)
    rng = np.random.default_rng(seed_i)
    warm(m, rng)
    st = story(rng, topic=topic, subj="girl", A_override=A)
    return run_episode(m, [dict(word=w) for w in st]), st


def episode_garden(W, topic_gain=3.0):
    # first trial (in seed order) where SST off follows the topic at B and SST on
    # follows the A word -- an illustration, the rates are in results
    found = False
    for s in range(60):
        for topic, A in (("winter", "sea"), ("beach", "bed"), ("night", "hill"),
                         ("winter", "bed"), ("beach", "hill"), ("night", "sea")):
            on, st = _garden_trial(W, True, 500 + s, topic_gain, topic, A)
            off, _ = _garden_trial(W, False, 500 + s, topic_gain, topic, A)
            B_A, B_topic = TOPICS[WORLD_OF[A]][1], TOPICS[topic][1]
            if on[6]["pred"] == B_A and off[6]["pred"] == B_topic:
                found = True
                break
        if found:
            break
    for recs, tag in ((on, "on"), (off, "off")):
        plot_episode(recs, f"Garden path while reading — SST {tag}",
                     os.path.join(FIG, f"garden_path_sst_{tag}.png"), show_pred=True,
                     subtitle=f"{st[0]} … to {st[4].upper()}: the topic points one way, the A word another; "
                              f"by the world rule B must follow the A word ({st[7]}). "
                              f"Topic written with gain {topic_gain:g}. "
                              "Prediction row = what the late slot predicts for the next word. "
                              "(Example trial; rates are in results.)")
    return on, off


def episode_no(W):
    for s in range(40):
        m = fresh(W, seed=900 + s)
        rng = np.random.default_rng(900 + s)
        warm(m, rng)
        pre = [dict(word=w) for w in ["beach", "boy", "walks", "to"]]
        free = [dict(word=None, reject=(True, True, "to"))] + [dict(word=None) for _ in range(9)]
        recs = run_episode(m, pre + free)
        if recs[4]["published"] == "sea":
            break
    plot_episode(recs, "“No, not that” — reject a branch mid-sequence",
                 os.path.join(FIG, "no_not_that.png"),
                 subtitle="Eyes closed after '… walks to'. The machine says 'sea'; an outside 'no' arrives. "
                          "SST closes the route that steered it, the item is suppressed, and the sequence "
                          "re-launches from 'to'.")
    return recs


# ------------------------------------------------------------------ result figure
def results_figure(path_json, out):
    R = json.load(open(path_json))
    P = R["pooled"]
    W = pickle.load(open(os.path.join(OUT, "weights.pkl"), "rb"))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.subplots_adjust(hspace=0.55, wspace=0.32, left=0.06, right=0.98, top=0.9, bottom=0.08)
    fig.suptitle(f"Stage 0 results — {len(R['seeds'])} seeds, mean with min–max across seeds",
                 x=0.06, ha="left", fontsize=12, fontweight="bold")

    def err(d):
        return d["mean"], [[d["mean"] - d["min"]], [d["max"] - d["min"] - (d["mean"] - d["min"])]]

    # (a) learning curve
    ax = axs[0, 0]
    curves = np.array([W["curves"][s] for s in R["seeds"]])  # seeds x stories x 2
    k = 20
    def smooth(x):
        return np.convolve(x, np.ones(k) / k, mode="valid")
    xs = np.arange(curves.shape[1] - k + 1) + k
    for j, (lab, col) in enumerate((("context branches", S1), ("deterministic", S2))):
        ys = np.array([smooth(c[:, j]) for c in curves])
        ax.fill_between(xs, ys.min(0), ys.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(xs, ys.mean(0), color=col, lw=2)
        ax.text(xs[-1], ys.mean(0)[-1] - (0.08 if j == 0 else -0.04), lab, color=INK2, ha="right", fontsize=8)
    ax.axhline(1 / 3, color=MUTED, lw=1, ls=":")
    ax.text(10, 1 / 3 + 0.02, "chance at a 3-way branch", color=MUTED, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("stories read (local rules only)")
    ax.set_ylabel("next word predicted")
    ax.set_title("a  Learning while reading", loc="left", fontsize=10, fontweight="bold")

    # (b) dead time
    ax = axs[0, 1]
    from .experiments import DEADS, SPONTS
    cols = [S1, S2, S3, "#eda100"]
    for j, ps in enumerate(SPONTS):
        ys = [P["T3"][f"spont={ps} dead={d}"]["valid"] for d in DEADS]
        m_ = [y["mean"] for y in ys]
        lo = [y["mean"] - y["min"] for y in ys]
        hi = [y["max"] - y["mean"] for y in ys]
        ax.errorbar(DEADS, m_, yerr=[lo, hi], color=cols[j], lw=2, marker="o", ms=5,
                    capsize=0, label=f"{ps * 1000:g} spurious ignitions / cycle")
    ax.invert_xaxis()
    ax.set_xticks(DEADS)
    ax.set_xlabel("PV dead time (steps of 50)")
    ax.set_ylabel("valid transitions, eyes closed")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("b  Remove the dead time", loc="left", fontsize=10, fontweight="bold")

    # (c) apical rule
    ax = axs[0, 2]
    from .experiments import MULT_GAINS, ADD_GAINS
    for kind, gains, col, lab in (("mult", MULT_GAINS, S1, "multiplicative"),
                                  ("add", ADD_GAINS, S2, "additive")):
        gains = [g for g in gains if g > 0]
        fv = [P["T4"][kind][str(g)]["free_valid"] for g in gains]
        intr = [P["T4"][kind][str(g)]["intrusions_per_cycle"]["mean"] for g in gains]
        ax.errorbar(gains, [x["mean"] for x in fv],
                    yerr=[[x["mean"] - x["min"] for x in fv], [x["max"] - x["mean"] for x in fv]],
                    color=col, lw=2, capsize=0, label=lab, zorder=2)
        for g, x, i in zip(gains, fv, intr):
            hollow = i > 0.05
            ax.plot(g, x["mean"], "o", ms=7, mfc=SURF if hollow else col, mec=col, mew=2, zorder=3)
            if hollow:
                ax.annotate(f"{i:.2f}", (g, x["mean"]), textcoords="offset points",
                            xytext=(6, 2), fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 4, 6])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "4", "6"])
    ax.set_xlabel("apical gain (log scale)")
    ax.set_ylabel("eyes-closed run valid and in the cued world")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(0.62, 0.5, "hollow = context fires words with no\nbasal support (number = per cycle)",
            transform=ax.transAxes, ha="center", fontsize=7, color=INK2)
    ax.set_title("c  Apical rule: multiplicative vs additive", loc="left", fontsize=10, fontweight="bold")

    # (d) silent vs freeze
    ax = axs[1, 0]
    conds = ("AIS closed", "freeze (trace held)", "freeze (trace decays)")
    outcomes = (("on time", S1), ("stale", S2), ("silent/died", MUTED))
    ks = (1, 2, 3, 4)
    width = 0.26
    for ci, c in enumerate(conds):
        bottom = np.zeros(len(ks))
        for o, col in outcomes:
            v = np.array([P["T5"][f"{c} | k={k}"][o]["mean"] for k in ks])
            ax.bar(np.arange(len(ks)) + (ci - 1) * width, v, width * 0.92, bottom=bottom,
                   color=col, edgecolor=SURF, lw=1, label=o if ci == 0 else None)
            bottom += v
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("first word after release (fraction)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    ax.set_xlabel("cycles closed\nbars in each group: AIS shut | rhythm frozen, trace held | rhythm frozen, trace decays",
                  fontsize=8)
    ax.set_title("d  Silence the output vs stop the rhythm", loc="left", fontsize=10, fontweight="bold")

    # (e) garden path vs topic gain
    ax = axs[1, 1]
    from .experiments import TOPIC_GAINS
    for tag, col in (("SST on", S1), ("SST off", S2)):
        d = [P["T6b"][f"topic_gain={tg}"][tag]["B_follows_A"] for tg in TOPIC_GAINS]
        ax.errorbar(TOPIC_GAINS, [x["mean"] for x in d],
                    yerr=[[x["mean"] - x["min"] for x in d], [x["max"] - x["mean"] for x in d]],
                    color=col, lw=2, marker="o", ms=5, capsize=0, label=tag)
    ax.set_xticks(TOPIC_GAINS)
    ax.set_xlabel("how strongly the topic was written into context")
    ax.set_ylabel("B follows the A word (not the topic)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("e  Garden path while reading", loc="left", fontsize=10, fontweight="bold")

    # (f) no, not that
    ax = axs[1, 2]
    from .experiments import T6_CONDS
    conds = [c for c in T6_CONDS if c != "accept (no 'no')"]
    series = (("avoided", "1.0", S1, "avoids the rejected word, topic gain 1"),
              ("avoided", "3.0", S3, "avoids the rejected word, topic gain 3"),
              ("downstream_consistent", "1.0", S2, "B, C follow the new A word, gain 1"),
              ("downstream_consistent", "3.0", "#eda100", "B, C follow the new A word, gain 3"))
    y = np.arange(len(conds))
    h = 0.2
    for j, (met, tg, col, lab) in enumerate(series):
        v = [P["T6"][f"topic_gain={tg}"][c][met]["mean"] for c in conds]
        ax.barh(y + (j - 1.5) * h, v, h * 0.9, color=col, edgecolor=SURF, lw=1, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("backup", "re-launch") for c in conds], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(len(conds) - 0.5, -1.6)
    ax.set_xlabel("fraction of trials")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2, bbox_to_anchor=(-0.02, 1.02))
    ax.set_title("f  \u201cNo, not that\u201d with eyes closed", loc="left", fontsize=10, fontweight="bold")

    for a in axs.flat:
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    W = pickle.load(open(os.path.join(OUT, "weights.pkl"), "rb"))
    w0 = W["weights"][0]
    os.makedirs(FIG, exist_ok=True)
    episode_window(w0)
    episode_garden(w0)
    episode_no(w0)
    results_figure(os.path.join(OUT, "results.json"), os.path.join(FIG, "results.png"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
