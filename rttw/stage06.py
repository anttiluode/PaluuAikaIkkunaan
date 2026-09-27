"""Stage 0.6 experiments: the composite (ring = place order, chain = learned order).
Run:  python -m rttw.stage06   (from the repo root, after rttw.experiments)

C1  populations     does spreading a region over P units fix the ring's fragility?
C2  forward recall  heard once; chain only vs ring only vs composite, familiar vs novel
C3  backward recall same, backward
C4  jump in         "what came after X?"
C5  partial episode some positions never bound; who fills the gaps, and with what?
C6  noise + restart long recall with a noisy ring; restart from the last word
C7  checkpoint      mirrored rings, OR coupling, AND only at publication
"""
import json, os, pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
import numpy as np

from .grammar import TOPICS, SUBJECTS, VOCAB, IDX, WORLD, valid_transition
from .machine import Machine, Params
from .composite import RingPop, FREE, LOCKED, Composite, CompParams
from .ring import (Ring, PAIR_OR, PAIR_AND, onsets_of, propagation, winding_trace)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
SEEDS = (0, 1, 2, 3, 4)          # Stage 0 weight seeds
GAINS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
G_MAIN = 8.0
T, DEAD = 50, 18
AB_C = (4, 7, 9)                 # slots of the A, B, C words


def sentences():
    fam = [[t, s, "walks", "to", A, "with", "a", B, "and", C, "."]
           for t, (A, B, C) in TOPICS.items() for s in SUBJECTS]
    rng = np.random.default_rng(6)
    worlds = list(TOPICS)
    nov = []
    for t in worlds:
        n = 0
        while n < 4:
            x, y, z = rng.choice(worlds, 3)
            if x == y == z == t:
                continue
            s = SUBJECTS[n % 2]
            nov.append([t, s, "walks", "to", TOPICS[x][0], "with", "a", TOPICS[y][1],
                        "and", TOPICS[z][2], "."])
            n += 1
    return fam, nov


FAM, NOV = sentences()


def weights():
    return pickle.load(open(os.path.join(OUT, "weights.pkl"), "rb"))["weights"]


def make(W, comp, P=1, ps=0.0, seed=0):
    m = Machine(Params(), seed=seed)
    m.J, m.U = W[0].copy(), W[1].copy()
    return Composite(m, replace(LOCKED, P=P, p_spont=ps), comp, seed=seed)


def schema_word(sent, slot):
    """What the topic's world would put in this slot."""
    return TOPICS[sent[0]][AB_C.index(slot)] if slot in AB_C else None


# ------------------------------------------------------------------- C1
RATES = (0.0002, 0.0005, 0.002, 0.005)       # per unit per step (x400 = per second)
PS = (1, 4, 8, 16)


def _free(P, ps, seed):
    R = RingPop(replace(FREE, P=P, p_spont=ps), seed=seed)
    R.schedule_kick(0, +1)
    ons = [(t, int(k)) for t in range(3030) for k in R.step()]
    wt = winding_trace(ons, propagation(ons, 8, 8), 8, 3030, 72)
    bad = np.where(wt != 1)[0]
    return (bad[0] if len(bad) else len(wt)) / len(wt)


def _locked(P, ps, seed, n=66):
    R = RingPop(replace(LOCKED, P=P, p_spont=ps), seed=seed)
    R.schedule_kick(0, +1)
    per = {}
    for c in range(n):
        for k in range(T):
            for reg in R.step(gated=(k >= T - DEAD)):
                per.setdefault(c, []).append(int(reg))
    good = [per.get(c, []) == [c % 11] for c in range(n)]
    return (good.index(False) if False in good else n) / n


def c1():
    out = {"free": {}, "locked": {}}
    for P in PS:
        for ps in RATES:
            key = f"P={P} {ps * 400:g}/s"
            out["free"][key] = float(np.mean([_free(P, ps, s) for s in range(6)]))
            out["locked"][key] = float(np.mean([_locked(P, ps, s) for s in range(6)]))
    return out


# ------------------------------------------------------------------- C2/C3
def recall_words(W, mode, sent, direction=+1, gB=G_MAIN, seed=0):
    """Returns (published words, fraction of cycles where two words were both
    strongly active in the early slot: runner-up E > half the winner's)."""
    if mode == "chain only":
        C = make(W, CompParams(g_B=0.0), seed=seed)
        C.hear(sent)
        C.reset()
        if direction < 0:
            return None
        outs = [C.cycle(sent[0], drive=False)] + [C.cycle(None) for _ in range(10)]
    else:
        C = make(W, CompParams(g_B=gB, chain=(mode == "composite")), seed=seed)
        C.hear(sent)
        outs = C.recall(11, direction=direction, start=0 if direction > 0 else 10)
    both = float(np.mean([np.sort(o["E"])[-2] > 0.5 * o["E"].max() for o in outs]))
    return [o["published"] for o in outs], both


def _score(words, target):
    """Accuracy over slots 1..10 (slot 0 is the cue for chain only), and
    how many errors at A/B/C slots were the topic's schema word."""
    acc = np.mean([w == t for w, t in zip(words[1:], target[1:])])
    return float(acc)


def c23():
    W = weights()
    res = {"forward": {}, "backward": {}, "schema_intrusions": {}}
    for direction, key in ((+1, "forward"), (-1, "backward")):
        for mode in ("chain only", "ring only", "composite"):
            gains = GAINS if mode == "composite" else (G_MAIN,)
            if mode == "chain only" and direction < 0:
                continue
            for g in gains:
                for kind, S in (("familiar", FAM), ("novel", NOV)):
                    accs, boths = [], []
                    intr, errs = 0, 0
                    for si, sent in enumerate(S):
                        for s in SEEDS:
                            words, both = recall_words(W[s], mode, sent, direction, g, seed=s * 100 + si)
                            boths.append(both)
                            target = sent if direction > 0 else sent[::-1]
                            accs.append(_score(words, target))
                            if direction > 0 and kind == "novel":
                                for slot in AB_C:
                                    if words[slot] != sent[slot]:
                                        errs += 1
                                        intr += int(words[slot] == schema_word(sent, slot))
                    name = mode if mode != "composite" else f"composite g_B={g:g}"
                    res[key].setdefault(name, {})[kind] = float(np.mean(accs))
                    res.setdefault("two_words_active_" + key, {}).setdefault(name, {})[kind] = float(np.mean(boths))
                    if direction > 0 and kind == "novel":
                        res["schema_intrusions"][name] = dict(errors=errs, schema=intr)
    return res


# ------------------------------------------------------------------- C4
def c4():
    W = weights()
    res = {}
    for mode in ("chain only", "ring only", "composite"):
        res[mode] = {}
        for kind, S in (("familiar", FAM), ("novel", NOV)):
            hits = []
            for si, sent in enumerate(S):
                for s in SEEDS:
                    comp = CompParams(g_B=0.0 if mode == "chain only" else G_MAIN,
                                      chain=(mode != "ring only"))
                    C = make(W[s], comp, seed=s * 100 + si)
                    C.hear(sent)
                    for pos in range(1, 10):          # ask about words 1..9
                        outs = C.jump(sent[pos], n=1)
                        hits.append(outs[1]["published"] == sent[pos + 1])
            res[mode][kind] = float(np.mean(hits))
    return res


# ------------------------------------------------------------------- C5
def c5():
    W = weights()
    res = {}
    rng = np.random.default_rng(55)
    for k_missing in (0, 2, 4):
        for mode in ("ring only", "composite"):
            for kind, S in (("familiar", FAM), ("novel", NOV)):
                accs, gap_hits, gaps = [], 0, 0
                for si, sent in enumerate(S):
                    for s in SEEDS:
                        C = make(W[s], CompParams(g_B=G_MAIN, chain=(mode == "composite")),
                                 seed=s * 100 + si)
                        C.hear(sent)
                        miss = rng.choice(np.arange(1, 11), k_missing, replace=False)
                        for pos in miss:
                            C.B[pos] = 0.0              # this position was never bound
                        words = [o["published"] for o in C.recall(11)]
                        accs.append(_score(words, sent))
                        for pos in miss:
                            gaps += 1
                            gap_hits += int(words[pos] == sent[pos])
                res.setdefault(f"{k_missing} missing", {}).setdefault(mode, {})[kind] = dict(
                    accuracy=float(np.mean(accs)),
                    gaps_filled_correctly=(gap_hits / gaps) if gaps else None)
    return res


# ------------------------------------------------------------------- C6
C6_RATES = (0.0005, 0.002)


def _pairs(pub, sent):
    nxt = {sent[i]: sent[(i + 1) % len(sent)] for i in range(len(sent))}
    pairs = [(a, b) for a, b in zip(pub, pub[1:]) if a is not None and b is not None]
    if not pairs:
        return 0.0, 0.0, float(np.mean([p is None for p in pub]))
    ep = np.mean([nxt.get(a) == b for a, b in pairs])
    intr = np.mean([nxt.get(a) != b and valid_transition(a, b) for a, b in pairs])
    return float(ep), float(intr), float(np.mean([p is None for p in pub]))


def c6():
    W = weights()
    conds = {"ring only": CompParams(g_B=G_MAIN, chain=False),
             "ring only + restart": CompParams(g_B=G_MAIN, chain=False, relaunch=True),
             "composite": CompParams(g_B=G_MAIN),
             "composite + restart": CompParams(g_B=G_MAIN, relaunch=True)}
    res = {}
    for P in (1, 8):
        for ps in C6_RATES:
            key = f"P={P} {ps * 400:g}/s"
            res[key] = {}
            for name, comp in conds.items():
                r = []
                for kind, S in (("familiar", FAM), ("novel", NOV)):
                    for si, sent in enumerate(S):
                        s = SEEDS[si % len(SEEDS)]
                        C = make(W[s], comp, P=P, ps=0.0, seed=s * 100 + si)
                        C.hear(sent)                              # hearing is clean
                        C.ring.p = replace(C.ring.p, p_spont=ps)   # recall is noisy
                        pub = [o["published"] for o in C.recall(44)]
                        r.append(_pairs(pub, sent))
                r = np.array(r)
                res[key][name] = dict(episode_pairs=float(r[:, 0].mean()),
                                      plausible_intrusions=float(r[:, 1].mean()),
                                      silent=float(r[:, 2].mean()))
    return res


# ------------------------------------------------------------------- C7
C7_RATES = (0.0002, 0.0005, 0.001)


def c7():
    """Two mirrored rings, OR coupling (robust running). Publication checkpoint:
    a ring-0 burst is published only if ring 1 bursts at the same region within
    +-3 steps. The checkpoint changes nothing in the dynamics, only what is said."""
    res = {}
    for ps in C7_RATES:
        rows = {"OR, no checkpoint": [], "OR, AND checkpoint": [], "AND everywhere": []}
        for s in range(8):
            for name, P in (("OR", PAIR_OR), ("AND everywhere", PAIR_AND)):
                R = Ring(replace(P, p_spont=ps), seed=s)
                rec = R.kick(0, +1) + R.run(3000)
                o0, o1 = onsets_of(rec, 0), onsets_of(rec, 1)
                lab = dict(zip(o0, propagation(o0, 8, 8)))
                wt = winding_trace(o0, list(lab.values()), 8, R.t, 72)
                bad = np.where(wt != 1)[0]
                surv = (bad[0] if len(bad) else len(wt)) / len(wt)
                wave = [b for b in o0 if lab[b] == 1]
                if name == "OR":
                    both = [b for b in o0 if any(k1 == b[1] and abs(t1 - b[0]) <= 3 for t1, k1 in o1)]
                    for nm, pub in (("OR, no checkpoint", o0), ("OR, AND checkpoint", both)):
                        kept_wave = sum(1 for b in pub if lab[b] == 1)
                        rows[nm].append((surv, kept_wave / max(len(pub), 1),
                                         kept_wave / max(len(wave), 1)))
                else:
                    rows[name].append((surv, len(wave) / max(len(o0), 1), 1.0))
        res[f"{ps * 400:g}/s"] = {k: dict(zip(("survival", "published_that_is_wave", "wave_words_kept"),
                                           np.array(v).mean(0).tolist())) for k, v in rows.items()}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = {"C1": c1, "C23": c23, "C4": c4, "C5": c5, "C6": c6, "C7": c7}
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        res = {k: f.result() for k, f in futs.items()}
    res["sentences"] = dict(familiar=[" ".join(s) for s in FAM], novel=[" ".join(s) for s in NOV])
    res["params"] = dict(locked=asdict(LOCKED), free=asdict(FREE), g_B=G_MAIN)
    with open(os.path.join(OUT, "stage06_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("params", "sentences")}, indent=1))


if __name__ == "__main__":
    main()
