"""Stage 0.7: who speaks when? A global schema gain vs a local gate.
Run:  python -m rttw.stage07   (after rttw.experiments)

The Stage 0.6 composite runs the chain (schema) at full strength all the time.
Here the chain's content pathways (learned transitions + rebound) are scaled by

  global  : a constant lam, the same for every cycle and every task;
  gate    : 1 - (episode drive at this moment), clipped to [0, 1] -- the chain is
            muted exactly while the ring is delivering a bound word, and speaks
            freely when the ring's place holds nothing (a gap) or the ring is gone.

Apical context is left on in every condition: it changes susceptibility, not
content. Every condition restarts a silent ring from the last spoken word the
episode actually holds.

Falsifier: if some single global lam matches the gate on all four tests at once
(forward novel, backward, gap filling, noisy recitation), the gate adds nothing.
"""
import json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

from .composite import CompParams
from . import stage06 as S

LAMS = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
GBS = (4.0, 8.0)
NOISE = 0.002        # per unit per step = 0.8 / unit / s
P_NOISE = 8


def conditions(gB):
    c = {"ring only": CompParams(g_B=gB, chain=False, restart_from="bound")}
    for lam in LAMS:
        c[f"global {lam:g}"] = CompParams(g_B=gB, schema="global", lam=lam, restart_from="bound")
    c["gate"] = CompParams(g_B=gB, schema="gate", restart_from="bound")
    return c


def _two(outs):
    return float(np.mean([np.sort(o["E"])[-2] > 0.5 * o["E"].max() for o in outs]))


def evaluate(args):
    gB, name = args
    comp = conditions(gB)[name]
    W = S.weights()
    rng = np.random.default_rng(77)
    r = {k: [] for k in ("fwd_familiar", "fwd_novel", "backward", "gaps_familiar", "gaps_novel",
                         "two_words_backward")}
    for si, sent in enumerate(S.FAM + S.NOV):
        kind = "familiar" if si < len(S.FAM) else "novel"
        for s in S.SEEDS:
            C = S.make(W[s], comp, seed=s * 100 + si)
            C.hear(sent)
            f = [o["published"] for o in C.recall(11)]
            r[f"fwd_{kind}"].append(S._score(f, sent))
            bo = C.recall(11, direction=-1, start=10)
            r["backward"].append(S._score([o["published"] for o in bo], sent[::-1]))
            r["two_words_backward"].append(_two(bo))
            B0 = C.B.copy()
            for pos in rng.choice(np.arange(1, 11), 4, replace=False):
                C.B[pos] = 0.0
            g = [o["published"] for o in C.recall(11)]
            r[f"gaps_{kind}"].append(S._score(g, sent))
            C.B = B0
    nz = []
    for si, sent in enumerate(S.FAM + S.NOV):
        for s in S.SEEDS[:2]:
            C = S.make(W[s], replace(comp, relaunch=True), P=P_NOISE, seed=s * 100 + si)
            C.hear(sent)
            C.ring.p = replace(C.ring.p, p_spont=NOISE)
            nz.append(S._pairs([o["published"] for o in C.recall(44)], sent))
    nz = np.array(nz)
    out = {k: float(np.mean(v)) for k, v in r.items()}
    out.update(noise_episode_pairs=float(nz[:, 0].mean()), noise_plausible=float(nz[:, 1].mean()),
               noise_silent=float(nz[:, 2].mean()))
    out["worst_of_four"] = min(out["fwd_novel"], out["backward"],
                               0.5 * (out["gaps_familiar"] + out["gaps_novel"]),
                               out["noise_episode_pairs"])
    return gB, name, out


def main():
    jobs = [(gB, name) for gB in GBS for name in conditions(gB)]
    with ProcessPoolExecutor(os.cpu_count() or 1) as ex:
        rows = list(ex.map(evaluate, jobs))
    res = {}
    for gB, name, out in rows:
        res.setdefault(f"g_B={gB:g}", {})[name] = out
    res["setup"] = dict(noise_per_unit_per_s=NOISE * 400, units_per_place=P_NOISE, lams=LAMS,
                        gaps=4, recitations=4)
    with open(os.path.join(S.OUT, "stage07_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        if k == "setup":
            continue
        print(k)
        for name, o in v.items():
            print(f"  {name:12s}", " ".join(f"{a} {b:.2f}" for a, b in o.items()))


if __name__ == "__main__":
    main()
