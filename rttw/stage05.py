"""Stage 0.5 experiments: the route door (ring) and the coincidence door
(two mirrored rings).  Run:  python -m rttw.stage05   (from the repo root)

R1  band        refractory time x ring size -> split / one-way / dies
R2  direction   same stored words, same weights, read forward or backward
R3  operations  "reverse from here" and "start at a word" on a running wave
R4  noise       spurious ignitions on one ring: does it die, or reverse?
R5  coincidence two mirrored rings, OR vs AND coupling, private vs shared noise
R6  lag         two rings started out of step: do they lock to zero lag?
"""
import json, os, pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
import numpy as np

from .ring import (Ring, RingParams, SINGLE, PAIR_OR, PAIR_AND, onsets_of,
                   propagation, winding_trace, lap_time)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
WORDS = "winter girl walks to hill with a sled".split()
SEEDS = range(8)
HOP = 9          # measured steps per hop (lap 72 / 8 regions), used for lap windows


def lap_steps(P):
    return HOP * P.N


def classify(P, seed, steps=3000):
    R = Ring(P, seed=seed)
    rec = R.kick(0, +1) + R.run(steps)
    ons = onsets_of(rec, 0)
    lab = propagation(ons, P.N, P.d)
    wt = winding_trace(ons, lab, P.N, R.t, lap_steps(P))
    if np.mean(wt == 1) > 0.9:
        return "one-way"
    if len(ons) < 3 * P.N:
        return "dies"
    return "split"


def refractory(P):
    """Steps after a burst before a one-neighbour-sized input (6 steps) can
    start a second burst. Neighbours disconnected, no noise."""
    for D in range(6, 400):
        R = Ring(replace(P, noise=0.0, w_x=0.0, w_c=0.0), seed=0)
        tr = []
        for s in range(D + 30):
            e = np.zeros((P.H, P.N))
            if s < 5:
                e[:, 0] = 2.0
            if D <= s < D + 6:
                e[:, 0] = 1.0
            R.step(ext=e)
            tr.append(R.r[0, 0])
        tr = np.array(tr)
        fell = np.where(tr < 0.2)[0]
        fell = fell[fell > 3]
        if len(fell) and (tr[fell[0]:] > 0.5).any():
            return D
    return None


# --------------------------------------------------------------------- R1
TAUS = (15, 20, 22, 25, 30, 35, 40, 45, 50, 60, 70, 80)
NS = (6, 8, 12, 16)


def r1_band():
    grid = {}
    for N in NS:
        for tp in TAUS:
            outs = [classify(replace(SINGLE, N=N, tau_p=tp), s) for s in range(3)]
            grid[f"N={N} tau_p={tp}"] = {o: outs.count(o) / 3 for o in ("split", "one-way", "dies")}
    R = {str(tp): refractory(replace(SINGLE, tau_p=tp)) for tp in TAUS}
    laps = {}
    for N in NS:
        Rg = Ring(replace(SINGLE, N=N, tau_p=25), seed=0)
        laps[str(N)] = lap_time(onsets_of(Rg.kick(0, +1) + Rg.run(1500), 0), N)
    return dict(grid=grid, refractory=R, lap=laps)


# --------------------------------------------------------------------- R2
def r2_direction():
    out = {}
    for d in (+1, -1):
        texts, ok = [], []
        for s in SEEDS:
            R = Ring(SINGLE, seed=s, words=WORDS)
            ons = onsets_of(R.kick(0, d) + R.run(1500), 0)
            seq = [k for _, k in ons]
            expect = [(d * i) % 8 for i in range(len(seq))]
            ok.append(float(np.mean(np.array(seq) == np.array(expect))))
            texts.append(" ".join(WORDS[k] for k in seq[:16]))
        out["forward" if d > 0 else "backward"] = dict(match=float(np.mean(ok)), example=texts[0])
    # the Stage 0 chain for comparison: learned transitions in the backward direction
    try:
        W = pickle.load(open(os.path.join(OUT, "weights.pkl"), "rb"))["weights"]
        from .grammar import IDX
        story = "winter girl walks to hill with a sled and slides .".split()
        fw, bw = [], []
        for J, _ in W.values():
            fw += [J[IDX[a], IDX[b]] for a, b in zip(story, story[1:])]
            bw += [J[IDX[b], IDX[a]] for a, b in zip(story, story[1:])]
        out["stage0_chain"] = dict(mean_forward_J=float(np.mean(fw)), max_backward_J=float(np.max(bw)))
    except FileNotFoundError:
        pass
    return out


# --------------------------------------------------------------------- R3
def _op_trial(seed, op):
    rng = np.random.default_rng(100 + seed)
    R = Ring(SINGLE, seed=seed, words=WORDS)
    rec = R.kick(0, +1) + R.run(int(rng.integers(200, 500)))
    ons = onsets_of(rec, 0)
    here = ons[-1][1]
    rec += R.silence(20)
    if op == "reverse from here":
        start, d = here, -1
    else:
        start, d = int(rng.integers(0, 8)), +1
    rec2 = R.kick(start, d) + R.run(700)
    seq = [k for _, k in onsets_of(rec2, 0)][:16]
    expect = [(start + d * i) % 8 for i in range(len(seq))]
    return float(np.mean(np.array(seq) == np.array(expect))) if seq else 0.0


def r3_operations():
    return {op: float(np.mean([_op_trial(s, op) for s in SEEDS]))
            for op in ("reverse from here", "start at a word")}


# --------------------------------------------------------------------- R4/R5
RATES = (0.0002, 0.0005, 0.001, 0.002)       # per region per step


def noise_trial(P, seed, steps=3000):
    R = Ring(P, seed=seed)
    rec = R.kick(0, +1) + R.run(steps)
    ons = onsets_of(rec, 0)
    lab = propagation(ons, P.N, P.d)
    wt = winding_trace(ons, lab, P.N, R.t, lap_steps(P))
    bad = np.where(wt != 1)[0]
    return dict(survival=(bad[0] if len(bad) else len(wt)) / len(wt),
                clean=float(np.mean([l == 1 for l in lab])) if lab else 0.0,
                reversed=float(np.mean(wt == -1)),
                dead=float(np.mean(wt == 0)))


def r45_noise():
    conds = {"single ring": (SINGLE, False),
             "OR pair, private": (PAIR_OR, False), "OR pair, shared": (PAIR_OR, True),
             "AND pair, private": (PAIR_AND, False), "AND pair, shared": (PAIR_AND, True)}
    res = {}
    for name, (P, shared) in conds.items():
        res[name] = {}
        for ps in RATES:
            trials = [noise_trial(replace(P, p_spont=ps, spont_shared=shared), s) for s in SEEDS]
            res[name][f"{ps * 400:.2f}/s"] = {k: float(np.mean([t[k] for t in trials])) for k in trials[0]}
    return res


# --------------------------------------------------------------------- R6
LAGS = (0, 2, 4, 6, 8, 10, 12, 16)


def r6_lag():
    res = {}
    for name, P in (("OR pair", PAIR_OR), ("AND pair", PAIR_AND)):
        res[name] = {}
        for lag in LAGS:
            surv, late = [], []
            for s in SEEDS:
                R = Ring(P, seed=s)
                rec = R.kick(0, +1, lag=lag) + R.run(1500)
                o0, o1 = onsets_of(rec, 0), onsets_of(rec, 1)
                lab = propagation(o0, P.N, P.d)
                wt = winding_trace(o0, lab, P.N, R.t, lap_steps(P))
                surv.append(float(np.mean(wt == 1)))
                L = [min((t1 - t0 for t0, k0 in o0 if k0 == k1 and abs(t1 - t0) < 30), key=abs, default=np.nan)
                     for t1, k1 in o1 if t1 > 1000]
                late.append(np.nanmean(L) if len(L) else np.nan)
            res[name][str(lag)] = dict(alive=float(np.mean(surv)),
                                       final_lag_steps=float(np.nanmean(late)) if not np.all(np.isnan(late)) else None)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = {"R1": r1_band, "R2": r2_direction, "R3": r3_operations,
            "R45": r45_noise, "R6": r6_lag}
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        res = {k: f.result() for k, f in futs.items()}
    res["params"] = dict(single=asdict(SINGLE), pair_or=asdict(PAIR_OR), pair_and=asdict(PAIR_AND))
    with open(os.path.join(OUT, "stage05_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
