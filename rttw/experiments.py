"""Stage 0 experiments. Run:  python -m rttw.experiments  (from the repo root)

Every test uses weights learned by local rules in reading mode (train()).
Several seeds are trained; every number is reported per seed and pooled.
"""
import json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
import numpy as np

from .grammar import (story, VOCAB, IDX, SLOT, TOPICS, WORLD, valid_transition,
                      successors, CONTEXT_BRANCH_SLOTS)
from .machine import Machine, Params

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = (0, 1, 2, 3, 4)
N_TRAIN = 1200


# ----------------------------------------------------------------- training
def train(seed, n=N_TRAIN, params=None):
    rng = np.random.default_rng(1000 + seed)
    m = Machine(params or Params(), seed=seed)
    curve = []
    for s in range(n):
        outs = m.read(story(rng), learn=True)
        hits = [o["pred_prev"] == o["word"] for o in outs
                if SLOT[o["word"]] in CONTEXT_BRANCH_SLOTS]
        det = [o["pred_prev"] == o["word"] for o in outs
               if SLOT[o["word"]] in (2, 3, 5, 6, 8, 10)]
        curve.append((np.mean(hits), np.mean(det)))
    return m, np.array(curve)


def fresh(weights, seed=0, **kw):
    m = Machine(replace(Params(), **kw), seed=seed)
    m.J, m.U = weights[0].copy(), weights[1].copy()
    return m


def warm(m, rng, n=1):
    """Read n random stories so context and carry are in a running state."""
    for _ in range(n):
        m.read(story(rng))


# ----------------------------------------------------------------- T0 learning
def t0_learning(weights, seed):
    rng = np.random.default_rng(2000 + seed)
    m = fresh(weights, seed)
    kinds = {"deterministic": (2, 3, 5, 6, 8, 10), "context branch": (4, 7, 9),
             "random branch": (0, 1)}
    acc = {k: [] for k in kinds}
    veto_at = {"random branch": [], "context branch": []}
    warm(m, rng)
    for _ in range(90):
        for o in m.read(story(rng)):
            for k, slots in kinds.items():
                if SLOT[o["word"]] in slots:
                    acc[k].append(o["pred_prev"] == o["word"])
                    if k in veto_at and o["mismatch"]:
                        veto_at[k].append(bool(o["vetoed"]))
    # cold start: reset state before each story (context only from this story)
    cold = []
    for i in range(90):
        m.reset_state()
        for o in m.read(story(rng, topic=list(TOPICS)[i % 3])):
            if SLOT[o["word"]] in CONTEXT_BRANCH_SLOTS:
                cold.append(o["pred_prev"] == o["word"])
    # no-context control: same weights, apical gain 0
    m0 = fresh(weights, seed, g_ap=0.0)
    warm(m0, rng)
    noctx = []
    for _ in range(90):
        for o in m0.read(story(rng)):
            if SLOT[o["word"]] in CONTEXT_BRANCH_SLOTS:
                noctx.append(o["pred_prev"] == o["word"])
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    # how often a mispredicted branch recruits an SST veto (> 0.2 on some route)
    for k, v in veto_at.items():
        res[f"SST veto after a miss at a {k}"] = float(np.mean(v)) if v else float("nan")
    res["context branch, cold start"] = float(np.mean(cold))
    res["context branch, apical off"] = float(np.mean(noctx))
    return res


# ----------------------------------------------------------------- free-run scoring
def score_run(words, cue_last, world):
    """words: published words (None = nothing published). Returns validity of
    consecutive transitions, whether the first story stayed in `world`, and
    whether the run stayed alive."""
    seq = [cue_last] + list(words)
    pairs = [(a, b) for a, b in zip(seq, seq[1:]) if a is not None and b is not None]
    valid = np.mean([valid_transition(a, b) for a, b in pairs]) if pairs else 0.0
    alive = all(w is not None for w in words)
    first = []
    for w in words:
        if w is None or w == ".":
            break
        if SLOT[w] in CONTEXT_BRANCH_SLOTS:
            first.append(WORLD[w] == world)
    consistent = float(np.mean(first)) if first else 0.0
    return float(valid), consistent, alive


# ----------------------------------------------------------------- T2 eyes closed
def t2_eyes_closed(weights, seed, n_trials=15, n_free=33):
    rng = np.random.default_rng(3000 + seed)
    out = {}
    for cue_len, label in ((1, "topic only"), (4, "... to"), (7, "... with a")):
        V_, C_, A_ = [], [], []
        for i in range(n_trials):
            topic = list(TOPICS)[i % 3]
            m = fresh(weights, seed * 100 + i)
            warm(m, rng)
            cue = story(rng, topic=topic)[:cue_len]
            m.read(cue)
            outs = m.free_run(n_free)
            v, c, a = score_run([o["published"] for o in outs], cue[-1], topic)
            V_.append(v); C_.append(c); A_.append(a)
        out[label] = dict(valid=float(np.mean(V_)), world_consistent=float(np.mean(C_)),
                          alive=float(np.mean(A_)))
    return out


# ----------------------------------------------------------------- T3 dead time
DEADS = (18, 12, 6, 0)
SPONTS = (0.0, 0.0002, 0.0005, 0.001)   # x1000 = ignitions per cycle


def t3_dead_time(weights, seed, n_trials=12, n_free=33):
    rng = np.random.default_rng(4000 + seed)
    res = {}
    for ps in SPONTS:
        for dead in DEADS:
            V_, M_, A_ = [], [], []
            for i in range(n_trials):
                topic = list(TOPICS)[i % 3]
                m = fresh(weights, seed * 100 + i)
                warm(m, rng)
                cue = story(rng, topic=topic)[:4]
                m.read(cue)
                m.p = replace(m.p, dead=dead, p_spont=ps)
                outs = m.free_run(n_free)
                v, c, a = score_run([o["published"] for o in outs], cue[-1], topic)
                multi = np.mean([int((o["E"] > m.p.thr_E).sum() > 1) for o in outs])
                V_.append(v); M_.append(multi); A_.append(a)
            res[f"spont={ps} dead={dead}"] = dict(valid=float(np.mean(V_)),
                                                  multi_current=float(np.mean(M_)),
                                                  alive=float(np.mean(A_)))
    return res


# ----------------------------------------------------------------- T4 additive vs multiplicative
MULT_GAINS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
ADD_GAINS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)


def _branch_and_intrusion(m, rng, n_stories=30):
    warm(m, rng)
    hits, intr, cyc = [], 0, 0
    for _ in range(n_stories):
        for o in m.read(story(rng), record=True):
            if SLOT[o["word"]] in CONTEXT_BRANCH_SLOTS:
                hits.append(o["pred_prev"] == o["word"])
            fired = o["r"][: m.p.win_end].max(0) > 0.5
            unsupported = o["basal_max"] < m.p.theta
            intr += int((fired & unsupported).sum())
            cyc += 1
    return float(np.mean(hits)), intr / cyc


def t4_apical_rule(weights, seed):
    rng = np.random.default_rng(5000 + seed)
    res = {"mult": {}, "add": {}}
    for g in MULT_GAINS:
        m = fresh(weights, seed, mult=True, g_ap=g)
        acc, intr = _branch_and_intrusion(m, rng)
        m = fresh(weights, seed, mult=True, g_ap=g)
        fr = _free_validity(m, rng)
        res["mult"][str(g)] = dict(branch_acc=acc, intrusions_per_cycle=intr, free_valid=fr)
    for g in ADD_GAINS:
        m = fresh(weights, seed, mult=False, g_add=g)
        acc, intr = _branch_and_intrusion(m, rng)
        m = fresh(weights, seed, mult=False, g_add=g)
        fr = _free_validity(m, rng)
        res["add"][str(g)] = dict(branch_acc=acc, intrusions_per_cycle=intr, free_valid=fr)
    return res


def _free_validity(m, rng, n_trials=6, n_free=22):
    V_ = []
    for i in range(n_trials):
        topic = list(TOPICS)[i % 3]
        m.reset_state()
        warm(m, rng)
        cue = story(rng, topic=topic)[:4]
        m.read(cue)
        outs = m.free_run(n_free)
        v, c, a = score_run([o["published"] for o in outs], cue[-1], topic)
        V_.append(v * c)   # valid AND in the cued world
    return float(np.mean(V_))


# ----------------------------------------------------------------- T5 silent vs freeze
def t5_silent(weights, seed, n_trials=12, i0=2, ks=(1, 2, 3, 4)):
    rng = np.random.default_rng(6000 + seed)
    res = {}
    conds = ("AIS closed", "freeze (trace decays)", "freeze (trace held)")
    tally = {(c, k): {"on time": 0, "stale": 0, "silent/died": 0, "other": 0}
             for c in conds for k in ks}
    for i in range(n_trials):
        topic = list(TOPICS)[i % 3]
        m = fresh(weights, seed * 100 + i)
        warm(m, rng)
        m.read(story(rng, topic=topic)[:4])
        snap = m.snapshot()
        ref = [o["published"] for o in m.free_run(i0 + max(ks) + 3)]
        for k in ks:
            closed = set(range(i0, i0 + k))
            for c in conds:
                m.restore(snap)
                if c == "AIS closed":
                    outs = m.free_run(i0 + k + 1, ais=closed)
                else:
                    m.p = replace(m.p, carry_decay=0.5 if "decays" in c else 1.0)
                    outs = m.free_run(i0 + k + 1, freeze=closed)
                    m.p = replace(m.p, carry_decay=0.5)
                got = outs[i0 + k]["published"]
                if got is None:
                    key = "silent/died"
                elif got == ref[i0 + k]:
                    key = "on time"
                elif got == ref[i0]:
                    key = "stale"
                else:
                    key = "other"
                tally[(c, k)][key] += 1
    for (c, k), d in tally.items():
        res[f"{c} | k={k}"] = {kk: v / n_trials for kk, v in d.items()}
    return res


# ----------------------------------------------------------------- T6 "no, not that"
T6_CONDS = {
    "accept (no 'no')": None,
    "backup only": dict(route=False, item=False),
    "backup + item": dict(route=False, item=True),
    "backup + route (SST)": dict(route=True, item=False),
    "backup + route + item": dict(route=True, item=True),
}


TOPIC_GAINS = (1.0, 2.0, 3.0, 4.0)


def t6_no_not_that(weights, seed, n_trials=15, n_after=9):
    return {f"topic_gain={tg}": _t6(weights, seed, tg, n_trials, n_after)
            for tg in (1.0, 3.0)}


def _t6(weights, seed, topic_gain, n_trials, n_after):
    rng = np.random.default_rng(7000 + seed)
    res = {c: dict(avoided=0, retry_valid=0, downstream_consistent=0, finished=0)
           for c in T6_CONDS}
    for i in range(n_trials):
        topic = list(TOPICS)[i % 3]
        m = fresh(weights, seed * 100 + i, topic_gain=topic_gain)
        warm(m, rng)
        m.read(story(rng, topic=topic)[:4])        # ... walks to
        first = m.cycle(None)                       # eyes closed: first branch word
        X = first["published"]
        snap = m.snapshot()
        for c, kw in T6_CONDS.items():
            m.restore(snap)
            if kw is not None:
                m.reject(X, base="to", **kw)
            outs = m.free_run(n_after)
            words = [o["published"] for o in outs]
            if kw is not None:
                # expect: 'to' again, then a new A word
                retry = words[1] if len(words) > 1 else None
                seq_after = words[1:]
                ok_retry = retry is not None and SLOT[retry] == 4
                res[c]["retry_valid"] += int(ok_retry)
                res[c]["avoided"] += int(ok_retry and retry != X)
                A = retry
            else:
                seq_after = [X] + words
                A = X
                res[c]["retry_valid"] += 1
            # downstream B, C must follow the world of the A word actually used
            bc = [w for w in seq_after[1:] if w is not None and SLOT[w] in (7, 9)][:2]
            if A is not None and SLOT.get(A) == 4 and len(bc) == 2:
                res[c]["downstream_consistent"] += int(all(WORLD[w] == WORLD[A] for w in bc))
            res[c]["finished"] += int("." in [w for w in seq_after if w])
    return {c: {k: v / n_trials for k, v in d.items()} for c, d in res.items()}


# ----------------------------------------------------------------- T6b reading garden path
def t6b_garden_path(weights, seed, n_trials=18):
    """Read a story whose A word contradicts its topic (winter ... to SEA ...).
    Does the prediction at B and C follow the A word (the world rule) or the
    topic? Swept over how strongly the topic was written into context."""
    return {f"topic_gain={tg}": _t6b(weights, seed, tg, n_trials) for tg in TOPIC_GAINS}


def _t6b(weights, seed, topic_gain, n_trials):
    rng = np.random.default_rng(8000 + seed)
    res = {}
    for sst in (True, False):
        hitB, hitC = [], []
        for i in range(n_trials):
            topic = list(TOPICS)[i % 3]
            other = [t for t in TOPICS if t != topic][i % 2]
            A = TOPICS[other][0]
            m = fresh(weights, seed * 100 + i, sst=sst, topic_gain=topic_gain)
            warm(m, rng)
            st = story(rng, topic=topic, A_override=A)
            outs = m.read(st)
            # outs[k]['pred'] is the prediction made during word k for word k+1
            hitB.append(outs[6]["pred"] == st[7])
            hitC.append(outs[8]["pred"] == st[9])
        res["SST on" if sst else "SST off"] = dict(B_follows_A=float(np.mean(hitB)),
                                                   C_follows_A=float(np.mean(hitC)))
    return res


# ----------------------------------------------------------------- driver
def run_seed(seed):
    t0 = time.time()
    m, curve = train(seed)
    W = (m.J.copy(), m.U.copy())
    r = dict(seed=seed)
    r["T0"] = t0_learning(W, seed)
    r["T2"] = t2_eyes_closed(W, seed)
    r["T3"] = t3_dead_time(W, seed)
    r["T4"] = t4_apical_rule(W, seed)
    r["T5"] = t5_silent(W, seed)
    r["T6"] = t6_no_not_that(W, seed)
    r["T6b"] = t6b_garden_path(W, seed)
    r["secs"] = time.time() - t0
    return r, W, curve


def pool(results):
    """Mean and min/max across seeds for every numeric leaf."""
    def walk(nodes):
        if isinstance(nodes[0], dict):
            return {k: walk([n[k] for n in nodes]) for k in nodes[0]}
        a = np.array(nodes, dtype=float)
        return dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))
    keys = [k for k in results[0] if k.startswith("T")]
    return {k: walk([r[k] for r in results]) for k in keys}


def main():
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(min(len(SEEDS), os.cpu_count() or 1)) as ex:
        runs = list(ex.map(run_seed, SEEDS))
    results = [r for r, _, _ in runs]
    weights = {r["seed"]: W for r, W, _ in runs}
    curves = {r["seed"]: c for r, _, c in runs}
    with open(os.path.join(OUT, "weights.pkl"), "wb") as f:
        pickle.dump(dict(weights=weights, curves=curves, params=asdict(Params())), f)
    summary = dict(params=asdict(Params()), seeds=list(SEEDS), n_train=N_TRAIN,
                   per_seed=results, pooled=pool(results))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["pooled"], indent=1))


if __name__ == "__main__":
    main()
