"""Stage 0.5 route door: a rotating wave on a ring of regions.

N regions sit on a ring (like the somatotopic strip a cortical spiral sweeps
around). Optionally there are H = 2 mirrored rings ("hemispheres") coupled
topographically, region k to region k. Each region is one rate unit:

    I_hk  = w_x (r_{h,k-1} + r_{h,k+1})(t-d)                 own ring, both sides
          + w_c  sum_{h' != h} (r_{h',k-1} + r_{h',k+1})(t-d)  the other ring
          + w_self r_hk + ext + spontaneous ignitions
    u_hk  = G relu(I_hk - theta) - w_p p_hk + noise
    r_hk <- r_hk + (clip(u_hk, 0, 1) - r_hk) / tau_r
    p_hk <- p_hk exp(-1/tau_p) + g_p r_hk          (local PV recruited by own firing)

Wiring is SYMMETRIC: every region excites both neighbours equally. There is no
global clock. A region's window opens when a neighbour's burst arrives; its
dead time is triggered by its own burst and lasts ~tau_p. The dead time is what
makes a symmetric wire one-way.

Content is local: region k stores word k and publishes it when it bursts. The
published sequence is the order in which the wave visits the regions; its
direction is a state of the dynamics, not of the weights.

Coincidence ("co-ripple") version: H = 2 and theta between one and two sources,
so a region fires only when the wave arrives from BOTH rings within the same
burst. A region that ignites on its own in one ring cannot recruit anything.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class RingParams:
    N: int = 8
    H: int = 1             # number of mirrored rings
    d: int = 8             # hop delay (steps of 2.5 ms)
    w_x: float = 1.0       # own-ring neighbour weight
    w_c: float = 0.0       # other-ring neighbour weight
    w_self: float = 1.0    # local recurrent excitation: bursts are all-or-none
    theta: float = 0.3
    G: float = 3.0
    tau_r: float = 2.0
    tau_p: float = 30.0    # refractory time constant (the ablation knob)
    g_p: float = 0.25
    w_p: float = 3.0
    noise: float = 0.02
    p_spont: float = 0.0   # per region per step
    spont_shared: bool = False  # True: an ignition hits region k in every ring at once
    spont_I: float = 2.0
    spont_len: int = 4
    onset: float = 0.5     # r crossing this upward = a burst = a published word


# the two regimes used in the experiments
SINGLE = RingParams()
# same total drive as SINGLE when both rings carry the wave (0.5 + 0.5)
PAIR_OR = replace(RingParams(), H=2, w_x=0.5, w_c=0.5)                     # one ring is enough (0.5 > theta)
PAIR_AND = replace(RingParams(), H=2, w_x=0.5, w_c=0.5, theta=0.6, w_self=2.0)   # needs both (0.5 < theta < 1)


class Ring:
    def __init__(self, params=None, seed=0, words=None):
        self.p = params or RingParams()
        self.rng = np.random.default_rng(seed)
        self.words = list(words) if words is not None else [str(k) for k in range(self.p.N)]
        self.reset()

    def reset(self):
        p = self.p
        shape = (p.H, p.N)
        self.r = np.zeros(shape)
        self.pv = np.zeros(shape)
        self.hist = np.zeros((p.d,) + shape)
        self.hptr = 0
        self.spont = np.zeros(shape, dtype=int)
        self.t = 0
        self.spont_log = []

    def step(self, ext=None, block=None):
        """ext, block: arrays (H, N) of extra input / extra inhibition.
        Returns list of (h, k) bursts that started this step."""
        p = self.p
        rd = self.hist[self.hptr]
        nb = np.roll(rd, 1, axis=1) + np.roll(rd, -1, axis=1)
        I = p.w_x * nb + p.w_self * self.r
        if p.H > 1:
            I = I + p.w_c * (nb.sum(axis=0, keepdims=True) - nb)
        if ext is not None:
            I = I + ext
        if p.p_spont > 0:
            if p.spont_shared:
                new = np.broadcast_to(self.rng.random(p.N) < p.p_spont, (p.H, p.N))
            else:
                new = self.rng.random((p.H, p.N)) < p.p_spont
            for h, k in zip(*np.where(new)):
                self.spont_log.append((self.t, int(h), int(k)))
            self.spont[new] = p.spont_len
            I = I + p.spont_I * (self.spont > 0)
            self.spont = np.maximum(self.spont - 1, 0)
        u = p.G * np.maximum(I - p.theta, 0.0) - p.w_p * self.pv
        if block is not None:
            u = u - block
        if p.noise > 0:
            u = u + p.noise * self.rng.standard_normal(u.shape)
        r_old = self.r.copy()
        self.r += (np.clip(u, 0.0, 1.0) - self.r) / p.tau_r
        self.pv = self.pv * np.exp(-1.0 / p.tau_p) + p.g_p * self.r
        self.hist[self.hptr] = self.r
        self.hptr = (self.hptr + 1) % p.d
        self.t += 1
        return [(int(h), int(k)) for h, k in zip(*np.where((r_old < p.onset) & (self.r >= p.onset)))]

    # ---------------------------------------------------------------- operations
    def kick(self, k, direction, rings=None, lag=0, amp=2.0, steps=5, block_amp=8.0):
        """Start a wave at region k going `direction` (+1/-1): drive k while
        blocking the neighbour on the other side (a one-sided block), in the
        given rings (default all). lag: ring 1 is kicked `lag` steps after ring 0."""
        p = self.p
        rings = list(range(p.H)) if rings is None else rings
        back = (k - direction) % p.N
        start = {h: (lag if h == 1 else 0) for h in rings}
        total = max(start.values()) + steps + p.d + 20
        rec = []
        for s in range(total):
            ext = np.zeros((p.H, p.N))
            blk = np.zeros((p.H, p.N))
            for h in rings:
                if start[h] <= s < start[h] + steps:
                    ext[h, k] = amp
                if s >= start[h]:
                    blk[h, back] = block_amp
            rec.append((self.t, self.step(ext=ext, block=blk), self.r.copy()))
        return rec

    def silence(self, steps, amp=8.0):
        """Global inhibition (stop everything)."""
        return [(self.t, self.step(block=np.full((self.p.H, self.p.N), amp)), self.r.copy())
                for _ in range(steps)]

    def run(self, steps):
        return [(self.t, self.step(), self.r.copy()) for _ in range(steps)]


# -------------------------------------------------------------------- analysis
def onsets_of(rec, ring=0):
    """Time-ordered list of (t, region) bursts in one ring."""
    return [(t, k) for t, ons, _ in rec for h, k in ons if h == ring]


def propagation(onsets, N, d, lo=-4, hi=8):
    """Label each burst by where it came from: +1 if region k-1 burst between
    d+lo and d+hi steps earlier (the wave arrived going +), -1 if region k+1
    did (going -), 0 if neither (a source: kick or spontaneous), 2 if both
    (a head-on collision)."""
    times = {}
    for t, k in onsets:
        times.setdefault(k, []).append(t)

    def came(t, j):
        return any(t - (d + hi) <= s <= t - (d + lo) for s in times.get(j % N, []))
    labels = []
    for t, k in onsets:
        a, b = came(t, k - 1), came(t, k + 1)
        labels.append(2 if (a and b) else (+1 if a else (-1 if b else 0)))
    return labels


def winding_trace(onsets, labels, N, T, lap):
    """Net winding per lap-long window: (# +1 bursts - # -1 bursts) / N,
    rounded. +1 = one wave going +, 0 = dead or balanced, -1 = one going -."""
    out = []
    for w0 in range(0, T - lap + 1, lap):
        s = sum(1 if l == 1 else (-1 if l == -1 else 0)
                for (t, _), l in zip(onsets, labels) if w0 <= t < w0 + lap)
        out.append(int(np.round(s / N)))
    return np.array(out)


def lap_time(onsets, N):
    """Median time for the wave to return to the same region."""
    per = {}
    for t, k in onsets:
        per.setdefault(k, []).append(t)
    gaps = [b - a for ts in per.values() for a, b in zip(ts, ts[1:])]
    return float(np.median(gaps)) if gaps else float("nan")
