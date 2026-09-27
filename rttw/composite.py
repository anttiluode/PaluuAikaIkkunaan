"""Stage 0.6: the composite. A ring of regions (place order) bound to the Stage 0
chain (learned order), sharing one rhythm.

RingPop — the Stage 0.5 ring with P units per region:

    m_k      = mean_i r_ki                                  region activity
    s_k     <- s_k + ((m_{k-1} + m_{k+1})(t-d) - s_k) / tau_s     synaptic trace
    I_ki     = w_x s_k + w_self m_k + ext_k + spont_ki
    u_ki     = G relu(I_ki - theta) - w_p p_k - PV_gate + noise
    p_k     <- p_k exp(-1/tau_p) + g_p m_k                  region PV (shared by its units)
    q_k     <- q_k exp(-1/tau_slow) + g_slow [end of a region burst]   slow refractoriness
    (u also subtracts w_p q_k; q is triggered only by a whole-region burst, so a
     few stray units do not build it up)

One unit igniting on its own moves its region's mean by 1/P. With P = 1 this is
exactly the Stage 0.5 ring.

Composite — the ring runs inside the chain's cycles:
  * the chain's PV dead time also gates the ring. A neighbour's burst arrives in
    the dead time and is held by a slow synaptic trace (tau_s) until the window
    opens, so one hop = one cycle = one word (the ring's version of the chain's
    rebound);
  * binding B[k, w]: one-shot, while hearing a sentence once, region k is bound to
    the word that was current while k was active;
  * during recall, active region k drives its bound word's BASAL input (g_B);
    the chain's own rebound, transitions and apical context act as in Stage 0;
  * word -> region (B transposed) is used by two designed operations: jump in at
    a word, and relaunch the ring from the word just spoken when the ring is silent.
"""
from dataclasses import dataclass, replace, field
import numpy as np

from .grammar import VOCAB, IDX, V


@dataclass
class RingPopParams:
    N: int = 8
    P: int = 1
    d: int = 8
    w_x: float = 1.0
    w_self: float = 1.0
    theta: float = 0.3
    G: float = 3.0
    tau_r: float = 2.0
    tau_p: float = 30.0
    g_p: float = 0.25
    w_p: float = 3.0
    noise: float = 0.02
    p_spont: float = 0.0     # per UNIT per step
    spont_I: float = 2.0
    spont_len: int = 4
    onset: float = 0.5
    pv_gate: float = 20.0    # inhibition applied when the shared rhythm is in dead time
    tau_s: float = 1.0       # synaptic decay of neighbour input (1 = none, as in Stage 0.5)
    g_slow: float = 0.0      # slow refractoriness triggered when a REGION burst ends,
    tau_slow: float = 150.0  # e.g. a burst-triggered afterhyperpolarization; 0 = off (Stage 0.5)
                             # units barely recruit it (1 = as in Stage 0.5)


# Stage 0.5 ring, now with populations (free-running, no shared rhythm)
FREE = RingPopParams()
# ring locked to the chain's 50-step cycle: one hop per cycle
LOCKED = replace(RingPopParams(), N=11, d=40, tau_s=10.0, w_x=3.5, tau_p=30.0,
                 g_slow=1.5, tau_slow=180.0)


class RingPop:
    def __init__(self, params=None, seed=0):
        self.p = params or RingPopParams()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        p = self.p
        self.r = np.zeros((p.N, p.P))
        self.pv = np.zeros(p.N)
        self.slow = np.zeros(p.N)
        self.hist = np.zeros((p.d, p.N))
        self.syn = np.zeros(p.N)
        self.hptr = 0
        self.spont = np.zeros((p.N, p.P), dtype=int)
        self.pending = []           # scheduled (steps_left, ext(N), block(N))
        self.t = 0
        self.n_spont = 0

    def m(self):
        return self.r.mean(axis=1)

    def schedule_kick(self, k, direction, amp=2.0, steps=5, block_amp=8.0, block_steps=None):
        """Drive region k for `steps` while blocking its neighbour on the other
        side (one-sided block): the wave leaves k going `direction`."""
        p = self.p
        block_steps = block_steps if block_steps is not None else p.d + 20
        back = (k - direction) % p.N
        for s in range(max(steps, block_steps)):
            ext = np.zeros(p.N)
            blk = np.zeros(p.N)
            if s < steps:
                ext[k] = amp
            if s < block_steps:
                blk[back] = block_amp
            if s < len(self.pending):
                e0, b0 = self.pending[s]
                self.pending[s] = (e0 + ext, b0 + blk)
            else:
                self.pending.append((ext, blk))

    def step(self, gated=False, ext=None, block=None):
        """One step. Returns list of regions whose burst started this step."""
        p = self.p
        m_old = self.m()
        md = self.hist[self.hptr]
        x = np.roll(md, 1) + np.roll(md, -1)
        self.syn += (x - self.syn) / p.tau_s
        I_reg = p.w_x * self.syn + p.w_self * m_old
        blk_reg = np.zeros(p.N)
        if self.pending:
            e, b = self.pending.pop(0)
            I_reg = I_reg + e
            blk_reg = blk_reg + b
        if ext is not None:
            I_reg = I_reg + ext
        if block is not None:
            blk_reg = blk_reg + block
        I = np.repeat(I_reg[:, None], p.P, axis=1)
        if p.p_spont > 0:
            new = self.rng.random((p.N, p.P)) < p.p_spont
            self.n_spont += int(new.sum())
            self.spont[new] = p.spont_len
            I = I + p.spont_I * (self.spont > 0)
            self.spont = np.maximum(self.spont - 1, 0)
        u = (p.G * np.maximum(I - p.theta, 0.0) - p.w_p * (self.pv + self.slow)[:, None]
             - blk_reg[:, None])
        if gated:
            u = u - p.pv_gate
        if p.noise > 0:
            u = u + p.noise * self.rng.standard_normal(u.shape)
        self.r += (np.clip(u, 0.0, 1.0) - self.r) / p.tau_r
        m = self.m()
        self.pv = self.pv * np.exp(-1.0 / p.tau_p) + p.g_p * m
        onset = (m_old < p.onset) & (m >= p.onset)
        offset = (m_old >= p.onset) & (m < p.onset)
        self.slow = self.slow * np.exp(-1.0 / p.tau_slow) + p.g_slow * offset
        self.hist[self.hptr] = m
        self.hptr = (self.hptr + 1) % p.d
        self.t += 1
        return list(np.where(onset)[0])


# ---------------------------------------------------------------------- composite
@dataclass
class CompParams:
    g_B: float = 3.0          # ring region -> bound word (basal)
    chain: bool = True        # chain transitions, rebound and context on
    relaunch: bool = False    # restart the ring from the spoken word when it is silent
    relaunch_after: int = 1   # silent cycles before relaunching
    bind_thr: float = 2.0     # region early-slot integral needed to bind
    # Stage 0.7: how much the chain's content pathways (transitions + rebound) speak
    schema: str = "on"        # "on" (Stage 0.6), "global" (constant lam), "gate" (local)
    lam: float = 1.0          # global schema gain for schema="global"
    gate_beta: float = 1.0    # gate: gain = clip(1 - beta * episode drive, 0, 1)
    restart_from: str = "spoken"  # "spoken" (Stage 0.6) or "bound" (last word the episode knows)


class Composite:
    """The Stage 0 machine (trained weights) plus a locked ring."""

    def __init__(self, machine, ring_params=None, comp=None, seed=0):
        self.m = machine
        self.ring = RingPop(ring_params or LOCKED, seed=seed + 7)
        self.c = comp or CompParams()
        N = self.ring.p.N
        self.B = np.zeros((N, V))
        self._J, self._U = machine.J.copy(), machine.U.copy()
        self.set_chain(self.c.chain)

    def set_chain(self, on):
        """Chain off = no learned transitions and no context (ring-only readout)."""
        self.m.J = self._J.copy() if on else np.zeros_like(self._J)
        self.m.U = self._U.copy() if on else np.zeros_like(self._U)
        # the rebound carry is what the chain uses to step itself; without J it
        # would only repeat the current word, so it is switched off too
        self.m.p = replace(self.m.p, kappa=1.0 if on else 0.0)

    def reset(self):
        self.m.reset_state()
        self.ring.reset()
        self.silent = 0
        self.last_word = None

    def cycle(self, word=None, drive=True, record=False, ais_open=True):
        """One chain cycle with the ring stepping inside it."""
        ring = self.ring
        acts = np.zeros(ring.p.N)
        rec = []
        early = self.m.p.early_end

        self._ep = 0.0             # episode drive at the previous step (for the gate)

        def gain_fn(k):
            if self.c.schema == "global":
                return self.c.lam
            return float(np.clip(1.0 - self.c.gate_beta * self._ep, 0.0, 1.0))

        def ext_fn(k, dead):
            ring.step(gated=dead)
            mm = ring.m()
            self._ep = float((self.B.T @ mm).max()) if drive else 0.0
            if k < early:
                acts[:] += mm
            if record:
                rec.append(mm.copy())
            if not drive:
                return 0.0
            return self.c.g_B * (self.B.T @ mm)

        o = self.m.cycle(word, ext_fn=ext_fn, record=record, ais_open=ais_open,
                         gain_fn=None if self.c.schema == "on" else gain_fn)
        o["ring_early"] = acts.copy()
        o["ring_region"] = int(np.argmax(acts)) if acts.max() > self.c.bind_thr else None
        if record:
            o["ring_r"] = np.array(rec)
        return o

    # -------------------------------------------------------------- protocols
    def hear(self, sentence):
        """Hear a sentence once (eyes open), ring running forward from region 0,
        binding each active region to the word current with it."""
        self.reset()
        self.B[:] = 0
        self.ring.schedule_kick(0, +1)
        outs = []
        for w in sentence:
            o = self.cycle(w, drive=False)
            if o["ring_region"] is not None:
                self.B[o["ring_region"], IDX[w]] = 1.0
            outs.append(o)
        return outs

    def recall(self, n, direction=+1, start=0, record=False, relaunch_dir=None):
        """Eyes closed: start the ring at region `start` going `direction` and
        publish whatever the machine says for n cycles."""
        self.reset()
        self.ring.schedule_kick(start, direction)
        outs = []
        for i in range(n):
            o = self.cycle(None, record=record)
            outs.append(o)
            if self.c.relaunch:
                self._maybe_relaunch(o, direction if relaunch_dir is None else relaunch_dir)
        return outs

    def _maybe_relaunch(self, o, direction):
        """Designed operation: if the ring has been silent, the word just spoken
        (or, if nothing was spoken, the last word spoken) re-ignites the region
        AFTER its bound region, blocked on the side it came from, so the ring
        resumes from there in the recall direction."""
        if o["ring_region"] is None:
            self.silent += 1
        else:
            self.silent = 0
        if self.c.restart_from == "bound":
            # resume from the most recent spoken word the episode actually holds
            if o["published"] is not None and self.B[:, IDX[o["published"]]].max() > 0:
                self.last_word = o["published"]
            w = self.last_word
        else:
            w = o["published"] if o["published"] is not None else self.last_word
            if o["published"] is not None:
                self.last_word = o["published"]
        if self.silent >= self.c.relaunch_after and w is not None:
            ks = np.where(self.B[:, IDX[w]] > 0)[0]
            if len(ks):
                k = (int(ks[0]) + direction) % self.ring.p.N   # the NEXT place
                self.ring.schedule_kick(k, direction)
                self.silent = 0

    def jump(self, word, direction=+1, n=3):
        """'What came after <word>?': hear the word (eyes open, one cycle), which
        re-ignites its bound region with a one-sided block, then eyes closed."""
        self.reset()
        ks = np.where(self.B[:, IDX[word]] > 0)[0]
        o0 = self.cycle(word, drive=False)
        if len(ks):
            self.ring.schedule_kick((int(ks[0]) + direction) % self.ring.p.N, direction)
        outs = [o0] + [self.cycle(None) for _ in range(n)]
        return outs
