"""The running-sequence machine (Stage 0).

One rate unit per word assembly. Time is discrete (1 step = 2.5 ms); one theta-
like cycle = T steps (default 50 = 125 ms). One word per cycle.

Per step, for every word assembly i:

    basal_i  = input_i(t) + rebound_i(t) + sum_j J[j,i] r_j(t - d)
    apical_i = sum_c U[c,i] ctx_c (1 - veto_c)          (constant within a cycle)
    drive_i  = G relu(basal_i - theta) (1 + g_ap apical_i)      # multiplicative
             | G relu(basal_i + g_add apical_i - theta)         # additive ablation
    u_i      = drive_i - w_a a_i - w_inh sum_{k!=i} r_k - PV(phase) + noise
    r_i     <- r_i + (clip(u_i, 0, 1) - r_i) / tau_r
    a_i     <- a_i + (r_i - a_i) / tau_a

The four doors, and nothing else, are the controls:

  * PV / basket rhythm   PV(phase) is zero in the open window and large in the
                         dead interval at the end of each cycle.
  * apical context       ctx (slow leaky trace of words) reaches assemblies only
                         through U, and only multiplies basal drive.
  * SST / Martinotti     veto_c closes context unit c's apical route. Recruited
                         by a failed prediction, in proportion to how much c
                         steered that prediction. Decays over a few cycles.
  * AIS / chandelier     gates PUBLICATION of the cycle's current word. It does
                         not touch r, carry or ctx.

Carry across the dead interval: the assembly most active in the LATE slot of a
cycle (the prediction) leaves a trace that rebounds at the opening of the next
window. With input (eyes open) the rebound competes with the arriving word;
without input (eyes closed) the rebound IS the next word. That single mechanism
is the crossover from world-driven to model-driven sequence.

Learning (local, reading mode only):
  * J  normalized Hebbian on consecutive confirmed words (row = transition
       probabilities, learned as an exponential moving average).
  * U  plateau rule: when the arriving word was not the carried prediction,
       a plateau in the arriving word's assembly potentiates U from the context
       units active now; the failed prediction's U from those units is depressed.
       No surprise, no learning.
The SST veto circuit is designed, not learned.
"""
from dataclasses import dataclass, field, replace
import numpy as np

from .grammar import VOCAB, IDX, V, successors, TOPICS

TOPIC_WORDS = set(TOPICS)


@dataclass
class Params:
    T: int = 50            # steps per cycle
    win_end: int = 32      # end of the open window (carry is read here)
    dead: int = 18         # steps of PV inhibition at end of cycle (ablation knob)
    early_end: int = 16    # early slot [0, early_end) = current word
    in_on: int = 2
    in_off: int = 14
    reb_off: int = 8
    d: int = 17            # propagation delay of one recurrent hop
    tau_r: float = 2.0
    tau_a: float = 8.0
    w_a: float = 1.2
    w_inh: float = 0.8
    theta: float = 0.08
    G: float = 2.5
    g_ap: float = 3.0      # multiplicative apical gain
    g_add: float = 0.0     # additive apical gain (ablation)
    mult: bool = True
    I_in: float = 1.0
    kappa: float = 1.0
    pv: float = 20.0
    noise: float = 0.03
    p_spont: float = 0.0   # prob per step per assembly of a spontaneous ignition
    spont_I: float = 1.0   # its input strength (same as a real word)
    spont_len: int = 6     # its duration in steps
    topic_gain: float = 1.0  # how strongly a topic word is written into context
    thr_E: float = 2.0     # early-slot integral needed to call a current word
    thr_L: float = 1.5     # late-slot integral needed to call a prediction
    tau_ctx: float = 6.0   # cycles
    tau_veto: float = 6.0  # cycles
    sst: bool = True
    eta_sst: float = 1.5   # veto per unit of steering credit
    credit_norm: bool = True
    u_rule: str = "delta"   # 'delta' or 'btsp' (weight-dependent potentiation)
    u_decay: float = 0.0    # shrink the plateau's columns before writing
    eta_J: float = 0.01    # floor of the consolidating 1/n rate
    eta_U: float = 0.2     # plateau rule step (potentiation of arrived word,
                           # depression of the failed prediction)
    theta_ap: float = 0.2  # tuft threshold
    ap_scale: float = 0.3  # tuft saturation width
    cosine: bool = True    # tuft reads cosine (True) or dot product (False)
    carry_decay: float = 0.5
    carry_cut: float = 0.5
    sup_gain: float = 3.0  # content suppression strength (item refractory)
    sup_decay: float = 0.3 # per cycle


class Machine:
    def __init__(self, params=None, seed=0):
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        self.J = np.zeros((V, V))
        self.U = np.zeros((V, V))
        self.nJ = np.zeros(V)
        self.reset_state()

    # ------------------------------------------------------------------ state
    def reset_state(self):
        p = self.p
        self.r = np.zeros(V)
        self.a = np.zeros(V)
        self.hist = np.zeros((p.d, V))   # ring buffer of r for the delay line
        self.hptr = 0
        self.carry = np.zeros(V)
        self.ctx = np.zeros(V)
        self.veto = np.zeros(V)
        self.sup = np.zeros(V)            # content suppression (item refractory)
        self.spont = np.zeros(V, dtype=int)
        self.prev_word = None             # last current word (confirmed or internal)
        self.t = 0

    def snapshot(self):
        return dict(r=self.r.copy(), a=self.a.copy(), hist=self.hist.copy(),
                    hptr=self.hptr, carry=self.carry.copy(), ctx=self.ctx.copy(),
                    veto=self.veto.copy(), sup=self.sup.copy(),
                    prev_word=self.prev_word, t=self.t,
                    rng=self.rng.bit_generator.state)

    def restore(self, s):
        self.r, self.a, self.hist = s["r"].copy(), s["a"].copy(), s["hist"].copy()
        self.hptr, self.carry, self.ctx = s["hptr"], s["carry"].copy(), s["ctx"].copy()
        self.veto, self.prev_word, self.t = s["veto"].copy(), s["prev_word"], s["t"]
        self.sup = s["sup"].copy()
        self.rng.bit_generator.state = s["rng"]

    def apical(self):
        """Tuft activation per assembly: cosine between the (veto-gated) context
        and the assembly's learned context prototype, passed through a
        threshold (dendritic calcium-spike-like nonlinearity). In [0, 1]."""
        p = self.p
        cv = self.ctx * (1.0 - self.veto)
        n = np.linalg.norm(cv)
        if n < 1e-9:
            return np.zeros(V)
        if p.cosine:
            x = (self.U.T @ (cv / n)) / (np.linalg.norm(self.U, axis=0) + 1e-9)
        else:
            x = self.U.T @ (cv / n)
        return np.clip((x - p.theta_ap) / p.ap_scale, 0.0, 1.0)

    def Un(self):
        return self.U / (np.linalg.norm(self.U, axis=0, keepdims=True) + 1e-9)

    # ------------------------------------------------------------------ cycle
    def cycle(self, word=None, ais_open=True, freeze=False, learn=False,
              record=False, clamp_carry=None, ext_fn=None, gain_fn=None):
        """Run one cycle. word=None means eyes closed (no input).
        freeze=True: PV inhibition for the whole cycle (recurrence stopped).
        Returns a dict describing the cycle."""
        p = self.p
        lam = np.exp(-1.0 / p.tau_ctx)
        lam_v = np.exp(-1.0 / p.tau_veto)

        carry_prev = self.carry.copy() if clamp_carry is None else clamp_carry
        pred_prev = VOCAB[int(np.argmax(carry_prev))] if carry_prev.max() > 0 else None

        ap = self.apical()
        amp = 1.0 + p.g_ap * ap
        inp = np.zeros(V)
        if word is not None:
            inp[IDX[word]] = p.I_in
        E = np.zeros(V)
        L = np.zeros(V)
        rec_r = np.zeros((p.T, V)) if record else None
        rec_basal_max = np.zeros(V)   # per-assembly max basal support this cycle

        for k in range(p.T):
            # delayed recurrent input
            r_del = self.hist[self.hptr]
            # gain_fn scales the chain's own content pathways (learned
            # transitions and the rebound); None = unscaled, as in Stage 0
            g = 1.0 if gain_fn is None else gain_fn(k)
            basal = J_in = self.J.T @ r_del
            basal = basal.copy() if gain_fn is None else g * basal
            if word is not None and p.in_on <= k < p.in_off:
                basal += inp
            if k < p.reb_off:
                basal += (p.kappa * carry_prev) if gain_fn is None else (g * p.kappa * carry_prev)
            if p.p_spont > 0 and not freeze:
                # spontaneous ignitions: a random assembly gets a brief input
                new = self.rng.random(V) < p.p_spont
                self.spont[new] = p.spont_len
                basal += p.spont_I * (self.spont > 0)
                self.spont = np.maximum(self.spont - 1, 0)
            if ext_fn is not None:
                # another circuit sharing this rhythm (e.g. the ring); it is told
                # whether PV is closing the window at this step
                basal += ext_fn(k, freeze or k >= p.T - p.dead)
            np.maximum(rec_basal_max, basal, out=rec_basal_max)

            if p.mult:
                drive = p.G * np.maximum(basal - p.theta, 0.0) * amp
            else:
                drive = p.G * np.maximum(basal + p.g_add * ap - p.theta, 0.0)

            pv = p.pv if (freeze or k >= p.T - p.dead) else 0.0
            tot = self.r.sum()
            u = drive - p.w_a * self.a - p.w_inh * (tot - self.r) - pv - p.sup_gain * self.sup
            if p.noise > 0:
                u = u + p.noise * self.rng.standard_normal(V)
            self.r += (np.clip(u, 0.0, 1.0) - self.r) / p.tau_r
            self.a += (self.r - self.a) / p.tau_a

            # write delay line
            self.hist[self.hptr] = self.r
            self.hptr = (self.hptr + 1) % p.d

            if k < p.early_end:
                E += self.r
            elif k < p.win_end:
                L += self.r
            if record:
                rec_r[k] = self.r
            self.t += 1

        # ---- read the cycle
        current = VOCAB[int(np.argmax(E))] if E.max() > p.thr_E else None
        confirmed = word if word is not None else current

        Lx = L.copy()
        if confirmed is not None:
            Lx[IDX[confirmed]] = 0.0
        # how many assemblies are strongly active in the prediction slot
        n_late = int((Lx > p.thr_L).sum())
        if freeze:
            self.carry = self.carry * p.carry_decay
        elif Lx.max() > p.thr_L:
            c = Lx / Lx.max()
            c[c < p.carry_cut] = 0.0
            self.carry = c
        else:
            self.carry = self.carry * p.carry_decay
            self.carry[self.carry < p.carry_cut] = 0.0
        pred = VOCAB[int(np.argmax(self.carry))] if self.carry.max() > 0 else None

        published = confirmed if (ais_open and confirmed is not None and word is None) else None
        if word is not None and ais_open:
            published = word   # reading aloud: pass the word on

        # ---- mismatch: last cycle's prediction vs what arrived
        mismatch = (word is not None and pred_prev is not None and pred_prev != word)
        surprise = (word is not None and pred_prev != word)

        if learn and word is not None:
            if self.prev_word is not None:
                P = IDX[self.prev_word]
                self.nJ[P] += 1
                eta = max(p.eta_J, 1.0 / self.nJ[P])   # consolidating: 1/n, floor eta_J
                self.J[P] *= (1.0 - eta)
                self.J[P, IDX[word]] += eta
            n = np.linalg.norm(self.ctx)
            if surprise and n > 0:
                cn = self.ctx / n
                Y = IDX[word]
                if p.u_decay > 0:
                    self.U[:, Y] *= (1.0 - p.u_decay)
                    if pred_prev is not None:
                        self.U[:, IDX[pred_prev]] *= (1.0 - p.u_decay)
                if p.u_rule == "delta":
                    self.U[:, Y] += p.eta_U * cn
                else:   # weight-dependent (BTSP-like): move toward the current context
                    self.U[:, Y] += p.eta_U * (cn - self.U[:, Y])
                if pred_prev is not None:
                    X = IDX[pred_prev]
                    self.U[:, X] -= p.eta_U * carry_prev[X] * cn
                np.maximum(self.U, 0.0, out=self.U)

        vetoed = None
        if mismatch and p.sst:
            vetoed = self.sst_veto(pred_prev)

        # ---- slow variables
        self.veto *= lam_v
        self.sup *= p.sup_decay
        if confirmed is not None:
            self.ctx = lam * self.ctx
            self.ctx[IDX[confirmed]] += p.topic_gain if confirmed in TOPIC_WORDS else 1.0
        else:
            self.ctx = lam * self.ctx
        prev = self.prev_word
        self.prev_word = confirmed

        out = dict(word=word, current=current, confirmed=confirmed, pred=pred,
                   pred_prev=pred_prev, published=published, mismatch=mismatch,
                   n_late=n_late, E=E, L=L, ap=ap, ais_open=ais_open,
                   freeze=freeze, vetoed=vetoed, veto=self.veto.copy(),
                   basal_max=rec_basal_max, prev=prev)
        if record:
            out["r"] = rec_r
        return out

    # ------------------------------------------------------------------ SST
    def sst_veto(self, failed_word, base_word=None):
        """Close the apical route of the context units that steered the failed
        prediction. Credit for unit c = ctx_c * (U[c, failed] - mean over the
        failed word's rival candidates of U[c, rival])+.  Returns vetoed words."""
        p = self.p
        base = base_word if base_word is not None else self.prev_word
        if base is None:
            return None
        cands = [w for w in successors(base)]
        X = IDX[failed_word]
        rivals = [IDX[w] for w in cands if w != failed_word]
        if not rivals:
            return None
        Uc = self.Un() if p.credit_norm else self.U
        credit = self.ctx * (1.0 - self.veto) * np.maximum(Uc[:, X] - Uc[:, rivals].mean(axis=1), 0.0)
        if credit.max() <= 1e-6:
            return None
        self.veto = np.minimum(1.0, self.veto + p.eta_sst * credit)
        return [VOCAB[i] for i in np.argsort(-credit)[:3] if p.eta_sst * credit[i] > 0.2]

    # ------------------------------------------------------------------ "no"
    def reject(self, word, base, route=True, item=True, backup=True):
        """An outside 'no, not that' arriving right after `word` was produced
        from branch point `base`.
          route : SST closes the apical route of the context units that steered
                  `word` (credit as in sst_veto)
          item  : the assembly of `word` itself is suppressed for a cycle or two
          backup: the sequence is re-launched from `base` (the last accepted
                  word rebounds in the next window)
        The rejected word is taken back out of the context trace."""
        X = IDX[word]
        self.ctx[X] = max(0.0, self.ctx[X] - 1.0)
        vetoed = None
        if route:
            vetoed = self.sst_veto(word, base_word=base)
        if item:
            self.sup[X] = 1.0
        if backup:
            c = np.zeros(V)
            c[IDX[base]] = 1.0
            self.carry = c
            self.prev_word = base
        return vetoed

    # ------------------------------------------------------------------ helpers
    def read(self, words, learn=False, record=False):
        return [self.cycle(w, learn=learn, record=record) for w in words]

    def free_run(self, n, record=False, ais=None, freeze=None):
        """Eyes closed for n cycles. ais / freeze: optional sets of cycle
        indices where the AIS is closed / the rhythm is frozen."""
        outs = []
        for i in range(n):
            outs.append(self.cycle(None, ais_open=not (ais and i in ais),
                                   freeze=bool(freeze and i in freeze),
                                   record=record))
        return outs
