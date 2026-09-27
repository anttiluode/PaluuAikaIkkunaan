"""Stage 0 story grammar.

Twenty words. Every story has the same eleven-slot skeleton:

    topic  subj  walks  to  A  with  a  B  and  C  .

Three kinds of branch point:
  * random, context-free:   '.' -> topic (3-way),  topic -> subj (girl/boy)
  * context-resolved:       'to' -> A,  'a' -> B,  'and' -> C   (3-way each)
  * deterministic:          everything else

The context-resolved branches B and C sit AFTER merged segments ("with a",
"and"), so the only thing that can pick the right continuation is context that
survived the merge. The world rule is: the A word determines B and C; the topic
usually agrees with A (always, in training data).
"""
import numpy as np

TOPICS = {
    "winter": ("hill", "sled", "slides"),
    "beach": ("sea", "ball", "swims"),
    "night": ("bed", "book", "reads"),
}
SUBJECTS = ("girl", "boy")

VOCAB = (
    ["winter", "beach", "night", "girl", "boy", "walks", "to"]
    + ["hill", "sea", "bed", "with", "a"]
    + ["sled", "ball", "book", "and", "slides", "swims", "reads", "."]
)
IDX = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)

# slot index of each word in the 11-slot skeleton (used to check position)
SLOT = {}
for w in TOPICS:
    SLOT[w] = 0
for w in SUBJECTS:
    SLOT[w] = 1
SLOT.update({"walks": 2, "to": 3, "with": 5, "a": 6, "and": 8, ".": 10})
for t, (A, B, C) in TOPICS.items():
    SLOT[A], SLOT[B], SLOT[C] = 4, 7, 9
NSLOT = 11

# which world (topic family) a branch word belongs to
WORLD = {}
for t, (A, B, C) in TOPICS.items():
    for w in (t, A, B, C):
        WORLD[w] = t

CONTEXT_BRANCH_SLOTS = (4, 7, 9)       # A, B, C
RANDOM_BRANCH_SLOTS = (0, 1)           # topic, subject


def story(rng, topic=None, subj=None, A_override=None):
    """One story. A_override puts a different world's A word in slot 4 and
    makes B, C follow the A word (the world rule)."""
    topic = topic or rng.choice(list(TOPICS))
    subj = subj or rng.choice(SUBJECTS)
    world = WORLD[A_override] if A_override else topic
    A, B, C = TOPICS[world]
    return [topic, subj, "walks", "to", A, "with", "a", B, "and", C, "."]


def valid_transition(w1, w2):
    """Is w1 -> w2 allowed by the grammar (ignoring context)?"""
    s1, s2 = SLOT[w1], SLOT[w2]
    return s2 == (s1 + 1) % NSLOT


def successors(w):
    s = SLOT[w]
    return [v for v in VOCAB if SLOT[v] == (s + 1) % NSLOT]
