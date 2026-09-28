"""Corpus acquisition (Zip-B: STANDARD MESSAGES FORMAT).

Order of preference:
1. HuggingFace dataset (only when online and --hf-dataset given): each example
   becomes a single-turn conversation {user: prompt, assistant: target}.
2. Bundled offline corpus (always available, zero network).

The bundled corpus generates multi-turn conversations in the standard
messages format:

    {"messages": [{"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}, ...]}

- 1-3 exchanges per conversation (user/assistant pairs).
- The LAST assistant turn is the training target for Stage 2/3; everything
  before it is context (history turns update latent-memory core states; the
  final user turn is the stimulus prompt).
- Stage 1 uses single-turn reconstruction only: individual utterances
  (+ the dedicated paraphrase pairs, kept for the consistency loss).
"""
from __future__ import annotations

import random
from typing import Dict, List

SUBJECTS = ["Alice", "Bob", "Tom", "Mary", "Mom", "Dad", "the boy", "the girl"]
OBJECTS = ["apple", "banana", "car", "book", "cup", "bread", "key", "flower"]
PLACES = ["school", "home", "the park", "the shop", "the library", "the kitchen"]
ACTIONS = ["picked up", "put down", "bought", "found", "lost", "looked at",
           "brought", "washed"]
TIME_WORDS = ["This morning", "Yesterday afternoon", "Just now", "Today",
              "Yesterday", "Last time"]
FEELINGS = ["very happy", "nervous", "calm", "a bit tired", "excited", "satisfied"]
TOPICS = ["music", "science", "history", "sports", "travel", "food", "weather",
          "learning", "art", "technology"]

CAUSE_TEMPLATES = [
    "Because {topic} is interesting, {subj} spends time on it every day",
    "{subj} likes {topic}, so he feels {feeling}",
    "Since {topic} is fascinating, {subj} forgot the time",
]
CONT_TEMPLATES = [
    "{time}, {subj} {action} the {obj} at {place}, and then smiled, feeling {feeling}",
    "{time} {subj} {action} a {obj} and felt {feeling}",
    "{subj} was at {place}, {action} a {obj}, and it felt {feeling}",
]
COND_TEMPLATES = [
    "If a {obj} falls off the table, it might break",
    "If it rains tomorrow, {subj} will stay at {place}",
    "If {subj} could get up earlier, he would not be late",
]
PARAPHRASE_PAIRS = [
    ("Alice ate an apple.", "An apple was eaten by Alice."),
    ("Bob bought a car.", "A car was bought by Bob."),
    ("The teacher is grading homework.", "The homework is being graded by the teacher."),
    ("The cat is chasing a bird.", "A bird is being chased by the cat."),
    ("Grandpa is watering the garden.", "The garden is being watered by Grandpa."),
    ("Mom cooked dinner.", "Dinner was cooked by Mom."),
    ("He sent the letter out.", "The letter was sent out by him."),
    ("We finished this project.", "This project was finished by us."),
]
QA_PAIRS = [
    ("What nutrition does an apple have?",
     "An apple contains dietary fiber, vitamins and many antioxidants."),
    ("Why is the sky blue?",
     "Because air molecules scatter short-wavelength blue light more strongly."),
    ("Why do cats like sunbathing?",
     "Sunbathing helps cats keep warm and feel relaxed."),
    ("Why do people sleep?",
     "Sleep helps the brain clear metabolic waste and consolidate memories."),
    ("At what temperature does water boil?",
     "At standard atmospheric pressure, water boils at one hundred degrees."),
    ("How to learn a language well?",
     "Practice listening, speaking, reading and writing every day, and read a lot."),
]

OPENER_PREFIX = [
    "The weather today is good, I decided",
    "One of the most important questions about AI is",
    "I like music because",
    "Remembering that day,",
    "Standing by the window, he suddenly thought",
]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        subj=rng.choice(SUBJECTS), obj=rng.choice(OBJECTS),
        place=rng.choice(PLACES), action=rng.choice(ACTIONS),
        time=rng.choice(TIME_WORDS), feeling=rng.choice(FEELINGS),
        topic=rng.choice(TOPICS))


def _sentence(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.45:
        return _fill(rng.choice(CONT_TEMPLATES), rng)
    if r < 0.75:
        return _fill(rng.choice(CAUSE_TEMPLATES), rng) + "."
    if r < 0.9:
        return _fill(rng.choice(COND_TEMPLATES), rng)
    return rng.choice(QA_PAIRS)[1]


def _user_turn(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return rng.choice(QA_PAIRS)[0]
    if r < 0.7:
        return rng.choice(OPENER_PREFIX)
    s = _sentence(rng)
    return s[: max(2, len(s) // 2)]  # a statement stem the assistant completes


def _assistant_turn(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.35:
        return rng.choice(QA_PAIRS)[1]
    if r < 0.6:
        p = rng.choice(OPENER_PREFIX)
        return p + _sentence(rng)
    return _sentence(rng) + ("." if not _sentence(rng).endswith(".") else "")


def build_bundled_corpus(num_convos: int = 4000, seed: int = 42
                         ) -> Dict[str, List[Dict]]:
    """Return {'train'/'val'/'test': [ {"messages": [...]}, ... ]}.

    Conversation shapes:
      - single exchange: user question/opener/statement -> assistant answer
      - 2-3 exchanges: multi-turn; later user turns reference nothing external,
        so context modeling is purely about the dialogue history
      - paraphrase convos: user = A, assistant = B (paraphrase pair)
    """
    rng = random.Random(seed)
    convos: List[Dict] = []
    for _ in range(num_convos):
        r = rng.random()
        if r < 0.10:  # paraphrase single-turn
            a, b = rng.choice(PARAPHRASE_PAIRS)
            convos.append({"messages": [{"role": "user", "content": a},
                                        {"role": "assistant", "content": b}]})
        elif r < 0.55:  # single exchange
            convos.append({"messages": [
                {"role": "user", "content": _user_turn(rng)},
                {"role": "assistant", "content": _assistant_turn(rng)}]})
        else:  # multi-turn: 2-3 exchanges
            n = rng.choice([2, 2, 3])
            msgs = []
            for i in range(n):
                u = _user_turn(rng) if i > 0 or rng.random() < 0.5 else \
                    rng.choice(QA_PAIRS)[0]
                a = _assistant_turn(rng)
                msgs.append({"role": "user", "content": u})
                msgs.append({"role": "assistant", "content": a})
            convos.append({"messages": msgs})
    seen, uniq = set(), []
    for c in convos:
        key = tuple((m["role"], m["content"]) for m in c["messages"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    rng.shuffle(uniq)
    total = len(uniq)
    n_val = max(1, int(total * 0.1))
    n_test = max(1, int(total * 0.1))
    return {"train": uniq[: total - n_val - n_test],
            "val": uniq[total - n_val - n_test: total - n_test],
            "test": uniq[total - n_test:]}