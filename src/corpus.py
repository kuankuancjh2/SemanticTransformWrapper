"""Corpus acquisition.

Order of preference:
1. HuggingFace dataset (only when online and --hf-dataset given).
2. Bundled offline corpus (always available, zero network).

The bundled corpus is a template-generated set of Chinese sentence pairs
(narrative / paraphrase / QA). It is intentionally structured so that the
semantic tests (paraphrase consistency, subject/object swap) have real
ground truth, while remaining fully synthetic and license-free.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

SUBJECTS = ["小明", "小红", "老师", "妈妈", "爷爷", "姐姐", "男孩", "女孩", "医生", "猫"]
OBJECTS = ["苹果", "香蕉", "汽车", "书", "水杯", "面包", "钥匙", "花", "手机", "球"]
PLACES = ["学校", "家里", "公园", "商店", "图书馆", "厨房", "操场", "教室", "医院", "车上"]
ACTIONS = ["拿起", "放下", "买到", "找到", "丢掉了", "看着", "带来了", "洗干净了"]
TIME_WORDS = ["今天早上", "昨天下午", "刚才", "今天", "昨天", "上次"]
FEELINGS = ["很开心", "很紧张", "很平静", "有点累", "非常激动", "很满足"]
TOPICS = ["音乐", "科学", "历史", "运动", "旅行", "食物", "天气", "学习", "艺术", "科技"]

CAUSE_TEMPLATES = [
    "因为{topic}很有趣，{subj}每天都花时间研究",
    "{subj}喜欢{topic}，所以心情{feeling}",
    "由于{topic}很吸引人，{subj}忘记了时间",
]
CONT_TEMPLATES = [
    "{time}，{subj}在{place}{action}{obj}，然后{feeling}地笑了",
    "{time}{subj}{action}{obj}，心里{feeling}",
    "{subj}在{place}，{action}一个{obj}，感觉{feeling}",
]
COND_TEMPLATES = [
    "如果一个{obj}从桌子上掉下来，那么它可能会摔坏",
    "如果明天下雨，{subj}就留在{place}里",
    "要是{subj}能早点起床，就不会迟到",
]
PARAPHRASE_PAIRS = [
    ("小明吃了一个苹果。", "一个苹果被小明吃掉了。"),
    ("小红买了一辆汽车。", "一辆汽车被小红买回来了。"),
    ("老师正在批改作业。", "作业正被老师批改着。"),
    ("猫在追一只鸟。", "一只鸟被猫追着。"),
    ("爷爷在花园里浇水。", "花园里的花被爷爷浇了水。"),
    ("妈妈做了晚饭。", "晚饭被妈妈做好了。"),
    ("他把信寄了出去。", "那封信被他寄了出去。"),
    ("我们完成了这个项目。", "这个项目被我们完成了。"),
]
QA_PAIRS = [
    ("苹果有什么营养？", "苹果含有膳食纤维、维生素和多种抗氧化物质。"),
    ("为什么天空是蓝色的？", "因为大气分子对短波长的蓝光散射更强。"),
    ("猫为什么喜欢晒太阳？", "晒太阳可以帮助猫保持体温并且感到放松。"),
    ("人为什么要睡觉？", "睡眠帮助大脑清理代谢废物并巩固记忆。"),
    ("水在多少度沸腾？", "在标准大气压下，水在一百度沸腾。"),
    ("怎样学好一门语言？", "坚持每天练习听说读写，并且大量接触真实的语言材料。"),
]

OPENER_PREFIX = [
    "今天天气很好，我决定",
    "人工智能最重要的问题之一是",
    "我喜欢音乐，因为",
    "回想起那一天，",
    "站在窗边，他忽然想到",
]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        subj=rng.choice(SUBJECTS),
        obj=rng.choice(OBJECTS),
        place=rng.choice(PLACES),
        action=rng.choice(ACTIONS),
        time=rng.choice(TIME_WORDS),
        feeling=rng.choice(FEELINGS),
        topic=rng.choice(TOPICS),
    )


def build_bundled_corpus(num_samples: int = 6000, seed: int = 42) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return {'train': [(prompt, target), ...], 'val': [...], 'test': [...]}.

    Each sample is a (prompt, target, ptype) triple. For most samples the target is a
    natural continuation (prompt == a prefix of the target, which is exactly
    the Stage-1 autoencoding + Stage-2 semantic-continuation setup).
    """
    rng = random.Random(seed)

    def one_sentence() -> str:
        kind = rng.random()
        if kind < 0.4:
            return _fill(rng.choice(CONT_TEMPLATES), rng) + "。"
        if kind < 0.7:
            return _fill(rng.choice(CAUSE_TEMPLATES), rng) + "。"
        if kind < 0.85:
            return _fill(rng.choice(COND_TEMPLATES), rng)
        return rng.choice(QA_PAIRS)[1]

    def continuation() -> Tuple[str, str]:
        s = one_sentence()
        # cut a prompt prefix at a character boundary, target completes it
        cut = max(2, len(s) // 2)
        cut += rng.randint(0, 2)
        cut = min(cut, len(s) - 1)
        prompt, target = s[:cut], s
        return prompt, target

    def paraphrase_pair() -> Tuple[str, str]:
        return rng.choice(PARAPHRASE_PAIRS)

    def qa() -> Tuple[str, str]:
        return rng.choice(QA_PAIRS)

    def opener() -> Tuple[str, str]:
        p = rng.choice(OPENER_PREFIX)
        t = p + one_sentence()
        return p, t

    samples: List[Tuple[str, str, str]] = []
    n = num_samples
    for _ in range(n):
        r = rng.random()
        if r < 0.55:
            samples.append((*continuation(), "cont"))
        elif r < 0.70:
            samples.append((*paraphrase_pair(), "para"))
        elif r < 0.85:
            samples.append((*qa(), "qa"))
        else:
            samples.append((*opener(), "open"))
    # dedup while keeping order
    seen, uniq = set(), []
    for s in samples:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    rng.shuffle(uniq)
    total = len(uniq)
    n_val = max(1, int(total * 0.1))
    n_test = max(1, int(total * 0.1))
    return {
        "train": uniq[: total - n_val - n_test],
        "val": uniq[total - n_val - n_test : total - n_test],
        "test": uniq[total - n_test :],
    }
