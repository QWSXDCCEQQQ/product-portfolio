"""纯 Python 的主题分析引擎。

不依赖 numpy / sklearn（本机 Anaconda 的 numpy C 扩展损坏），
分词用 jieba，向量化手写 TF-IDF，聚类用球形 K-means。

算法分工（这是实测出来的，不是拍脑袋）：
    · 账号赛道识别 → TF-IDF 加权 + 文档频次分组
      61 条短标题做层次聚类，阈值从 0.03 调到 0.16 都碎成 20+ 簇，
      而词频分布本身很干净（保研 17 / zotero 14 / 推免 14 断层领先），
      所以按「赛道词」归并比硬聚类更准、也更能给用户解释。
    · 竞品主题地图 → 球形 K-means（余弦距离，k-means++ 初始化）
      给定 k 个主题覆盖全部样本，得到的是「赛道被切成哪几块」，
      而不是一堆只有一个样本的碎簇。
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any, Iterable

try:  # 分词失败也不该让整个研究台挂掉
    import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False


# 通用词：出现在标题里但没有区分度
STOPWORDS = {
    "小红书", "笔记", "分享", "一个", "一下", "我们", "你们", "他们", "自己", "大家",
    "可以", "怎么", "什么", "为什么", "如何", "这样", "那样", "这个", "那个", "这些",
    "那些", "就是", "还是", "但是", "因为", "所以", "如果", "已经", "真的", "其实",
    "现在", "今天", "明天", "时候", "东西", "问题", "方法", "内容", "需要", "应该",
    "知道", "觉得", "看到", "不要", "不会", "不是", "没有", "还有", "一些", "很多",
    "非常", "特别", "一定", "可能", "起来", "出来", "过来", "下来", "上去", "之后",
    "之前", "以后", "以前", "里面", "上面", "下面", "这里", "那里", "哪里", "几个",
    "第一", "第二", "第三", "最新", "收藏", "点赞", "关注", "评论", "博主", "up",
    "教程", "干货", "建议", "经验", "攻略", "合集", "整理", "推荐", "必看", "一篇",
    "系列", "更新", "刚更", "刚刚", "最后", "全部", "所有", "以及", "而且", "或者",
    "不过", "虽然", "然后", "于是", "因此", "怎么样", "怎么办", "是不是", "有没有",
    "一次", "一点", "更多", "最好", "重要", "简单", "快速", "立刻", "马上",
    "到底", "继续", "即可", "一起", "这份", "使用", "记得", "感觉", "有用", "参考",
    "看看", "看完", "码住", "安利", "私信", "评论区", "姐妹", "宝子", "能不能", "要不要",
    "第一期", "第二期", "第三期", "第四期", "第五期", "第六期", "第七期", "第八期",
}

# 数字、年份、序数这类词不当赛道词，也不当种子词
_NUMERIC = re.compile(r"^[0-9]+$|^[0-9]+[届期个条步种年月日时分秒小时天周]$")
# 短英文词只有在本身是领域缩写时才保留，否则「保研er」会切出无意义的 er
ASCII_KEEP = {"ai", "ip", "3d", "qq", "ps", "pr", "ui", "ux", "ppt", "pdf", "word"}
GENERIC_SUFFIX = re.compile(r"(注意点|注意|记录|总结|情况|事情|方面|部分|时候)$")

TOOL_HINTS = re.compile(r"工具|插件|模板|清单|步骤|入门|教程|指南|手册|配置|安装|设置|流程|软件|文献")
LIST_HINTS = re.compile(r"[0-9]+[个条期步种]|[①②③④⑤⑥⑦⑧⑨]|合集|汇总|清单|盘点|整理")
EMOTION_HINTS = re.compile(r"救命|崩溃|焦虑|后悔|血泪|踩坑|避雷|翻车|劝退|太难|别再|千万别|慌了")


def tokenize(text: str) -> list[str]:
    """把标题/正文切成有信息量的词。"""
    text = re.sub(r"[#@]", " ", str(text or ""))
    text = re.sub(r"https?://\S+", " ", text)
    if _HAS_JIEBA:
        raw_tokens = jieba.lcut(text)
    else:  # 退化为字 bigram，保证功能可用
        cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)
        raw_tokens = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]

    tokens: list[str] = []
    for raw in raw_tokens:
        token = raw.strip().lower()
        if not token:
            continue
        if re.fullmatch(r"[a-z0-9][a-z0-9+.#\-]{1,}", token) and not _NUMERIC.match(token):
            if len(token) >= 3 or token in ASCII_KEEP:
                tokens.append(token)
            continue
        if not re.fullmatch(r"[\u4e00-\u9fa5]+", token):
            continue
        if token in STOPWORDS or _NUMERIC.match(token):
            continue
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def build_tfidf(docs: list[list[str]]) -> list[dict[str, float]]:
    """手写 TF-IDF：sublinear tf + 平滑 idf + L2 归一化。"""
    count = len(docs)
    df: Counter[str] = Counter()
    term_freqs: list[Counter[str]] = []
    for tokens in docs:
        freq = Counter(tokens)
        term_freqs.append(freq)
        for term in freq:
            df[term] += 1

    vectors: list[dict[str, float]] = []
    for freq in term_freqs:
        vector: dict[str, float] = {}
        for term, times in freq.items():
            tf = 1.0 + math.log(times)
            idf = math.log((count + 1) / (df[term] + 1)) + 1.0
            vector[term] = tf * idf
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({term: value / norm for term, value in vector.items()})
    return vectors


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def domain_terms(notes: list[dict[str, Any]], min_df: int = 2, limit: int = 12) -> list[dict[str, Any]]:
    """找出一个账号真正在反复写的赛道词（文档频次优先，TF-IDF 只做排序微调）。"""
    docs = [tokenize(str(note.get("title") or "")) for note in notes]
    if not docs:
        return []
    df: Counter[str] = Counter()
    likes_by_term: dict[str, list[int]] = {}
    for note, tokens in zip(notes, docs):
        for term in set(tokens):
            df[term] += 1
            likes_by_term.setdefault(term, []).append(int(note.get("likes") or 0))

    total = len(docs)
    terms: list[dict[str, Any]] = []
    for term, count in df.items():
        if count < min_df:
            continue
        if _NUMERIC.match(term) or GENERIC_SUFFIX.search(term):
            continue
        likes = likes_by_term.get(term, [0])
        avg_likes = sum(likes) / len(likes)
        # 覆盖度为主，表现为辅：出现得多说明是主赛道，平均赞高说明这条赛道有效
        weight = count * (1.0 + math.log1p(avg_likes) / 6.0)
        terms.append(
            {
                "term": term,
                "count": count,
                "coverage": round(count / total * 100),
                "avgLikes": round(avg_likes),
                "weight": round(weight, 2),
            }
        )
    terms.sort(key=lambda item: (-item["weight"], -item["count"]))
    return terms[:limit]


def _metrics(notes: list[dict[str, Any]]) -> dict[str, Any]:
    likes = [int(note.get("likes") or 0) for note in notes]
    collections = [int(note.get("collections") or 0) for note in notes]
    comments = [int(note.get("comments") or 0) for note in notes]
    total_like = sum(likes)
    total_collect = sum(collections)
    return {
        "count": len(notes),
        "likes": total_like,
        "collections": total_collect,
        "comments": sum(comments),
        "avgLikes": round(total_like / len(notes)) if notes else 0,
        "avgCollections": round(total_collect / len(notes)) if notes else 0,
        "maxLikes": max(likes) if likes else 0,
        "ratio": round(total_collect / total_like, 2) if total_like else 0,
        "commentRate": round(sum(comments) / total_like * 100, 2) if total_like else 0,
    }


def _flags(notes: list[dict[str, Any]]) -> dict[str, Any]:
    titles = [str(note.get("title") or "") for note in notes]
    if not titles:
        return {"tool": 0, "list": 0, "emotion": 0, "titleLen": 0}
    return {
        "tool": round(sum(1 for t in titles if TOOL_HINTS.search(t)) / len(titles) * 100),
        "list": round(sum(1 for t in titles if LIST_HINTS.search(t)) / len(titles) * 100),
        "emotion": round(sum(1 for t in titles if EMOTION_HINTS.search(t)) / len(titles) * 100),
        "titleLen": round(sum(len(t) for t in titles) / len(titles), 1),
    }


def _sample(notes: list[dict[str, Any]], size: int = 3) -> list[dict[str, Any]]:
    ordered = sorted(notes, key=lambda n: int(n.get("likes") or 0), reverse=True)[:size]
    return [
        {
            "title": str(n.get("title") or ""),
            "author": str(n.get("author") or ""),
            "likes": int(n.get("likes") or 0),
            "collections": int(n.get("collections") or 0),
            "authorId": str(n.get("authorId") or ""),
            "id": str(n.get("id") or ""),
        }
        for n in ordered
    ]


def analyze_account(notes: list[dict[str, Any]], other_label: str = "其他 / 未归类") -> dict[str, Any]:
    """判断账号在写哪几条赛道。按赛道词归并，一条笔记归给权重最高的赛道词。"""
    clean = [note for note in notes if str(note.get("title") or "").strip()]
    if not clean:
        return {"tracks": [], "total": 0, "medianLikes": 0, "terms": []}

    terms = domain_terms(clean, min_df=2, limit=12)
    if not terms:
        return {
            "tracks": [
                {
                    "label": "全部内容",
                    "term": "",
                    "count": len(clean),
                    **_metrics(clean),
                    "samples": _sample(clean),
                    "flags": _flags(clean),
                    "score": 0,
                }
            ],
            "total": len(clean),
            "medianLikes": sorted(int(n.get("likes") or 0) for n in clean)[len(clean) // 2],
            "terms": [],
        }

    # 只保留真正形成规模的主赛道词（覆盖度 >= 8% 或出现 >= 4 次）
    main_terms = [t for t in terms if t["count"] >= max(3, len(clean) * 0.08)][:6] or terms[:3]
    term_set = {t["term"]: t for t in main_terms}

    buckets: dict[str, list[dict[str, Any]]] = {t["term"]: [] for t in main_terms}
    buckets[other_label] = []
    for note in clean:
        tokens = set(tokenize(str(note.get("title") or "")))
        hits = [(term_set[t]["weight"], t) for t in tokens if t in term_set]
        if hits:
            buckets[max(hits)[1]].append(note)
        else:
            buckets[other_label].append(note)

    tracks: list[dict[str, Any]] = []
    for term, items in buckets.items():
        if not items:
            continue
        stats = _metrics(items)
        label = term if term == other_label else term_set[term]["term"]
        # 结构分：中位表现为主、天花板为辅，避免被一条爆款带偏
        score = round((stats["avgLikes"] ** 0.5) * 1.8 + (stats["maxLikes"] ** 0.5) * 0.8, 1)
        tracks.append(
            {
                "label": label,
                "term": "" if term == other_label else term,
                "count": stats["count"],
                "share": round(stats["count"] / len(clean) * 100),
                "avgLikes": stats["avgLikes"],
                "maxLikes": stats["maxLikes"],
                "likes": stats["likes"],
                "ratio": stats["ratio"],
                "samples": _sample(items),
                "flags": _flags(items),
                "score": score,
            }
        )

    tracks.sort(key=lambda t: (-t["score"], -t["count"]))
    return {
        "tracks": tracks,
        "total": len(clean),
        "medianLikes": sorted(int(n.get("likes") or 0) for n in clean)[len(clean) // 2],
        "terms": terms,
        "titles": [str(note.get("title") or "") for note in clean],
    }


def seed_keywords(account: dict[str, Any], limit: int = 4) -> list[str]:
    """用账号主赛道词生成搜索种子。

    实测必须做两件事：
      1. 只用覆盖度够高的「核心赛道词」（zotero / 保研 / 推免），
         否则像「捡漏」这种词在平台上指向的是二手捡漏、话剧捡漏，搜回来全是噪音；
      2. 非核心词要和它在你账号里共现过的核心词拼起来（保住「保研捡漏」这种组合语义），
         单独一个「捡漏」的搜索价值是负的。
    """
    terms = [t for t in account.get("terms", []) if not _NUMERIC.match(t["term"])]
    if not terms:
        return []
    total = max(account.get("total", 0), 1)
    core = [t for t in terms if t["coverage"] >= 15][:limit]
    if not core:
        core = terms[:2]

    seeds: list[str] = []
    for item in core:
        if item["term"] not in seeds:
            seeds.append(item["term"])

    # 剩余名额给「核心词 + 细分词」的组合
    for item in terms:
        if len(seeds) >= limit:
            break
        term = item["term"]
        if term in seeds or any(term in seed for seed in seeds):
            continue
        partner = ""
        for candidate in core:
            if term in candidate["term"]:
                continue
            if _co_occurs(account, candidate["term"], term):
                partner = candidate["term"]
                break
        if partner:
            combined = f"{partner}{term}"
            if combined not in seeds:
                seeds.append(combined)
    return seeds[:limit]


def _co_occurs(account: dict[str, Any], left: str, right: str) -> bool:
    titles = account.get("titles") or []
    return any(left in title and right in title for title in titles)


def core_terms(account: dict[str, Any], coverage: int = 15) -> set[str]:
    """覆盖度够高的赛道词，用来判定一条竞品笔记是不是真的同赛道。"""
    return {
        t["term"] for t in account.get("terms", []) if t["coverage"] >= coverage
    }


def spherical_kmeans(
    vectors: list[dict[str, float]], k: int, iterations: int = 14, seed: int = 42
) -> list[list[int]]:
    """球形 K-means：余弦相似度 + k-means++ 初始化 + 固定随机种子，结果可复现。"""
    n = len(vectors)
    if n == 0:
        return []
    k = max(1, min(k, n))
    rng = random.Random(seed)

    centroids: list[dict[str, float]] = [dict(vectors[rng.randrange(n)])]
    while len(centroids) < k:
        # k-means++：越远离已有中心，越可能被选中
        distances = []
        for vector in vectors:
            best = max(_dot(vector, centroid) for centroid in centroids)
            distances.append(max(1e-6, 1 - best))
        total = sum(distances)
        point = rng.random() * total
        acc = 0.0
        chosen = 0
        for index, distance in enumerate(distances):
            acc += distance
            if acc >= point:
                chosen = index
                break
        centroids.append(dict(vectors[chosen]))

    assignments = [-1] * n
    for _ in range(iterations):
        changed = False
        for index, vector in enumerate(vectors):
            scores = [_dot(vector, centroid) for centroid in centroids]
            best = max(range(len(scores)), key=lambda i: scores[i])
            if assignments[index] != best:
                assignments[index] = best
                changed = True
        # 重算中心并归一化
        buckets: list[list[int]] = [[] for _ in range(k)]
        for index, cluster in enumerate(assignments):
            buckets[cluster].append(index)
        for cluster, members in enumerate(buckets):
            if not members:
                continue
            acc: Counter[str] = Counter()
            for member in members:
                acc.update(vectors[member])
            norm = math.sqrt(sum(v * v for v in acc.values())) or 1.0
            centroids[cluster] = {term: value / norm for term, value in acc.items()}
        if not changed:
            break

    groups: list[list[int]] = [[] for _ in range(k)]
    for index, cluster in enumerate(assignments):
        groups[cluster].append(index)
    return [group for group in groups if group]


def _label(indices: Iterable[int], vectors: list[dict[str, float]], size: int = 3) -> list[str]:
    scores: Counter[str] = Counter()
    for index in indices:
        for term, weight in vectors[index].items():
            scores[term] += weight
    return [term for term, _ in scores.most_common(size) if scores[term] > 0]


def analyze_topics(notes: list[dict[str, Any]], k: int | None = None) -> dict[str, Any]:
    """把一批竞品笔记切成 k 个主题，并判断每个主题的供需关系。"""
    clean = [note for note in notes if str(note.get("title") or "").strip()]
    if not clean:
        return {"topics": [], "total": 0, "medianLikes": 0}

    docs = [tokenize(str(note.get("title") or "") + " " + str(note.get("desc") or "")[:100]) for note in clean]
    vectors = build_tfidf(docs)
    if k is None:
        k = min(9, max(3, round(len(clean) ** 0.5)))
    groups = spherical_kmeans(vectors, k)

    med_likes = sorted(int(n.get("likes") or 0) for n in clean)[len(clean) // 2] or 1
    total = len(clean)
    topics: list[dict[str, Any]] = []

    for indices in groups:
        members = [clean[i] for i in indices]
        stats = _metrics(members)
        supply = stats["count"] / total
        demand = (stats["avgLikes"] / med_likes) ** 0.5
        opportunity = round(min(10.0, demand / (supply * 2 + 0.5)), 2)

        if stats["count"] <= 1:
            verdict, advice = "单点样本", "只有一条样本，先别当结论；要么再多搜几条确认，要么小成本试一次"
        elif demand >= 1.35 and supply <= 0.16:
            verdict, advice = "供给缺口", "需求明显高于赛道中位数、同类内容却很少，优先测试"
        elif demand >= 1.1:
            verdict, advice = "红海但有效", "同类已经很多但确实有量，要写就换人群或换角度，别出同构清单"
        elif demand < 0.75:
            verdict, advice = "需求偏低", "这个角度在这个赛道里不吃香，除非你要的是信任而不是流量"
        else:
            verdict, advice = "稳定跟随", "表现接近赛道中位数，适合作为常规更新维持热度"

        flags = _flags(members)
        topics.append(
            {
                "label": " · ".join(_label(indices, vectors)) or "未分类",
                "terms": _label(indices, vectors, 5),
                "count": stats["count"],
                "share": round(supply * 100),
                "avgLikes": stats["avgLikes"],
                "avgCollections": stats["avgCollections"],
                "maxLikes": stats["maxLikes"],
                "ratio": stats["ratio"],
                "commentRate": stats["commentRate"],
                "verdict": verdict,
                "advice": advice,
                "opportunity": opportunity,
                "flags": flags,
                "samples": _sample(members),
                "noteIds": [str(n.get("id") or "") for n in members],
            }
        )

    # 有样本量的主题排前面：单条样本的机会分再高，也不能排在可执行结论之上
    topics.sort(key=lambda t: (t["count"] < 2, -t["opportunity"], -t["avgLikes"]))
    return {"topics": topics, "total": total, "medianLikes": med_likes, "k": len(groups)}


def curate_pool(
    notes: list[dict[str, Any]],
    core: set[str],
    anchors: set[str],
    own_id: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """清洗竞品池：剔除跑偏的笔记和账号自己的笔记。

    实测必须做：种子词一旦偏掉（例如「捡漏」搜出二手交易、话剧转让），
    后续所有统计和聚类都会被污染。判定规则是「命中核心赛道词」或「命中两个以上锚点词」。
    """
    kept: list[dict[str, Any]] = []
    dropped = 0
    for note in notes:
        if own_id and str(note.get("authorId") or "") == own_id:
            dropped += 1
            continue
        tokens = set(tokenize(str(note.get("title") or "") + " " + str(note.get("desc") or "")[:80]))
        if tokens & core or len(tokens & anchors) >= 2:
            kept.append(note)
        else:
            dropped += 1
    return kept, dropped


def aggregate_authors(
    notes: list[dict[str, Any]], own_id: str = "", limit: int = 12
) -> list[dict[str, Any]]:
    """按作者聚合，形成同赛道博主池。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for note in notes:
        author_id = str(note.get("authorId") or "")
        if not author_id or author_id == own_id:
            continue
        buckets.setdefault(author_id, []).append(note)

    authors: list[dict[str, Any]] = []
    for author_id, items in buckets.items():
        stats = _metrics(items)
        words: Counter[str] = Counter()
        for item in items:
            words.update(set(tokenize(str(item.get("title") or ""))))
        top = max(items, key=lambda n: int(n.get("likes") or 0))
        authors.append(
            {
                "authorId": author_id,
                "author": str(items[0].get("author") or ""),
                "appearances": len(items),
                "likes": stats["likes"],
                "avgLikes": stats["avgLikes"],
                "maxLikes": stats["maxLikes"],
                "ratio": stats["ratio"],
                "keywords": [w for w, _ in words.most_common(4)],
                "bestNote": str(top.get("title") or ""),
                "bestLikes": int(top.get("likes") or 0),
                "token": str(top.get("token") or ""),
                "notes": [
                    {
                        "title": str(n.get("title") or ""),
                        "likes": int(n.get("likes") or 0),
                        "collections": int(n.get("collections") or 0),
                        "token": str(n.get("token") or ""),
                        "id": str(n.get("id") or ""),
                    }
                    for n in sorted(items, key=lambda n: int(n.get("likes") or 0), reverse=True)[:3]
                ],
            }
        )

    # 上榜次数优先：只靠一条爆款的号不该排第一
    authors.sort(key=lambda a: (-a["appearances"], -a["maxLikes"]))
    return authors[:limit]


def diff_titles(mine: list[str], theirs: list[str], limit: int = 8) -> dict[str, list[str]]:
    """对比「我写过」和「同赛道在写」的用词差集。"""
    mine_terms: Counter[str] = Counter()
    for title in mine:
        mine_terms.update(tokenize(title))
    theirs_terms: Counter[str] = Counter()
    for title in theirs:
        theirs_terms.update(tokenize(title))

    gaps = [word for word, _ in theirs_terms.most_common(80) if mine_terms.get(word, 0) == 0]
    unique = [word for word, _ in mine_terms.most_common(80) if theirs_terms.get(word, 0) == 0]
    return {"untapped": gaps[:limit], "onlyMine": unique[:limit]}
