import json
from topic_engine import analyze_topics


titles = [
    "Zotero 文献管理与标签分类",
    "Zotero 文献管理与去重",
    "Zotero 文献管理与引文导出",
    "组会汇报的论文方法讲解",
    "组会汇报的论文结果讲解",
    "组会汇报的论文提纲与练习",
]
notes = [{"title": title, "likes": i + 5, "collections": i + 2} for i, title in enumerate(titles)]
result = analyze_topics(notes, k=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
