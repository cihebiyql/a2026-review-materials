# -*- coding: utf-8 -*-
# 抽取全部A题国一PDF全文并统计去AI味候选词频（校准语料扩充：n=3 -> n=33）
import csv, os, re, json
import fitz  # PyMuPDF

BASE = r"C:/Users/lirui/Desktop/jingsai"
CSV = BASE + r"/获奖论文/论文目录.csv"
OUT = r"C:/Users/lirui/AppData/Local/Temp/deai_corpus"
os.makedirs(OUT, exist_ok=True)

KEYWORDS = [
    # A档候选
    "值得注意的是","值得一提的是","值得一提","需要指出的是","需要说明的是",
    "毋庸置疑","众所周知","毫无疑问","总而言之","简而言之",
    "赋能","助力","抓手","闭环","有力支撑","有力支持","奠定","扮演","发挥着","进一步地",
    # B档候选
    "综上所述","综上","首先","其次","最后","此外","与此同时","不仅",
    "由此可见","可以看出","不难发现","不难看出","显然",
    "有效地","高效地","显著提升","大幅提升","极大地",
    "旨在","深入","充分利用","综合考虑","优异","先进",
    "本节","本小节","我们","下面","接下来","进而","事实上","实际上",
    "完善","强大","成熟","新颖",
]

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8-sig"))
        if r["题号"] == "A" and r["奖项"] == "国一"]
print(f"A题国一共 {len(rows)} 篇")

stats = {k: [0, 0, 0, 0.0] for k in KEYWORDS}  # total, papers_using, max_per_paper, max_density_per万字
kept, dropped = [], []
for row in rows:
    pdf = BASE + "/" + row["文件路径"].replace("\\", "/")
    name = os.path.basename(row["文件路径"]).replace(".pdf", "")
    if not os.path.exists(pdf):
        dropped.append((name, "MISSING")); continue
    try:
        doc = fitz.open(pdf)
        text = "".join(p.get_text() for p in doc)
        doc.close()
    except Exception as e:
        dropped.append((name, f"ERR:{e}")); continue
    open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8").write(text)
    nzh = len(re.findall(r"[\u4e00-\u9fff]", text))
    if nzh < 5000:  # 扫描件/无文本层，剔除
        dropped.append((name, f"LOW_TEXT zh={nzh}")); continue
    w = nzh / 10000.0
    kept.append((name, nzh))
    for k in KEYWORDS:
        c = text.count(k)
        if c:
            s = stats[k]
            s[0] += c; s[1] += 1; s[2] = max(s[2], c)
            s[3] = max(s[3], c / w)

print(f"有效 {len(kept)} 篇，剔除 {len(dropped)} 篇")
for n, r in dropped: print("  DROP", n, r)
tot_zh = sum(n for _, n in kept)
print(f"语料总量 {tot_zh} 汉字 ≈ {tot_zh/10000:.1f} 万字\n")
print(f"{'词条':<8}\t总次\t出现篇数\t单篇最大\t最大密度/万字")
for k in KEYWORDS:
    s = stats[k]
    if s[0]:
        print(f"{k:<8}\t{s[0]}\t{s[1]}\t{s[2]}\t{s[3]:.2f}")
    else:
        print(f"{k:<8}\t0\t0\t0\t0.00")
json.dump({k: stats[k] for k in KEYWORDS}, open(OUT + "/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
