# -*- coding: utf-8 -*-
"""重点加粗密度检查（D051，终检 10 号 D 节调用；18 号 A3.6 起草口径对照）。

口径（D051，依据 notes/20260926_国一正文加粗调查.md，33 篇 A 题国一实测）：
- 统计 \textbf{...} 与 {\bfseries ...}/{\bf ...} 的中文加粗处（含 CJK 判定）；
  纯拉丁/数字内容（数学变量、英文标识符）单列、不计入密度。
- 密度 = 中文加粗处 / 正文页数；目标区间 0.5–1.5 处/页。
- 08_references.tex（参考文献）不计入；论文.tex 只统计本体内联文本（input/include 的
  行不重复计数）。
- 页数自动取 build/ 下 PDF（PyMuPDF；用 anaconda python 运行），失败或无 PDF 时用
  --pages N 手工给定，两者皆缺则只报计数不判档。

用法（Git Bash）：
  /c/<anaconda路径>/python.exe scripts/check_bold_density.py [--pages N] [--list]
"""
import argparse
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TARGET_RANGE = (0.5, 1.5)
CJK = re.compile(u'[\u4e00-\u9fff]')


def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        if line[i] == '%' and (i == 0 or line[i-1] != '\\'):
            break
        out.append(line[i])
        i += 1
    return ''.join(out)


def match_brace(s, start):
    """s[start] == '{'，返回配对 '}' 的索引；不配对返回 -1。"""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{' and (i == 0 or s[i-1] != '\\'):
            depth += 1
        elif s[i] == '}' and s[i-1] != '\\':
            depth -= 1
            if depth == 0:
                return i
    return -1


def find_bolds(text):
    """返回 [(内容, 来源标记)]；来源标记 textbf/bfseries。"""
    hits = []
    for m in re.finditer(r'\\textbf\s*', text):
        i = m.end()
        if i < len(text) and text[i] == '{':
            j = match_brace(text, i)
            if j > 0:
                hits.append((text[i+1:j], 'textbf'))
    for m in re.finditer(r'\{\\bfseries\s|\{\\bf\s|\{\\bfseries\{|\{\\bfseries$', text):
        i = m.start()
        j = match_brace(text, i)
        if j > 0:
            inner = text[i+1:j]
            hits.append((inner, 'bfseries'))
    return hits


def collect(path, is_main):
    """统计单文件；主入口文件剔除 input/include/includepdf 行避免重复计数。"""
    cjk_hits, lat_hits = [], []
    with io.open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = strip_comment(raw)
            if is_main and re.search(r'\\(input|include|includepdf)\b', line):
                continue
            for content, src in find_bolds(line):
                if 'TODO' in content or '\\pending' in content:
                    continue  # 占位标记不计（终检工程门另行清零）
                (cjk_hits if CJK.search(content) else lat_hits).append(content.strip())
    return cjk_hits, lat_hits


def pdf_pages(build_dir):
    try:
        import fitz
    except ImportError:
        return None
    pdfs = sorted(glob.glob(os.path.join(build_dir, '*.pdf')),
                  key=os.path.getmtime, reverse=True)
    if not pdfs:
        return None
    try:
        return len(fitz.open(pdfs[0]))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pages', type=int, default=None, help='正文页数（不自动读 PDF 时给定）')
    ap.add_argument('--list', action='store_true', help='逐处列出中文加粗内容（落点核对用）')
    args = ap.parse_args()

    files = [(os.path.join(ROOT, '论文.tex'), True)]
    files += [(p, False) for p in sorted(glob.glob(os.path.join(ROOT, 'paper', 'sections', '*.tex')))
              if not p.endswith('08_references.tex')]
    files += [(p, False) for p in sorted(glob.glob(os.path.join(ROOT, 'paper', 'appendix', '*.tex')))]

    total_cjk = total_lat = 0
    rows = []
    for path, is_main in files:
        if not os.path.exists(path):
            continue
        cjk, lat = collect(path, is_main)
        total_cjk += len(cjk)
        total_lat += len(lat)
        if cjk or lat:
            rows.append((os.path.relpath(path, ROOT), len(cjk), len(lat), cjk))

    print('文件                                   中文加粗处  拉丁加粗处')
    for name, nc, nl, _ in rows:
        print('%-46s %6d %8d' % (name, nc, nl))
    print('-' * 66)
    print('合计：中文加粗 %d 处，拉丁加粗 %d 处（拉丁不计入密度）' % (total_cjk, total_lat))

    pages = args.pages if args.pages else pdf_pages(os.path.join(ROOT, 'build'))
    if pages:
        d = float(total_cjk) / pages
        verdict = '偏轻（<0.5，可按落点清单增补）' if d < TARGET_RANGE[0] else (
            '偏重（>1.5，按 D051 减量）' if d > TARGET_RANGE[1] else '达标（0.5–1.5）')
        print('正文页数 %d → 中文加粗密度 %.2f 处/页 → %s' % (pages, d, verdict))
    else:
        print('（未取到页数：无 build PDF 且未给 --pages；仅报计数）')

    if args.list:
        print('\n中文加粗逐处清单（终检落点核对：方法名首现/关键结论+核心数字/段首引导词/注意句）：')
        for name, nc, nl, cjk in rows:
            for c in cjk:
                print('  [%s] %s' % (name, c))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    main()
