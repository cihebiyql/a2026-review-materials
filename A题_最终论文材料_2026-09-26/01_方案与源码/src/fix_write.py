import ast

lines = open('baseline_hinted.py', encoding='utf-8').read().split('\n')
out = []
i = 0
while i < len(lines):
    l = lines[i]
    stripped = l.rstrip()
    # 断裂形态: "... + '" 后跟 "')'" 行
    if stripped.endswith("+ '") and i + 1 < len(lines) and lines[i+1].strip() == "')":
        out.append(stripped[:-1] + "n')")
        i += 2
        continue
    out.append(l)
    i += 1
src = '\n'.join(out)
open('baseline_hinted.py', 'w', encoding='utf-8').write(src)
try:
    ast.parse(src)
    print('fixed ok')
except SyntaxError as e:
    print('FAIL', e.lineno, repr(out[e.lineno-1][:70]) if e.lineno-1 < len(out) else '')
