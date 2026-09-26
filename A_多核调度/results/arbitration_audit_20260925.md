# GPT 大纲评审仲裁·交付包补充复算记录（2026-09-25 深夜）

会话：AI-08（A025）。对象：`../../../../A题_交付数据包_0925夜/`（MANIFEST.csv 1200 行、stats/ 六表）。用途：核验 GPT 大纲评审所引数字是否仍成立；发现的问题已登记 [02_confirmed_results.md](../notes/02_confirmed_results.md) §1.8/§8 与 [问师兄清单](../notes/20260925_问师兄数据确认清单.md)。环境：Git Bash awk。MANIFEST 列：`case,q,N,mk,sp,source`。

## 1. P2 N5 均值复算（与答复 4.270 一致）

```
$ awk -F, '$3==5 && $2==2 && $5!=""{s+=$5; n++} END{print s/n, n}' MANIFEST.csv
4.26969 100
```

## 2. N5 sp>5 计数（与答复 B8"共 14 例"不符 → 问清单 #6）

```
$ awk -F, '$3==5 && $5>5{print $2}' MANIFEST.csv | sort | uniq -c
      4 1     ← P1
     24 2     ← P2
     21 3     ← P3
$ comm -12 <(P2 sp>5 案例列表) <(P3 sp>5 案例列表) | wc -l
19          ← P2∩P3 交集
```

P2 N5 sp>5 前 5 例（来源随列）：008 6.6327(gpu_climb)、084 6.6237(refined2_q2)、023 6.4493(gpu_climb)、017 6.2820(gpu_climb)、092 5.9689(G5TF-X2)。

## 3. N5 胜者来源构成（GPT"mechA 为核心方法"叙事的核验依据）

```
$ awk -F, '$3==5{print $2"\t"$6}' MANIFEST.csv | sort | uniq -c | sort -rn
（P1）climb:P1-LNS 53、session:MERGE-N5 26、archive:v2_cpc 12、audit:strand_n5 7、archive:v3_main 2
（P2）audit:gpu_climb 53、x2:A-P2-G5TF-X2 19、archive:v2_cpc 9、audit:refined2_q2 5、audit:audit_B_best 5、
      session:G5TF-X2 3、audit:refined 3、audit:cross_best 2、archive:v3_main 1
（P3）audit:gpu_climb_p3 36、x2:A-P3-G5TF-X2 17、session:G5TF-X2 9、audit:cross_best 9、archive:v2_cpc 9、
      audit:refined 5、audit:audit_B_best 5、archive:v3_main 4、bxcpu_best 2、refined2 2、refined6/7 各 1
```

mechA 未以本名出现于 source 列；构成与"单一算法"叙事的张力见问清单 #7。

## 4. case_084 复核（F16 依据；旧 8.83/8.79 叙事作废、机制结论成立）

```
$ grep "^case_084" MANIFEST.csv | grep -E ",5,"
case_084,2,5,378550,6.623727380795139,audit:refined2_q2
case_084,3,5,,6.623727380795139,audit:cross_best      ← 23 行空 mk 之一，sp 有效
$ awk -F, 'NR==1 || $1=="case_084"' stats/p3_scache_full.csv
case_084,2..5：mk_cache=mk_nocache（全核数相同），S_Cache=1.0
$ awk -F, 'NR==1 || $1=="case_084"' stats/p3_hit_full.csv
hit_rate=0.0（N2–N5 全部），miss_bytes≈9.9e6–1.4e7
```

结论：084 超线性（6.62）来自调度结构、Cache 零贡献——在新数据下成立；旧"8.8344/8.7942、天花板 8.75"作废。

## 5. S_Cache 分布（F14/F15 与 6.5 起草口径的新数）

```
$ awk -F, '$2==5 && NR>1{print $5}' stats/p3_scache_full.csv | sort -n | awk '{a[NR]=$1} END{...}'
N5: n=100, median=1.0038/1.0039, min=1.0, max=2.1481
$ awk -F, '$2==5 && $5==1.0{c++} END{print c}' stats/p3_scache_full.csv
30      ← S_Cache 恰好=1.0 的例数（旧 CSV 层为 38）
top（N5）：044 2.1481 / 080 1.5955 / 090 1.5205 / 046 1.5182 / 039 1.3873
```

## 6. 包内文件新旧甄别

- `stats/speedup_stats.csv`：**陈旧**（问题 1 N5 均值 2.583、问题 2 N5 均值 2.942、问题 3 相对增益仅到 ×1.018），与 MANIFEST/答复成绩单完全不符，不得引用；
- `speedup_stats_v4.csv`（包根）：与答复成绩单对应，但 P1N5 3.709/3.7048、P2N4 3.459/3.5509、P3N4 3.525/3.6367 三处与 MANIFEST 不一致（落地核验已记，按 RES-010 以 MANIFEST 为准）；
- `stats/p12_eval_full.csv`：列 `case,q,N,mk,added_MB`——P1/P2 逐例 mk＋额外搬运，可支撑 F17/TBL-004/005 的搬运字段（此前以为只有 P3 有 added）。

## 7. 局限

静态复算（文件内数值聚合与交叉比对），未重跑官方评估器；MANIFEST 为 2026-09-25 夜版本，N2/N3 LNS 增量后以新 MANIFEST 为准。
