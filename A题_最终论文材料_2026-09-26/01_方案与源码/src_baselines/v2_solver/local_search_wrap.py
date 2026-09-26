"""局部搜索快照导出（Spearman 实验用）：在搜索轨迹上取 3 个中间切点。"""
from pipeline import local_search  # noqa: E402  (复用同一实现)


def ls_snapshot(ev, cuts, kappa, iters=100):
    """返回搜索过程中的 3 个中间最优切点向量（不含起点、含终点）。"""
    import copy
    import random
    rng = random.Random(0)
    cur = list(cuts)
    cur_mk, _ = ev.evaluate(cur, kappa=kappa)
    best, best_mk = list(cur), cur_mk
    snaps = []
    sigma = max(4, ev.n // 64)
    points = {int(iters * f) for f in (0.35, 0.7, 1.0)}
    for it in range(1, iters + 1):
        cand = list(cur)
        K = len(cand) - 1
        move = rng.randrange(4)
        if move == 0 and K >= 2:
            j = rng.randrange(1, K)
            delta = rng.choice([-1, 1]) * rng.randint(1, sigma)
            new = cand[j] + delta
            if cand[j - 1] + 1 <= new <= cand[j + 1] - 1:
                cand[j] = new
            else:
                continue
        elif move == 1 and K >= 2:
            weights = [ev.seg_weight(cand, j) for j in range(K)]
            j = weights.index(max(weights))
            a, b = cand[j], cand[j + 1]
            if b - a < 16:
                continue
            mid = min(range(a + 8, b - 8), key=lambda p: ev.cross[p - 1])
            cand = cand[:j + 1] + [mid] + cand[j + 1:]
        elif move == 2 and K >= 3:
            weights = [ev.seg_weight(cand, j) for j in range(K)]
            j = weights.index(min(weights))
            m = j if j < K - 1 else j - 1
            cand = cand[:m + 1] + cand[m + 2:]
        else:
            if len(cand) <= 2:
                continue
            j = rng.randrange(1, len(cand) - 1)
            delta = rng.choice([-1, 1]) * rng.randint(1, 2 * sigma)
            new = cand[j] + delta
            if cand[j - 1] + 1 <= new <= cand[j + 1] - 1:
                cand[j] = new
            else:
                continue
        try:
            mk, _ = ev.evaluate(cand, kappa=kappa)
        except RuntimeError:
            continue
        if mk < cur_mk or rng.random() < 0.05:
            cur, cur_mk = cand, mk
            if mk < best_mk:
                best, best_mk = list(cand), mk
        if it in points:
            snaps.append(list(best))
    if not snaps:
        snaps.append(list(best))
    return snaps
