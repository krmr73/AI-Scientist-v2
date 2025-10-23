import json
import math
from pathlib import Path

INPUT_PATH = "results/matches_process/combined_matches_latest.json"
OUT_RATINGS_JSON = "results/final_elo_by_idea.json"
OUT_GROUP_JSON = "results/final_group_means.json"
OUT_GROUP_STATS_JSON = "results/final_group_stats.json"


# ==== IO ====
def load_matches(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure. Expect a list or JSONL.")


# ==== m-ELO ====
def expected_score(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def log_likelihood(matches, ratings):
    ll = 0.0
    eps = 1e-12
    for m in matches:
        a = (m["group_a"], m["idx_a"])
        b = (m["group_b"], m["idx_b"])
        ra, rb = ratings[a], ratings[b]
        p_a = expected_score(ra, rb)
        wr = m.get("winner_raw")
        s_a = (
            1.0
            if wr == "left"
            else 0.0 if wr == "right" else 0.5 if wr == "tie" else float(m.get("sa_effective", 0.5))
        )
        p_a = min(max(p_a, eps), 1 - eps)
        ll += s_a * math.log(p_a) + (1 - s_a) * math.log(1 - p_a)
    return ll


def melo_estimate(
    matches, init_rating=1000.0, lr=0.5, epochs=2000, anchor_center=True, early_stop=True, tol=1e-6, patience=10
):
    C = math.log(10) / 400.0
    items = {(m["group_a"], m["idx_a"]) for m in matches} | {(m["group_b"], m["idx_b"]) for m in matches}
    ratings = {k: init_rating for k in items}
    history, best_ll, stall = [], -float("inf"), 0
    for _ in range(epochs):
        grad = {k: 0.0 for k in items}
        for m in matches:
            a = (m["group_a"], m["idx_a"])
            b = (m["group_b"], m["idx_b"])
            ra, rb = ratings[a], ratings[b]
            wr = m.get("winner_raw")
            s_a = (
                1.0
                if wr == "left"
                else 0.0 if wr == "right" else 0.5 if wr == "tie" else float(m.get("sa_effective", 0.5))
            )
            p_a = expected_score(ra, rb)
            g = (math.log(10) / 400.0) * (s_a - p_a)
            grad[a] += g
            grad[b] -= g
        for k in ratings:
            ratings[k] += lr * grad[k]
        if anchor_center:
            mean_r = sum(ratings.values()) / len(ratings)
            shift = init_rating - mean_r
            for k in ratings:
                ratings[k] += shift
        ll = log_likelihood(matches, ratings)
        history.append(ll)
        if early_stop:
            if ll > best_ll + tol:
                best_ll, stall = ll, 0
            else:
                stall += 1
                if stall >= patience:
                    break
    return ratings, history


# ==== 連結・SE ====
def connected_components(items, edges):
    n = len(items)
    g = [[] for _ in range(n)]
    for i, j in edges:
        g[i].append(j)
        g[j].append(i)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for to in g[v]:
                if not seen[to]:
                    seen[to] = True
                    stack.append(to)
        comps.append(comp)
    return comps


def fisher_information_laplacian(matches, ratings, items_index):
    import math

    import numpy as np

    C = math.log(10) / 400.0
    n = len(items_index)
    L = np.zeros((n, n), float)
    for m in matches:
        a = (m["group_a"], m["idx_a"])
        b = (m["group_b"], m["idx_b"])
        ia, ib = items_index[a], items_index[b]
        ra, rb = ratings[a], ratings[b]
        p = expected_score(ra, rb)
        w = p * (1 - p)
        L[ia, ia] += w
        L[ib, ib] += w
        L[ia, ib] -= w
        L[ib, ia] -= w
    return L, C


def idea_se_from_laplacian(L, C, comps):
    import numpy as np

    n = L.shape[0]
    var = np.full(n, np.nan)
    for comp in comps:
        if len(comp) <= 1:
            continue
        Lc = L[np.ix_(comp, comp)]
        Lc_red = Lc[1:, 1:]
        try:
            inv_red = np.linalg.inv(Lc_red)
        except np.linalg.LinAlgError:
            inv_red = np.linalg.pinv(Lc_red)
        var[comp[0]] = 0.0
        diag = np.diag(inv_red)
        for local_i, node in enumerate(comp[1:], start=0):
            var[node] = diag[local_i]
    var = var / (C**2)
    se = var**0.5
    return var, se


def group_stats_from_item_cov(groups, items_order, var_vec):
    import math as _m
    from collections import defaultdict

    import numpy as np

    members = defaultdict(list)
    for i, (g, idx) in enumerate(items_order):
        members[g].append(i)
    g_se = {}
    for g, idxs in members.items():
        n = len(idxs)
        v = float(np.nansum(var_vec[idxs])) / (n * n)
        g_se[g] = _m.sqrt(v) if v == v else float("nan")
    return g_se


# ==== メイン（推定→JSON出力のみ） ====
def main():
    matches = load_matches(INPUT_PATH)
    ratings, history = melo_estimate(matches, init_rating=1000.0, lr=0.5, epochs=2000)

    items = sorted(ratings.keys())
    items_index = {k: i for i, k in enumerate(items)}
    edges = [(items_index[(m["group_a"], m["idx_a"])], items_index[(m["group_b"], m["idx_b"])]) for m in matches]
    comps = connected_components(items, edges)
    if len(comps) > 1:
        print(f"[WARN] 比較グラフが {len(comps)} 成分に分割：成分間オフセットは不定")

    L, C = fisher_information_laplacian(matches, ratings, items_index)
    var_vec, se_vec = idea_se_from_laplacian(L, C, comps)

    from collections import defaultdict

    group_members = defaultdict(list)
    for (g, idx), r in ratings.items():
        group_members[g].append(((g, idx), r))
    group_means = {g: (sum(r for _, r in v) / len(v)) for g, v in group_members.items()}
    g_se = group_stats_from_item_cov(group_members, items, var_vec)

    Path(OUT_RATINGS_JSON).parent.mkdir(parents=True, exist_ok=True)
    final_by_idea = []
    for (g, idx), r in sorted(ratings.items(), key=lambda x: (x[0][0], x[0][1])):
        i = items_index[(g, idx)]
        se = float(se_vec[i]) if se_vec[i] == se_vec[i] else None
        final_by_idea.append(
            {
                "group": g,
                "idx": idx,
                "elo": float(r),
                "se": se,
                "ci95_low": float(r - 1.96 * se) if se is not None else None,
                "ci95_high": float(r + 1.96 * se) if se is not None else None,
            }
        )
    with open(OUT_RATINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_by_idea, f, ensure_ascii=False, indent=2)
    with open(OUT_GROUP_JSON, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in group_means.items()}, f, ensure_ascii=False, indent=2)
    group_stats = {}
    for g, mean in group_means.items():
        se = g_se.get(g, float("nan"))
        group_stats[g] = {
            "elo_mean": float(mean),
            "se": None if se != se else float(se),
            "ci95_low": None if se != se else float(mean - 1.96 * se),
            "ci95_high": None if se != se else float(mean + 1.96 * se),
        }
    with open(OUT_GROUP_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(group_stats, f, ensure_ascii=False, indent=2)

    print("[OK] 推定完了:")
    print(f" - Idea ratings -> {OUT_RATINGS_JSON}")
    print(f" - Group means  -> {OUT_GROUP_JSON}")
    print(f" - Group stats  -> {OUT_GROUP_STATS_JSON}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e)
        print("サンプル入力を用意してから再実行してください。")
