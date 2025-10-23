import collections
import json
import math
import os
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt

OUT_PREFIX = "results/figs/posthoc"  # 出力先のプレフィックス（自由に変更可）
os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)
path = "results/elo_combined_matches__.json"
matches = json.load(open(path, "r"))

pair_n = collections.Counter()
pair_w = collections.Counter()

for m in matches:
    a, b = m["group_a"], m["group_b"]
    key = tuple(sorted([a, b]))
    pair_n[key] += 1
    pair_w[key] += m["sa_effective"]  # 引分=0.5扱い

print("Pairwise coverage & 95% CI (sorted A vs B, A視点勝率)")
for g1, g2 in sorted(pair_n):
    n = pair_n[(g1, g2)]
    p = pair_w[(g1, g2)] / n
    se = (p * (1 - p) / n) ** 0.5
    lo, hi = p - 1.96 * se, p + 1.96 * se
    flag = "OK" if (hi < 0.5 or lo > 0.5) else "UNCERTAIN"
    print(f"{g1} vs {g2}: n={n}, p={p:.3f}, 95%CI=({lo:.3f},{hi:.3f}) [{flag}]")

print(f"Total matches: {len(matches)}")


# groupごとのelo値を計算
def compute_elo(matches, k=32, initial_elo=1500):
    elos = collections.defaultdict(lambda: initial_elo)

    for m in matches:
        a, b = m["group_a"], m["group_b"]
        sa, sb = m["sa_effective"], 1 - m["sa_effective"]

        ea = 1 / (1 + 10 ** ((elos[b] - elos[a]) / 400))
        eb = 1 / (1 + 10 ** ((elos[a] - elos[b]) / 400))

        elos[a] += k * (sa - ea)
        elos[b] += k * (sb - eb)

    return elos


def wilson_ci(k, n, z=1.96):
    """Wilson 95%CI（勝ち数k/試行n） -> (lo, hi, p) を返す"""
    if n == 0:
        return (0.0, 1.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half), p)


# ---- Elo収束図（各グループ平均Eloの推移） -----------------------------------
def collect_groups_and_items_for_convergence(matches):
    """全試合から group と (group, idx) の全集合を収集（安定順）"""
    groups = OrderedDict()
    for m in matches:
        ga, ia = m["group_a"], m["idx_a"]
        gb, ib = m["group_b"], m["idx_b"]
        groups.setdefault(ga, set()).add((ga, ia))
        groups.setdefault(gb, set()).add((gb, ib))
    groups_items = {g: sorted(list(s)) for g, s in groups.items()}
    return list(groups_items.keys()), groups_items


def compute_elo_convergence_from_logs(matches, groups, groups_items, initial=1000.0):
    """
    ログの ra_after/rb_after を使って、各アイテムのEloを逐次更新しつつ、
    各グループの平均Eloを時系列で記録する。
    """
    elo = {}
    for g, items in groups_items.items():
        for key in items:
            elo[key] = initial

    series = {g: [] for g in groups}
    for t, m in enumerate(matches, start=1):
        ga, ia = m["group_a"], m["idx_a"]
        gb, ib = m["group_b"], m["idx_b"]
        # ログの after 値で2アイテムを更新
        elo[(ga, ia)] = float(m["ra_after"])
        elo[(gb, ib)] = float(m["rb_after"])
        # 各グループ平均を記録
        for g in groups:
            vals = [elo[key] for key in groups_items[g]]
            series[g].append(sum(vals) / len(vals))
    return series


def save_convergence_csv(series, out_csv):
    groups = list(series.keys())
    T = max(len(v) for v in series.values())
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("t," + ",".join(groups) + "\n")
        for t in range(T):
            row = [str(t + 1)]
            for g in groups:
                row.append(f"{series[g][t]:.6f}")
            f.write(",".join(row) + "\n")


def plot_convergence(series, out_png):
    plt.figure(figsize=(7.0, 4.0))
    T = max(len(v) for v in series.values())
    xs = list(range(1, T + 1))
    for g, ys in series.items():
        plt.plot(xs[: len(ys)], ys, label=g)  # 色は指定しない
    plt.xlabel("Matches")
    plt.ylabel("Mean Elo")
    plt.title("Elo Convergence by Group")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ---- ペア勝率バーグラフ（95%CI付き） ----------------------------------------
def compute_pairwise_rows(matches):
    """
    ペア（A,BでA<Bの辞書順）ごとに、A視点の勝率pと95%CI、nを算出して行データ化。
    重要: sa_effective は常に group_a 視点なので、A!=group_a の場合は 1 - sa_effective に反転。
    """
    pair_n = defaultdict(int)
    pair_k = defaultdict(float)  # 勝ち数（引分=0.5）
    for m in matches:
        ga, gb = m["group_a"], m["group_b"]
        sa = float(
            m.get("sa_effective", 1.0 if m["winner_raw"] == "left" else 0.0 if m["winner_raw"] == "right" else 0.5)
        )
        A, B = sorted([ga, gb])
        sA = sa if ga == A else (1.0 - sa)  # A視点に変換
        pair_n[(A, B)] += 1
        pair_k[(A, B)] += sA

    rows = []
    for A, B in sorted(pair_n.keys()):
        n = pair_n[(A, B)]
        k = pair_k[(A, B)]
        lo, hi, p = wilson_ci(k, n, z=1.96)
        rows.append({"pair": f"{A} vs {B}", "A": A, "B": B, "n": n, "p": p, "ci_lo": lo, "ci_hi": hi})
    return rows


def save_pairwise_csv(rows, out_csv):
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("pair,A,B,n,p,ci_lo,ci_hi\n")
        for r in rows:
            f.write(f"{r['pair']},{r['A']},{r['B']},{r['n']},{r['p']:.6f},{r['ci_lo']:.6f},{r['ci_hi']:.6f}\n")


def plot_pairwise_bars(rows, out_png, sort_key=None):
    """
    sort_key: 並び替え用に 'n' や 'absdiff'（= |p-0.5| 降順）などを指定可能
    """
    if sort_key == "n":
        rows = sorted(rows, key=lambda r: r["n"], reverse=True)
    elif sort_key == "absdiff":
        rows = sorted(rows, key=lambda r: abs(r["p"] - 0.5), reverse=True)

    labels = [r["pair"] for r in rows]
    ps = [r["p"] for r in rows]
    yerr_lower = [r["p"] - r["ci_lo"] for r in rows]
    yerr_upper = [r["ci_hi"] - r["p"] for r in rows]
    yerr = [yerr_lower, yerr_upper]

    plt.figure(figsize=(max(6.5, 0.4 * len(labels) + 2), 4.0))
    xs = list(range(len(labels)))
    plt.bar(xs, ps)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.errorbar(xs, ps, yerr=yerr, fmt="none", linewidth=1, capsize=3)
    plt.ylim(0.0, 1.0)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("Win rate (A-side)")
    plt.title("Pairwise Win Rates with 95% CI")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


elos = compute_elo(matches)
print("\nElo ratings:")
for g, e in sorted(elos.items()):
    print(f"  {g}: {e:.1f}")
mean_elo = sum(elos.values()) / len(elos)
print(f"Mean Elo: {mean_elo:.1f}")
print("Elo differences from mean:")
for g, e in sorted(elos.items()):
    print(f"  {g}: {e - mean_elo:+.1f}")

# 1) Elo収束図
_groups, _groups_items = collect_groups_and_items_for_convergence(matches)
_series = compute_elo_convergence_from_logs(matches, _groups, _groups_items, initial=1000.0)
save_convergence_csv(_series, OUT_PREFIX + "_convergence.csv")
plot_convergence(_series, OUT_PREFIX + "_convergence.png")

# 2) ペア勝率バー
_rows = compute_pairwise_rows(matches)
save_pairwise_csv(_rows, OUT_PREFIX + "_pairwise.csv")
# 並び順は必要に応じて 'n' や 'absdiff' に変更可
plot_pairwise_bars(_rows, OUT_PREFIX + "_pairwise_winrate.png", sort_key="n")

print("[plots]")
print(" -", OUT_PREFIX + "_convergence.png")
print(" -", OUT_PREFIX + "_pairwise_winrate.png")
print("[csv]")
print(" -", OUT_PREFIX + "_convergence.csv")
print(" -", OUT_PREFIX + "_pairwise.csv")
