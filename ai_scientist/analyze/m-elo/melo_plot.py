import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GROUP_JSON = "results/final_group_means.json"
GROUP_STATS_JSON = "results/final_group_stats.json"
MATCHES_JSON = "results/matches_process/combined_matches_latest.json"
OUT_PNG_RANK = "results/figs/melo_ranking.png"
OUT_PNG_WINRATE = "results/figs/pairwise_winrate.png"


def plot_ranking_points_with_ci(group_means, group_se=None, out_png=OUT_PNG_RANK, ci_multiplier=1.96, sort_desc=True):
    items = sorted(group_means.items(), key=lambda x: x[1], reverse=sort_desc)
    labels = [k for k, _ in items]
    vals = np.array([v for _, v in items], float)
    se_vals = np.array([group_se.get(g, np.nan) for g in labels], float) if group_se else np.zeros_like(vals)
    plt.figure(figsize=(len(labels) * 1.2, 5))
    x = np.arange(len(labels))
    plt.errorbar(
        x=x,
        y=vals,
        yerr=ci_multiplier * se_vals,
        fmt="o",
        ecolor="tab:blue",
        elinewidth=2,
        capsize=6,
        markersize=5,
        color="tab:blue",
    )
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("MLE Elo Ratings")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.margins(x=0.2)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pairwise_winrate(win_mat, labels, out_png):
    arr = np.array(win_mat, float)
    for i in range(len(labels)):
        arr[i, i] = float("nan")
    plt.figure(figsize=(6.0, 5.5))
    plt.imshow(arr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Pairwise Win Rate (row beats column)")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    group_means = json.load(open(GROUP_JSON, encoding="utf-8"))
    group_stats = json.load(open(GROUP_STATS_JSON, encoding="utf-8"))
    g_se = {g: v["se"] for g, v in group_stats.items() if v["se"] is not None}

    # 1) ランキング
    plot_ranking_points_with_ci(group_means, group_se=g_se, out_png=OUT_PNG_RANK)

    # 2) 観測ベース勝率ヒートマップ
    matches = json.load(open(MATCHES_JSON, encoding="utf-8"))
    groups_sorted = sorted(group_means.keys())
    idx_of = {g: i for i, g in enumerate(groups_sorted)}
    n = len(groups_sorted)
    win = [[0] * n for _ in range(n)]
    cnt = [[0] * n for _ in range(n)]
    for m in matches:
        ga, gb = m["group_a"], m["group_b"]
        i, j = idx_of[ga], idx_of[gb]
        wr = m.get("winner_raw")
        if wr == "left":
            win[i][j] += 1
            cnt[i][j] += 1
            cnt[j][i] += 1
        elif wr == "right":
            win[j][i] += 1
            cnt[i][j] += 1
            cnt[j][i] += 1
        elif wr == "tie":
            win[i][j] += 0.5
            win[j][i] += 0.5
            cnt[i][j] += 1
            cnt[j][i] += 1
        else:
            s_a = float(m.get("sa_effective", 0.5))
            win[i][j] += s_a
            win[j][i] += 1.0 - s_a
            cnt[i][j] += 1
            cnt[j][i] += 1
    winrate = [
        [None if i == j else (win[i][j] / cnt[i][j] if cnt[i][j] > 0 else None) for j in range(n)] for i in range(n)
    ]
    plot_pairwise_winrate(winrate, groups_sorted, out_png=OUT_PNG_WINRATE)

    print("[OK] 図を出力しました:")
    print(f" - {OUT_PNG_RANK}")
    print(f" - {OUT_PNG_WINRATE}")


if __name__ == "__main__":
    main()
