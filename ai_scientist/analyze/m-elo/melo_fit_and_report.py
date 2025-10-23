import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# ==============================
# 入出力設定
# ==============================
INPUT_PATH = "../results/combined_matches_latest.json"  # 例: .jsonl か、配列JSONなら "matches.json"
OUT_RATINGS_JSON = "../results/final_elo_by_idea.json"
OUT_GROUP_JSON = "../results/final_group_means.json"
OUT_GROUP_STATS_JSON = "../results/final_group_stats.json"  # 追加: CIつき
OUT_DIR_FIG = Path("../results/figs")
OUT_PNG_RANK = OUT_DIR_FIG / "melo_ranking.png"
OUT_PNG_WINRATE = OUT_DIR_FIG / "pairwise_winrate.png"


# ==============================
# ユーティリティ：JSON 読み込み（jsonl / 配列JSON両対応）
# ==============================
def load_matches(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data

    raise ValueError("Unsupported JSON structure. Expect a list or JSONL.")


# ==============================
# m-ELO（MLE）ロジック
# ==============================
def expected_score(r_a, r_b):
    """Eloの期待勝率（base-10 ロジスティック）。"""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def log_likelihood(matches, ratings):
    """現在のratingsにおける対数尤度（定数項は無視）"""
    ll = 0.0
    for m in matches:
        a = (m["group_a"], m["idx_a"])
        b = (m["group_b"], m["idx_b"])
        ra, rb = ratings[a], ratings[b]
        p_a = expected_score(ra, rb)

        wr = m.get("winner_raw")
        if wr == "left":
            s_a = 1.0
        elif wr == "right":
            s_a = 0.0
        elif wr == "tie":
            s_a = 0.5
        else:
            s_a = float(m.get("sa_effective", 0.5))

        # loglik: s*log p + (1-s)*log(1-p)
        # 数値安定のためクリップ
        eps = 1e-12
        p_a = min(max(p_a, eps), 1 - eps)
        ll += s_a * math.log(p_a) + (1 - s_a) * math.log(1 - p_a)
    return ll


def melo_estimate(
    matches,
    init_rating=1000.0,
    lr=0.5,
    epochs=2000,
    anchor_center=True,
    early_stop=True,
    tol=1e-6,
    patience=10,
):
    """
    m-ELO（MLE）で全試合を一括推定。
    戻り値: ratings(dict), history(list[ll])
    """
    items = set()
    for m in matches:
        items.add((m["group_a"], m["idx_a"]))
        items.add((m["group_b"], m["idx_b"]))

    ratings = {k: init_rating for k in items}
    C = math.log(10) / 400.0  # d/dx sigmoid_base10(x) の係数
    history = []
    best_ll = -float("inf")
    stall = 0

    for _ in range(epochs):
        grad = {k: 0.0 for k in items}
        for m in matches:
            a = (m["group_a"], m["idx_a"])
            b = (m["group_b"], m["idx_b"])
            ra, rb = ratings[a], ratings[b]

            wr = m.get("winner_raw")
            if wr == "left":
                s_a = 1.0
            elif wr == "right":
                s_a = 0.0
            elif wr == "tie":
                s_a = 0.5
            else:
                s_a = float(m.get("sa_effective", 0.5))

            p_a = expected_score(ra, rb)
            g = C * (s_a - p_a)
            grad[a] += g
            grad[b] -= g

        for k in ratings.keys():
            ratings[k] += lr * grad[k]

        if anchor_center:
            mean_r = sum(ratings.values()) / len(ratings)
            shift = init_rating - mean_r
            for k in ratings.keys():
                ratings[k] += shift

        # 収束監視（任意）
        ll = log_likelihood(matches, ratings)
        history.append(ll)
        if early_stop:
            if ll > best_ll + tol:
                best_ll = ll
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break

    return ratings, history


# ==============================
# 連結性チェック & ヘッセ行列ベースの不確かさ
# ==============================
def connected_components(items, edges):
    """
    items: list[item_key]
    edges: list[(i_idx, j_idx)]  比較が一度でもあったら辺を張る
    return: list[list[int]]  各連結成分のitemインデックス配列
    """
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
    """
    負のヘッセ行列 (-H) = C^2 * L（Lは重み付きラプラシアン）
    w_ij = sum_over_matches(i,j) [ p(1-p) ]
    戻り値: L（n×n）
    """
    import numpy as np

    C = math.log(10) / 400.0
    n = len(items_index)
    L = np.zeros((n, n), dtype=float)

    # ペア単位でp(1-p)を積算
    for m in matches:
        a = (m["group_a"], m["idx_a"])
        b = (m["group_b"], m["idx_b"])
        ia = items_index[a]
        ib = items_index[b]
        ra, rb = ratings[a], ratings[b]
        p = expected_score(ra, rb)
        w = p * (1 - p)  # s_aに依らず情報量はp(1-p)

        L[ia, ia] += w
        L[ib, ib] += w
        L[ia, ib] -= w
        L[ib, ia] -= w

    # スケールは C^2 * L が -H なので、共分散は (1/C^2) * L^+
    return L, C


def idea_se_from_laplacian(L, C, comps):
    """
    各連結成分ごとにラプラシアンの擬似逆を使って分散を算出。
    戻り値: var_vec (n,), se_vec (n,)
    """
    import numpy as np

    n = L.shape[0]
    var = np.full(n, np.nan, dtype=float)

    for comp in comps:
        # 成分が1点のみなら分散は不定（比較が無い）→ NaN のまま
        if len(comp) <= 1:
            continue
        # 固定アンカー：成分内で1つの自由度があるので、任意に1点を落として逆行列
        idx = np.ix_(comp, comp)
        Lc = L[idx]
        # 成分内の1行1列を削除して可逆化
        Lc_red = Lc[1:, 1:]
        try:
            inv_red = np.linalg.inv(Lc_red)
        except np.linalg.LinAlgError:
            # 数値的に厳しい場合は擬似逆
            inv_red = np.linalg.pinv(Lc_red)

        # 成分内の分散（擬似逆の対角成分に相当）
        # 1点目の分散:
        v0 = 0.0  # アンカー相対なので0。平均センタリングなら相対分散を与える指標
        var[comp[0]] = v0
        # 残り
        diag = np.diag(inv_red)
        for local_i, node in enumerate(comp[1:], start=0):
            var[node] = diag[local_i]

    # スケール調整：Cov = (1/C^2) * L^+
    var = var / (C**2)
    se = np.sqrt(var)
    return var, se


def group_stats_from_item_cov(groups, items_order, var_vec):
    """
    グループ平均の分散 = G * Cov * G^T の対角（相関無視の近似を避けるため、
    ここでは簡易に「同一成分内の共分散=0、成分間はそもそも不定」の前提で
    アイデアの分散のみで近似する場合は、n_g^2 で割った分散和に相当。
    ただし上で出している var は「相対アンカー」のため、成分内の共分散を厳密に
    反映するには完全な擬似逆行列が必要。ここでは実用性重視で対角近似を採用。
    """
    import numpy as np

    # 対角近似（共分散を0とみなす）
    members = defaultdict(list)
    for i, (g, idx) in enumerate(items_order):
        members[g].append(i)

    g_mean = {}
    g_se = {}
    for g, idxs in members.items():
        n = len(idxs)
        # 分散の合計 / n^2
        v = float(np.nansum(var_vec[idxs])) / (n * n)
        g_se[g] = math.sqrt(v) if v == v else float("nan")  # NaN対策
    return g_se


# ==============================
# 図の描画（matplotlibのみ）
# ==============================
def plot_ranking_points_with_ci(
    group_means: dict,
    group_se: dict | None = None,
    out_png: str = "results/melo_pointplot_vertical.png",
    ci_multiplier: float = 1.96,
    sort_desc: bool = True,
):
    """
    棒なしで点と95%CIのエラーバーを表示（縦方向）
    - group_means: {group: mean_elo}
    - group_se: {group: se}（標準誤差）
    - ci_multiplier: 1.96で95%CI相当
    - sort_desc: Trueならレーティング降順に並べる
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # 並べ替え（降順が一般的）
    items = sorted(group_means.items(), key=lambda x: x[1], reverse=sort_desc)
    labels = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float)

    # CI計算
    if group_se is not None:
        se_vals = np.array([group_se.get(g, np.nan) for g in labels], dtype=float)
    else:
        se_vals = np.zeros_like(vals)

    # プロット
    plt.figure(figsize=(len(labels) * 1.2, 5))
    x_pos = np.arange(len(labels))

    plt.errorbar(
        x=x_pos,
        y=vals,
        yerr=ci_multiplier * se_vals,
        fmt="o",
        ecolor="tab:blue",
        elinewidth=2,
        capsize=6,
        markersize=5,
        color="tab:blue",
    )

    # 軸ラベルなど
    plt.xticks(x_pos, labels, rotation=60, ha="right")
    plt.ylabel("MLE Elo Ratings")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.margins(x=0.2)
    plt.tight_layout()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] 縦方向の点＋エラーバーグラフを保存しました: {out_png}")


def plot_pairwise_winrate(win_mat, labels, out_png):
    import numpy as np

    arr = np.array(win_mat, dtype=float)
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


# ==============================
# メイン処理
# ==============================
def run_melo_and_plot(
    input_path=INPUT_PATH,
    out_png_rank=OUT_PNG_RANK,
    out_png_winrate=OUT_PNG_WINRATE,
    out_ratings_json=OUT_RATINGS_JSON,
    out_group_json=OUT_GROUP_JSON,
    out_group_stats_json=OUT_GROUP_STATS_JSON,
    init_rating=1000.0,
    lr=0.5,
    epochs=2000,
):
    matches = load_matches(input_path)

    # m-ELO 推定（アイデア単位）
    ratings, history = melo_estimate(
        matches,
        init_rating=init_rating,
        lr=lr,
        epochs=epochs,
        anchor_center=True,
        early_stop=True,
        tol=1e-6,
        patience=10,
    )

    # 連結性チェック用のインデックス化
    items = sorted(ratings.keys())  # 安定順
    items_index = {k: i for i, k in enumerate(items)}
    edges = []
    for m in matches:
        a = (m["group_a"], m["idx_a"])
        b = (m["group_b"], m["idx_b"])
        edges.append((items_index[a], items_index[b]))
    comps = connected_components(items, edges)
    if len(comps) > 1:
        print(f"[WARN] 比較グラフが {len(comps)} 連結成分に分かれています。成分間の相対オフセットは不定です。")

    # 情報行列（ラプラシアン）からSE算出
    L, C = fisher_information_laplacian(matches, ratings, items_index)
    var_vec, se_vec = idea_se_from_laplacian(L, C, comps)

    # グループ=モデル単位の集計
    group_members = defaultdict(list)
    for (g, idx), r in ratings.items():
        group_members[g].append(((g, idx), r))
    group_means = {g: (sum(r for _, r in v) / len(v)) for g, v in group_members.items()}

    # グループ平均の標準誤差（対角近似）
    g_se = group_stats_from_item_cov(group_members, items, var_vec)

    # JSON出力（アイデア単位）
    Path(out_ratings_json).parent.mkdir(parents=True, exist_ok=True)
    final_by_idea = []
    for (g, idx), r in sorted(ratings.items(), key=lambda x: (x[0][0], x[0][1])):
        i = items_index[(g, idx)]
        final_by_idea.append(
            {
                "group": g,
                "idx": idx,
                "elo": float(r),
                "se": float(se_vec[i]) if se_vec[i] == se_vec[i] else None,  # NaN->None
                "ci95_low": float(r - 1.96 * se_vec[i]) if se_vec[i] == se_vec[i] else None,
                "ci95_high": float(r + 1.96 * se_vec[i]) if se_vec[i] == se_vec[i] else None,
            }
        )
    with open(out_ratings_json, "w", encoding="utf-8") as f:
        json.dump(final_by_idea, f, ensure_ascii=False, indent=2)

    # JSON出力（グループ平均）
    with open(out_group_json, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in group_means.items()}, f, ensure_ascii=False, indent=2)

    # 追加: グループ統計（SE/CIつき）
    group_stats = {}
    for g, mean in group_means.items():
        se = g_se.get(g, float("nan"))
        if se == se:  # not NaN
            group_stats[g] = {
                "elo_mean": float(mean),
                "se": float(se),
                "ci95_low": float(mean - 1.96 * se),
                "ci95_high": float(mean + 1.96 * se),
            }
        else:
            group_stats[g] = {
                "elo_mean": float(mean),
                "se": None,
                "ci95_low": None,
                "ci95_high": None,
            }
    with open(out_group_stats_json, "w", encoding="utf-8") as f:
        json.dump(group_stats, f, ensure_ascii=False, indent=2)

    # 1) ランキング棒グラフ（グループ平均）
    plot_ranking_points_with_ci(group_means, group_se=g_se, out_png=OUT_PNG_RANK)

    # 2) モデル間勝率ヒートマップ（観測ベース）
    groups_sorted = sorted(group_means.keys())
    idx_of = {g: i for i, g in enumerate(groups_sorted)}
    n = len(groups_sorted)
    win = [[0 for _ in range(n)] for __ in range(n)]
    cnt = [[0 for _ in range(n)] for __ in range(n)]

    for m in matches:
        ga, ia = m["group_a"], m["idx_a"]
        gb, ib = m["group_b"], m["idx_b"]
        i = idx_of[ga]
        j = idx_of[gb]
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

    winrate = [[0.0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                winrate[i][j] = None
            else:
                if cnt[i][j] > 0:
                    winrate[i][j] = win[i][j] / cnt[i][j]
                else:
                    winrate[i][j] = None

    plot_pairwise_winrate(winrate, groups_sorted, out_png_winrate)

    print(f"[OK] m-ELO done.")
    print(f" - Idea-level ratings -> {out_ratings_json}")
    print(f" - Group mean ratings -> {out_group_json}")
    print(f" - Group stats (CI)   -> {out_group_stats_json}")
    print(f" - Figure (ranking)   -> {out_png_rank}")
    print(f" - Figure (winrate)   -> {out_png_winrate}")


# スクリプトとして実行
if __name__ == "__main__":
    # 必要に応じてパスやハイパラを変更してください
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
    try:
        run_melo_and_plot(
            input_path=INPUT_PATH,
            out_png_rank=OUT_PNG_RANK,
            out_png_winrate=OUT_PNG_WINRATE,
            out_ratings_json=OUT_RATINGS_JSON,
            out_group_json=OUT_GROUP_JSON,
            init_rating=1000.0,
            lr=0.5,
            epochs=2000,
        )
    except FileNotFoundError as e:
        # 入力が無い環境でもコードをすぐ使えるよう、エラーメッセージを出力
        print(e)
        print("サンプル入力を用意してから再実行してください。")
