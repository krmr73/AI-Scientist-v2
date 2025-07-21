import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize

# --- 定数 ---
NAME = "qd"
GROUP_NAMES = ["Reflection-only", "Literature-informed", "Proposed"]
COLORS = ["steelblue", "darkorange", "forestgreen"]
UMAP_OUTPUT_PATH = f"../results/{NAME}/umap_3groups.png"
DISTRIBUTION_OUTPUT_PATH = f"../results/{NAME}/distribution_3groups_stacked.png"
HEATMAP_PATHS = [
    f"../results/{NAME}/heatmap_baseline.png",
    f"../results/{NAME}/heatmap_literature.png",
    f"../results/{NAME}/heatmap_proposed.png",
]


# --- ユーティリティ関数 ---
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_idea_text(idea):
    return " ".join(
        [idea.get("Title", "").strip(), idea.get("Short Hypothesis", "").strip(), idea.get("Abstract", "").strip()]
    )


def load_unique_ideas(file_paths):
    seen_ids = set()
    unique_ideas = []
    for path in file_paths:
        for idea in load_json(path):
            idea_id = idea.get("ID")
            if idea_id and idea_id not in seen_ids:
                unique_ideas.append(idea)
                seen_ids.add(idea_id)
    return unique_ideas


def compute_mean_std_within_group(group_embs):
    if len(group_embs) < 2:
        return np.nan, np.nan
    dists = cosine_distances(group_embs)
    upper = dists[np.triu_indices(len(group_embs), k=1)]
    return np.mean(upper), np.std(upper)


def get_pairwise_distances(embeddings):
    dist_matrix = cosine_distances(embeddings)
    return dist_matrix[np.triu_indices(len(embeddings), k=1)]


def plot_umap(reduced, labels):
    plt.figure(figsize=(8, 6))
    for i in range(3):
        subset = reduced[np.array(labels) == i]
        plt.scatter(subset[:, 0], subset[:, 1], label=GROUP_NAMES[i], alpha=0.7, color=COLORS[i])
    plt.title("UMAP of Research Ideas (3 Conditions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(UMAP_OUTPUT_PATH)  # 変数名はそのままでもOK（ファイル名を変えるならここも修正）


def plot_distance_histograms(d0, d1, d2):
    all_dists = np.concatenate([d0, d1, d2])
    bins = np.histogram_bin_edges(all_dists, bins=15)
    ymax = max(np.histogram(d, bins=bins)[0].max() for d in [d0, d1, d2])

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, dist, name, color in zip(axes, [d0, d1, d2], GROUP_NAMES, COLORS):
        ax.hist(dist, bins=bins, color=color, alpha=0.7, edgecolor="grey")
        ax.set_title(name)
        ax.set_ylabel("Density")
        ax.set_ylim(0, ymax)
    axes[2].set_xlabel("Cosine Distance Between Ideas")
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_OUTPUT_PATH)


def plot_heatmap(embeddings, title, save_path):
    dist_matrix = cosine_distances(embeddings)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        dist_matrix,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Cosine Distance"},
        xticklabels=False,
        yticklabels=False,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_elites_number(df, output_path):

    # グラフの作成
    plt.figure(figsize=(10, 6))
    plt.plot(df["generation"], df["num_elites"], label="Number of Elites")
    plt.plot(df["generation"], df["total_ideas"], label="Total Ideas")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.title("Elites and Total Ideas over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


# --- メイン処理 ---

idea_num = 80

# 1. データ読み込み
proposed_ideas = load_json(f"../results/{NAME}/elites/gen_50.json")[:idea_num]

existing_ideas = load_json("ideas/polya_urn_model.json")[:idea_num]
existing_literature_ideas = load_json("ideas/polya_urn_model_with_semanticscholar.json")[: len(proposed_ideas)]

# 2. テキストとラベル準備
texts, labels = [], []

for idea in existing_ideas:
    text = preprocess_idea_text(idea)
    if text.strip():
        texts.append(text)
        labels.append(0)

for idea in existing_literature_ideas:
    text = preprocess_idea_text(idea)
    if text.strip():
        texts.append(text)
        labels.append(1)

for idea in proposed_ideas:
    text = preprocess_idea_text(idea)
    if text.strip():
        texts.append(text)
        labels.append(2)

# 3. 埋め込み
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
print(embeddings.size)
labels = np.array(labels)
embeddings = np.array(embeddings)

# # UMAP による次元削減
# umap_reducer = umap.UMAP(n_components=2, metric="cosine", random_state=0)
# reduced = umap_reducer.fit_transform(embeddings)
# print(reduced)

# plot_umap(reduced, labels)

# 5. グループ分割・距離計算
emb_groups = [embeddings[labels == i] for i in range(3)]
means_stds = [compute_mean_std_within_group(g) for g in emb_groups]

print("✅ Group-wise Mean Pairwise Distances:")
for i, (mean, std) in enumerate(means_stds):
    print(f"  {GROUP_NAMES[i]} (n={len(emb_groups[i])}): Mean = {mean:.4f}, Std = {std:.4f}")

# 6. 距離分布ヒストグラム
dists = [get_pairwise_distances(g) for g in emb_groups]
plot_distance_histograms(*dists)

# 7. ヒートマップ描画
for i, group in enumerate(emb_groups):
    plot_heatmap(group, f"Heatmap ({GROUP_NAMES[i]})", HEATMAP_PATHS[i])


def find_similar_ideas(embeddings, labels, threshold=0.8):
    """
    同一グループ内でコサイン類似度 >= threshold のアイデアペアを列挙。
    返回: {group_name: [(i,j,sim), ...]}
    """
    # 1. L2 正規化（任意だが明示的に）
    emb_norm = normalize(embeddings, norm="l2", axis=1)

    # 2. 類似度行列
    sim_mat = cosine_similarity(emb_norm)

    # 3. 結果格納
    similar_pairs = {g: [] for g in GROUP_NAMES}

    # 4. 上三角だけ走査
    for i, j in zip(*np.triu_indices_from(sim_mat, k=1)):
        sim = sim_mat[i, j]
        if sim >= threshold and labels[i] == labels[j]:
            group = GROUP_NAMES[labels[i]]
            similar_pairs[group].append((i, j, sim))

    return similar_pairs


# 類似度が高いペアをグループごとに抽出
similar_ideas = find_similar_ideas(embeddings, labels, threshold=0.8)

# 提案手法と他の手法との比較結果を表示
print(f"✅ 類似度が{0.8}以上のアイデアペア:")
for group_pair, pairs in similar_ideas.items():
    print(f"{group_pair} の類似ペア数: {len(pairs)}")
    # for i, j, sim in pairs:
    #     print(f"アイデア {i} と アイデア {j} はコサイン類似度 {sim:.4f} で類似しています。")


df = pd.read_csv("../results/qd/qd_history.csv")

# エリート数の推移をプロット
plot_elites_number(df, "../results/qd/elites_number_over_generations.png")
