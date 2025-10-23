import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# --- 定数 ---
GROUP_NAMES = ["Reflection-only", "Literature-informed", "Proposed"]
COLORS = ["steelblue", "darkorange", "forestgreen"]
UMAP_OUTPUT_PATH = "../results/umap_3groups.png"
DISTRIBUTION_OUTPUT_PATH = "../results/distribution_3groups_stacked.png"
HEATMAP_PATHS = [
    "../results/heatmap_baseline.png",
    "../results/heatmap_literature.png",
    "../results/heatmap_proposed.png",
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


# --- メイン処理 ---

# 1. データ読み込み
proposed_paths = [
    "../results/elites/gen_30_Feasibility.json",
    "../results/elites/gen_30_Interestingness.json",
    "../results/elites/gen_30_Novelty.json",
]
proposed_ideas = load_unique_ideas(proposed_paths)
with open("../results/elites/unique.json", "w", encoding="utf-8") as f:
    json.dump(proposed_ideas, f, ensure_ascii=False, indent=4)

existing_ideas = load_json("ideas/polya_urn_model.json")[: len(proposed_ideas)]
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


# # 4. t-SNE 可視化
# tsne = TSNE(n_components=2, perplexity=min(30, max(2, len(embeddings) // 3)), random_state=0)
# reduced = tsne.fit_transform(embeddings)
# plot_tsne(reduced, labels)

# UMAP による次元削減
umap_reducer = umap.UMAP(n_components=2, metric="cosine", random_state=0)
reduced = umap_reducer.fit_transform(embeddings)
print(reduced)

plot_umap(reduced, labels)

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
    コサイン類似度が指定した閾値以上のアイデアペアを抽出し、その個数とペアを返す。
    グループごとに類似度を計算し、結果を表示します。
    """
    dist_matrix = cosine_distances(embeddings)
    similar_pairs = {group: [] for group in GROUP_NAMES}

    # 上三角行列を取り出し、コサイン類似度がthreshold以上のペアを探す
    for i, j in zip(*np.triu_indices_from(dist_matrix, k=1)):
        sim = 1 - dist_matrix[i, j]  # コサイン類似度
        label_i = labels[i]
        label_j = labels[j]

        # 提案手法と他の手法（Reflection-only, Literature-informed）間の比較
        if sim >= threshold:
            group_i = GROUP_NAMES[label_i]
            group_j = GROUP_NAMES[label_j]
            if group_i != group_j:
                similar_pairs[f"{group_i} vs {group_j}"].append((i, j, sim))

    return similar_pairs


# 類似度が高いペアをグループごとに抽出
similar_ideas = find_similar_ideas(embeddings, labels, threshold=0.8)

# 提案手法と他の手法との比較結果を表示
print(f"✅ 類似度が{0.8}以上のアイデアペア:")
for group_pair, pairs in similar_ideas.items():
    print(f"{group_pair} の類似ペア数: {len(pairs)}")
    for i, j, sim in pairs:
        print(f"アイデア {i} と アイデア {j} はコサイン類似度 {sim:.4f} で類似しています。")
