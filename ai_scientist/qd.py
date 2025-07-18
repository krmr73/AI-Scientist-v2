import argparse
import json
import os
from typing import Dict, List, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import umap
from archive import SimpleGridArchive
from generate_ideas import mutate_ideas, pairwise_evaluate
from sentence_transformers import SentenceTransformer

from ai_scientist.llm import AVAILABLE_LLMS, create_client

NORMALIZED_MIN = 1
NORMALIZED_MAX = 20
MAP_RESOLUTION = 20  # QDマップの解像度（20x20）
MARGIN_RATIO = 0.5


def export_elites(archive, idea_lookup, out_path):
    """
    QDアーカイブに残っているエリートのアイデアをJSONとして保存
    """
    elites = [idea_lookup[idea_id] for idea_id in archive.all_idea_ids()]
    print(f"Archive[{archive.name}]: {archive.all_idea_ids()}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(elites, f, ensure_ascii=False, indent=4)


def save_all_ideas(all_ideas, path):
    """
    すべての生成済みアイデア（淘汰含む）を1つのファイルに保存
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_ideas, f, ensure_ascii=False, indent=4)


def save_qd_map_image(archive: SimpleGridArchive, out_path: str):
    """
    QDマップをヒートマップとして画像保存する。
    各セルの値は quality （最適化対象のスコア）
    """
    grid_data = np.full((archive.resolution, archive.resolution), np.nan)

    for (x, y), entry in archive.get_grid().items():
        grid_data[y, x] = entry["quality"]  # matplotlib の座標は [y, x]

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    im = plt.imshow(grid_data, origin="lower", cmap=cmap, vmin=1, vmax=20)
    plt.colorbar(im, label="Quality Score")
    plt.title(f"QD Map: {archive.name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def preprocess_idea_text(idea):
    return " ".join(
        [idea.get("Title", "").strip(), idea.get("Short Hypothesis", "").strip(), idea.get("Abstract", "").strip()]
    )


def compute_measures(
    idea: Dict[str, str],
    transformer_model: SentenceTransformer,
    umap_model: umap.UMAP,
    bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    text = preprocess_idea_text(idea)
    embedding = np.array(transformer_model.encode([text]))
    reduced = np.array(umap_model.transform(embedding))[0]  # 1つだけ

    min_x, max_x, min_y, max_y = bounds
    bc1 = normalize(reduced[0], min_x, max_x, NORMALIZED_MIN, NORMALIZED_MAX)
    bc2 = normalize(reduced[1], min_y, max_y, NORMALIZED_MIN, NORMALIZED_MAX)
    return bc1, bc2


def normalize(value: float, min_val: float, max_val: float, target_min: float, target_max: float) -> float:
    if value < min_val or value > max_val:
        print(f"[CLIP] value={value:.3f} outside range ({min_val:.3f}, {max_val:.3f})")

    value = np.clip(value, min_val, max_val)
    scale = (target_max - target_min) / (max_val - min_val)
    return (value - min_val) * scale + target_min


def fit_umap_on_initial_ideas(
    all_ideas: List[Dict[str, str]], transformer_model: SentenceTransformer
) -> Tuple[umap.UMAP, Tuple[float, float, float, float]]:
    texts = [preprocess_idea_text(idea) for idea in all_ideas]
    embeddings = np.array(transformer_model.encode(texts))

    n_samples = len(embeddings)
    n_neighbors = min(10, n_samples - 1)

    umap_model = umap.UMAP(n_components=2, metric="cosine", random_state=0, n_neighbors=n_neighbors)
    reduced = np.array(umap_model.fit_transform(embeddings))

    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title("UMAP Reduced Space")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    min_x, max_x = np.min(reduced[:, 0]), np.max(reduced[:, 0])
    min_y, max_y = np.min(reduced[:, 1]), np.max(reduced[:, 1])

    # マージンを加えて範囲を広げる
    range_x = max_x - min_x
    range_y = max_y - min_y
    bounds = (
        min_x - MARGIN_RATIO * range_x,
        max_x + MARGIN_RATIO * range_x,
        min_y - MARGIN_RATIO * range_y,
        max_y + MARGIN_RATIO * range_y,
    )
    return umap_model, bounds


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate AI scientist proposals - template free")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")

    base_dir = args.workshop_file.replace(".md", "")

    output_dir = "results/qd"
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/elites", exist_ok=True)

    # 初期アイデア読み込み & 辞書構築
    path = os.path.join(base_dir, "initial_ideas.json")
    with open(path, "r", encoding="utf-8") as f:
        all_ideas = json.load(f)
    idea_lookup = {idea["ID"]: idea for idea in all_ideas}

    # アーカイブの初期化と登録
    archive = SimpleGridArchive(name="", resolution=MAP_RESOLUTION)

    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model, bounds = fit_umap_on_initial_ideas(all_ideas, transformer_model)

    for idea in all_ideas:
        measures = compute_measures(idea, transformer_model, umap_model, bounds)
        quality = 0
        archive.overwrite_idea_at(idea["ID"], quality, measures)

    # QD進化ループ（20世代）
    for generation in range(1, 11):

        # エリートをランダムに3件選出し、親とする
        elites = archive.sample_elites(3)
        parents = [idea_lookup[e["idea_id"]] for e in elites]

        # 新しいアイデアを生成（+スコア付き）
        new_ideas = mutate_ideas(
            base_dir=base_dir,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            ideas=parents,
            generation=generation,
        )

        # 全アイデアに追加し、アーカイブに保存
        for idea in new_ideas:
            all_ideas.append(idea)
            idea_id = idea["ID"]
            idea_lookup[idea_id] = idea

            measures = compute_measures(idea, transformer_model, umap_model, bounds)
            exsiting_idea, exsiting_quality = archive.get_idea_at(measures)
            if exsiting_idea and exsiting_quality:
                better_idea = pairwise_evaluate(
                    idea_a=exsiting_idea,
                    idea_b=idea,
                    client=client,
                    model=client_model,
                    workshop_description=workshop_description,
                )
                quality = exsiting_quality + 1
                print(f"override: {better_idea['ID']}")
            else:
                better_idea = idea
                quality = 0
                print(f"add new idea: {idea_id}")
            archive.overwrite_idea_at(better_idea["ID"], quality, measures)

        # 可視化保存
        save_qd_map_image(archive, f"{output_dir}/images/gen_{generation}.png")

        # 5世代ごとにエリートと全アイデアを保存
        if generation % 5 == 0:
            export_elites(archive, idea_lookup, f"{output_dir}/elites/gen_{generation}.json")

        save_all_ideas(all_ideas, f"{output_dir}/all_ideas.json")
