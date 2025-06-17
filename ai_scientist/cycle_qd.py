import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from archive import SimpleGridArchive
from generate_ideas import mutate_ideas

from ai_scientist.llm import AVAILABLE_LLMS, create_client

# 評価指標
CRITERIAS = ["Feasibility", "Interestingness", "Novelty"]
MAP_RESOLUTION = 20  # QDマップの解像度（20x20）


def load_initial_ideas(base_dir: str) -> List[Dict]:
    """
    指定ディレクトリ内の3軸の初期アイデアファイルをすべて読み込む
    """
    all_ideas = []
    for crit in CRITERIAS:
        path = os.path.join(base_dir, f"{crit}.json")
        with open(path, "r", encoding="utf-8") as f:
            ideas = json.load(f)
            all_ideas.extend(ideas)
    return all_ideas


def build_archives() -> Dict[str, SimpleGridArchive]:
    """
    3つの評価軸ごとにQDマップ（アーカイブ）を初期化
    """
    return {crit: SimpleGridArchive(name=crit, resolution=MAP_RESOLUTION) for crit in CRITERIAS}


def register_to_archives(archives, ideas):
    """
    新しいアイデアを各アーカイブに登録する。
    BC（行動特徴）は評価軸を除いた2軸とする。
    """
    for crit in CRITERIAS:
        bc_keys = [c for c in CRITERIAS if c != crit]
        for idea in ideas:
            measures = [idea.get(k, 0) for k in bc_keys]
            quality = idea.get(crit, 0)
            archives[crit].add(idea_id=idea["ID"], quality=quality, measures=measures)


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

    bc_keys = [c for c in CRITERIAS if c != archive.name]

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    im = plt.imshow(grid_data, origin="lower", cmap=cmap, vmin=1, vmax=20)
    plt.colorbar(im, label="Quality Score")
    plt.title(f"QD Map: {archive.name}")
    plt.xlabel(bc_keys[0])
    plt.ylabel(bc_keys[1])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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

    initial_idea_dir = args.workshop_file.replace(".md", "")

    output_dir = "results"
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/elites", exist_ok=True)

    # 初期アイデア読み込み & 辞書構築
    all_ideas = load_initial_ideas(initial_idea_dir)
    idea_lookup = {idea["ID"]: idea for idea in all_ideas}

    # 各評価軸用アーカイブの初期化と登録
    archives = build_archives()
    register_to_archives(archives, all_ideas)

    # QD進化ループ（20世代）
    for generation in range(1, 4):
        crit = CRITERIAS[generation % 3]  # CycleQD: 評価軸を1つ選択
        archive = archives[crit]

        # エリートをランダムに3件選出し、親とする
        elites = archive.sample_elites(3)
        parents = [idea_lookup[e["idea_id"]] for e in elites]

        # 新しいアイデアを生成（+スコア付き）
        new_ideas = mutate_ideas(
            base_dir=initial_idea_dir,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            ideas=parents,
            generation=generation,
        )

        # 全アイデアに追加し、辞書にも反映
        for idea in new_ideas:
            all_ideas.append(idea)
            idea_lookup[idea["ID"]] = idea

        # すべてのアーカイブに登録（評価軸が異なるため）
        register_to_archives(archives, new_ideas)

        # 可視化保存
        for cri in CRITERIAS:
            save_qd_map_image(archives[cri], f"{output_dir}/images/gen_{generation}_{cri}.png")

            # 5世代ごとにエリートと全アイデアを保存
            if generation % 5 == 0:
                export_elites(archives[cri], idea_lookup, f"{output_dir}/elites/gen_{generation}_{crit}.json")
        save_all_ideas(all_ideas, f"{output_dir}/all_ideas.json")
