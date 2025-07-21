import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
from generate_ideas import mutate_ideas, pairwise_evaluate
from ribs.archives import CVTArchive
from sentence_transformers import SentenceTransformer

from ai_scientist.llm import AVAILABLE_LLMS, create_client

MAP_RESOLUTION = 10
NORMALIZED_MIN = 0
NORMALIZED_MAX = MAP_RESOLUTION - 1  # 9
MARGIN_RATIO = 0.2


def export_elites(archive, idea_lookup, out_path):
    """
    QDアーカイブに残っているエリートのアイデアをJSONとして保存
    """
    elites = archive.data(fields=["solution", "objective", "measures"], return_type="dict")
    elite_idxs = elites["solution"].flatten().tolist()
    elite_ideas = [idea_idx_lookup[idx] for idx in elite_idxs]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(elite_ideas, f, ensure_ascii=False, indent=4)


def save_all_ideas(all_ideas, path):
    """
    すべての生成済みアイデア（淘汰含む）を1つのファイルに保存
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_ideas, f, ensure_ascii=False, indent=4)


def preprocess_idea_text(idea):
    return " ".join(
        [idea.get("Title", "").strip(), idea.get("Short Hypothesis", "").strip(), idea.get("Abstract", "").strip()]
    )


def ideas_to_vecs(
    ideas: List[Dict[str, str]],
    transformer_model: SentenceTransformer,
) -> np.ndarray:
    texts = [preprocess_idea_text(idea) for idea in ideas]
    embeddings = transformer_model.encode(texts, convert_to_numpy=True)  # shape: (N, 384)
    # L2ノルムで正規化（各ベクトルの長さが1になる）
    embeddings = normalize(embeddings, norm="l2", axis=1)

    return embeddings


def sample_valid_elites(archive, n, rng=np.random):
    """
    埋まっているセル（NaN を含まないエリート）のみから
    一様ランダムで n 件サンプルして返す。

    Returns
    -------
    dict
        {"solution": (n, D_sol), "objective": (n,), "measures": (n, D_meas)}
    """
    # ── 1) アーカイブ中の全エリートを取得（dict 形式でコピーが返る）
    elites = archive.data(fields=["solution", "objective", "measures"], return_type="dict")

    # ── 2) NaN を含む行を除外（occupied 行だけ残す）
    valid_mask = ~np.isnan(elites["objective"])
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < n:
        raise ValueError(f"有効エリートが {len(valid_idx)} 件しかありません。")

    # ── 3) 有効インデックスからランダムに n 件選ぶ（重複なし）
    sampled_idx = rng.choice(valid_idx, size=n, replace=False)

    # ── 4) 選んだインデックスで各フィールドを抽出して返す
    sampled = {
        "solution": elites["solution"][sampled_idx],
        "objective": elites["objective"][sampled_idx],
        "measures": elites["measures"][sampled_idx],
    }
    return sampled


def analyze_measure_ranges(measures: np.ndarray):
    min_vals = measures.min(axis=0)
    max_vals = measures.max(axis=0)
    mean_vals = measures.mean(axis=0)
    std_vals = measures.std(axis=0)

    overall_min = min_vals.min()
    overall_max = max_vals.max()
    overall_mean = mean_vals.mean()
    overall_std = std_vals.mean()

    print("📊 measures 全体統計:")
    print(f"  min:  {overall_min:.6f}")
    print(f"  max:  {overall_max:.6f}")
    print(f"  mean: {overall_mean:.6f}")
    print(f"  std:  {overall_std:.6f}")

    return overall_min, overall_max


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

    # all_ideasの前半の3件を使用
    # all_ideas = all_ideas[:3]
    idea_lookup = {idea["ID"]: idea for idea in all_ideas}

    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    solution_values = []
    objectives = np.zeros(len(all_ideas), dtype=np.float32)
    # measures = np.array(ideas_to_vecs(all_ideas, transformer_model))

    vecs = ideas_to_vecs(all_ideas, transformer_model)
    print("min:", vecs.min(), "max:", vecs.max(), "mean:", vecs.mean(), "std:", vecs.std())

    pca = PCA(n_components=20)
    pca.fit(vecs)

    measures = np.array(pca.transform(vecs))

    mins = measures.min(axis=0)
    maxs = measures.max(axis=0)
    range_margin = 0.05
    ranges = [(float(min_val - range_margin), float(max_val + range_margin)) for min_val, max_val in zip(mins, maxs)]

    # アーカイブの初期化と登録
    archive = CVTArchive(solution_dim=1, cells=512, ranges=ranges, seed=42, centroid_method="sobol")

    # 0の配列で初期化
    # shape: (N, 1) ここでNはアイデアの

    idea_idx_lookup = {}
    for idx, idea in enumerate(all_ideas):
        idea_idx_lookup[idx] = idea  # dictとして保持
        solution_values.append(idx)
    solutions = np.array(solution_values, dtype=np.int32).reshape(-1, 1)

    assert not np.any(np.isnan(solutions))
    assert not np.any(np.isnan(objectives))
    assert not np.any(np.isnan(measures))
    archive.add(
        solution=solutions,  # shape: (N, 1)
        objective=objectives,  # shape: (N,)
        measures=measures,  # shape: (N, 384)
    )
    history = []

    print(f"Initial archive size: {archive.stats.num_elites}")

    # QD進化ループ（20世代）
    for generation in range(1, 51):

        # 1. エリートをランダムにサンプリング
        # num_elites = archive.stats.num_elites
        # print(f"エリートの数: {num_elites}")

        some_elites = archive.sample_elites(5)
        # some_elites = sample_valid_elites(archive, 2)

        if some_elites is None:
            print(f"[Gen {generation}] Not enough elites to sample parents.")
            break
        old_idea_idxs = some_elites["solution"].flatten().tolist()
        old_ideas = [idea_idx_lookup[idx] for idx in old_idea_idxs]

        # 2. 変異による新しいアイデアの生成
        new_ideas = mutate_ideas(
            base_dir=base_dir,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            ideas=old_ideas,
            generation=generation,
        )

        num_existing = len(all_ideas)
        # 3. 新しいアイデアを登録
        for i, idea in enumerate(new_ideas):
            all_ideas.append(idea)
            idea_lookup[idea["ID"]] = idea
            idea_idx_lookup[num_existing + i] = idea

        vecs = ideas_to_vecs(new_ideas, transformer_model)
        pca_vecs = pca.transform(vecs)

        # 該当セルの既存エリートを取得
        occupieds, elites = archive.retrieve(pca_vecs)

        # 追加するsolutionとobjectiveの初期化
        new_solution_values = []
        new_objective_values = []
        new_measure_values = []

        for i, occupied in enumerate(occupieds):

            new_idea = new_ideas[i]

            # 4. アーカイブに登録
            if occupied:
                # 既存のアイデアがある場合は上書き

                existing_quality = elites["objective"][i]
                existing_embed = elites["measures"][i]
                existing_idea_idx = elites["solution"][i][0]  # 1次元のソリューションから取得
                existing_idea = idea_idx_lookup[existing_idea_idx]

                better_idea = pairwise_evaluate(
                    idea_a=existing_idea,
                    idea_b=new_idea,
                    client=client,
                    model=client_model,
                    workshop_description=workshop_description,
                )
                new_objective_values.append(existing_quality + 1.0)  # 既存のスコアに1.0を加算
                print(f"override: {better_idea['ID']}")
            else:
                better_idea = new_idea
                new_objective_values.append(0.0)  # 新規アイデアは初期スコア0
                print(f"add new idea: {better_idea['ID']}")

            if better_idea == new_idea:
                new_solution_values.append(num_existing + i)
                new_measure_values.append(pca_vecs[i])
            else:
                # 既存のアイデアを上書きした場合はそのidxを使用
                new_solution_values.append(existing_idea_idx)
                new_measure_values.append(existing_embed)

        assert not np.any(np.isnan(solutions))
        assert not np.any(np.isnan(objectives))
        assert not np.any(np.isnan(measures))
        archive.add(
            solution=np.array(new_solution_values, dtype=np.int32).reshape(-1, 1),
            objective=np.array(new_objective_values, dtype=np.float32),
            measures=np.array(new_measure_values, dtype=np.float32),
        )
        print(f"Generation {generation} complete. Archive size: {archive.stats.num_elites}")
        print(f"カバレッジ: {archive.stats.coverage:.2%}")
        print(f"QDスコア: {archive.stats.qd_score:.4f}")

        # 世代の記録を保存
        record = {
            "generation": generation,
            "qd_score": archive.stats.qd_score,
            "coverage": archive.stats.coverage,
            "num_elites": archive.stats.num_elites,
        }
        history.append(record)

        # 5世代ごとにエリートと全アイデアを保存
        if generation % 5 == 0:
            export_elites(archive, idea_lookup, f"{output_dir}/elites/gen_{generation}.json")

    # CSV形式で保存（Pandasでプロットなどが簡単）
    df = pd.DataFrame(history)
    df.to_csv(f"{output_dir}/qd_history.csv", index=False)

    # save_all_ideas(all_ideas, f"{output_dir}/all_ideas.json")
    # # grid = archive.get_grid()
    # # with open(f"{output_dir}/grid.json", "w", encoding="utf-8") as f:
    # #     json.dump(grid, f, ensure_ascii=False, indent=4)
