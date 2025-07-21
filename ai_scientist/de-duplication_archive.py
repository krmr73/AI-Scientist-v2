import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from generate_ideas import mutate_ideas, pairwise_evaluate
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from ai_scientist.llm import AVAILABLE_LLMS, create_client


class DistanceThresholdArchiveWithScore:
    def __init__(self, dim, threshold=0.85):
        self.dim = dim
        self.threshold = threshold
        self.vecs = []  # 正規化済みベクトル (N, dim)
        self.ideas = []  # 各ベクトルに対応するアイデア
        self.win_counts = []  # 各アイデアの「勝利数」

    def _find_similar_index(self, vec):
        if not self.vecs:
            return None, 0.0
        mat = np.vstack(self.vecs)
        sims = mat @ vec.reshape(-1, 1)
        idx = np.argmax(sims)
        max_sim = sims[idx].item()
        if max_sim >= self.threshold:
            return idx, max_sim
        return None, max_sim

    def add_or_replace(self, vec, idea, evaluate_fn=None):
        """近傍に似たものがあるかを確認し、あれば評価関数で置換判断"""
        idx, sim = self._find_similar_index(vec)

        if idx is None:
            # 類似セルなし → 新規追加
            self.vecs.append(vec)
            self.ideas.append(idea)
            self.win_counts.append(0)
            return True, "add_new"
        else:
            existing_idea = self.ideas[idx]
            better = evaluate_fn(existing_idea, idea)
            if better == idea:
                # 勝ったら置換・スコア初期化
                self.vecs[idx] = vec
                self.ideas[idx] = idea
                self.win_counts[idx] = 1
                return True, "override"
            else:
                # 負けたら相手に1勝加算
                self.win_counts[idx] += 1
                return False, "discard"

    def sample_elites(self, n, rng=np.random):
        if not self.vecs:
            raise IndexError("Archive is empty.")
        idxs = rng.choice(len(self.vecs), size=n, replace=True)
        return {
            "vecs": np.vstack([self.vecs[i] for i in idxs]),
            "ideas": [self.ideas[i] for i in idxs],
            "win_counts": [self.win_counts[i] for i in idxs],
        }

    @property
    def num_elites(self):
        return len(self.ideas)


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

    # モデルと初期ベクトル
    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = ideas_to_vecs(all_ideas, transformer_model)

    # アーカイブ初期化
    archive = DistanceThresholdArchiveWithScore(dim=384, threshold=0.8)

    # 初期登録
    for idea, vec in zip(all_ideas, vecs):
        archive.add_or_replace(vec, idea, evaluate_fn=lambda a, b: a)

    print(f"Initial elites: {archive.num_elites}")
    history = []

    # QD進化ループ
    for generation in range(1, 51):
        # 1. 親アイデアの選択
        some_elites = archive.sample_elites(5)
        parent_ideas = some_elites["ideas"]

        # 2. 子アイデアの生成
        new_ideas = mutate_ideas(
            base_dir=base_dir,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            ideas=parent_ideas,
            generation=generation,
        )

        # 3. ベクトル化 & 登録
        new_vecs = ideas_to_vecs(new_ideas, transformer_model)
        for idea, vec in zip(new_ideas, new_vecs):
            all_ideas.append(idea)
            idea_lookup[idea["ID"]] = idea

            for elite_vec in archive.vecs:
                if cosine_similarity([vec], [elite_vec])[0][0] >= threshold:
                    skip = True
                    break

            added, reason = archive.add_or_replace(
                vec,
                idea,
                evaluate_fn=lambda a, b: pairwise_evaluate(
                    idea_a=a,
                    idea_b=b,
                    client=client,
                    model=client_model,
                    workshop_description=workshop_description,
                ),
            )

            if reason == "override":
                print(f"override: {idea['ID']}")
            elif reason == "add_new":
                print(f"add new idea: {idea['ID']}")

        # 進捗の記録
        record = {
            "generation": generation,
            "num_elites": archive.num_elites,
            "total_ideas": len(all_ideas),
        }
        history.append(record)
        print(f"Gen {generation} | Elites: {archive.num_elites}")

        # エリート保存
        if generation % 5 == 0:
            elite_ideas = archive.ideas
            with open(f"{output_dir}/elites/gen_{generation}.json", "w", encoding="utf-8") as f:
                json.dump(elite_ideas, f, ensure_ascii=False, indent=2)

    # 履歴を保存
    df = pd.DataFrame(history)
    df.to_csv(f"{output_dir}/qd_history.csv", index=False)
