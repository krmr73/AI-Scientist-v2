"""
Minimal Elo ranking for LLM pairwise judging (group-level)
- Input: multiple JSON files, each with {"group": "A", "ideas": ["..."]}
- Sampling: uniformly over group pairs, then randomly pick an idea from each group
- Judge: LLM-as-a-judge returns winner in {left,right,tie}
- Output: item-level Elo, group average Elo, simple CSV logs

Usage:
  python elo.py \
    --groups A.json B.json C.json D.json \
    --matches 1200 --K 16 --seed 42 --out-prefix results/elo \
    --workshop-file workshop.md

Notes:
- This version uses the external LLM client wrapper (create_client, get_response_from_llm) with model fixed to gpt-4o-2024-05-13.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import deque
from itertools import product
from typing import Dict, List, Tuple

import dotenv
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_parent_dir)
sys.path.append(parent_dir)


from llm import create_client, get_response_from_llm

dotenv.load_dotenv()


# -------------------------- I/O --------------------------


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------- LLM-as-judge ---------------------
# User-provided prompt style (expert workshop reviewer)
pairwise_evaluation_system_prompt = "You are an expert reviewer for an academic workshop. Your role is to evaluate and compare research ideas submitted to the workshop. You must provide an impartial and expert judgment based on the criteria of novelty, feasibility, and relevance to the workshop theme."

WORKSHOP_DESC = """"""

PAIRWISE_USER_TMPL = """
Workshop Description:
{workshop_description}

Here are two research ideas:

[Idea A]
{idea_a}

[Idea B]
{idea_b}

Which idea is better overall based on the workshop's theme and the following criteria:
1. Novelty
2. Feasibility
3. Significance

Please respond with only: "A is better" or "B is better".
""".strip()

client, model = create_client("gpt-4o-2024-05-13")


def judge_pair(idea_left: str, idea_right: str, temperature: float = 0.0) -> str:
    """Use the workshop-style prompt. Map 'A is better'/'B is better' to left/right.
    We randomize A/B outside this fn via caller; here A=left, B=right as given.
    """
    user_prompt = PAIRWISE_USER_TMPL.format(
        workshop_description=WORKSHOP_DESC or "",
        idea_a=idea_left,
        idea_b=idea_right,
    )
    content, _ = get_response_from_llm(
        prompt=user_prompt,
        client=client,
        model=model,
        system_message=pairwise_evaluation_system_prompt,
        temperature=temperature,
    )
    txt = (content or "").strip().lower()
    if "a is better" in txt:
        return "left"
    if "b is better" in txt:
        return "right"
    # fallback: try to detect lone 'a' or 'b'
    if re.fullmatch(r"a", txt):
        return "left"
    if re.fullmatch(r"b", txt):
        return "right"
    raise ValueError(f"Unexpected judge output: {content}")


# -------------------------- Elo --------------------------


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, sa: float, K: float) -> Tuple[float, float]:
    ea = expected_score(ra, rb)
    eb = 1.0 - ea
    ra2 = ra + K * (sa - ea)
    rb2 = rb + K * ((1.0 - sa) - eb)
    return ra2, rb2


# ----------------------- Main logic ----------------------


def run(groups_data: Dict[str, List[str]], matches: int, K: float, seed: int, temperature: float) -> Dict:
    rng = random.Random(seed)

    # --- 初期化（アイテムEloは従来通り） ---
    items = []
    for g, lst in groups_data.items():
        for i, t in enumerate(lst):
            items.append((g, i, t))
    R = {(g, i): 1000.0 for (g, i, _) in items}

    group_names = sorted(groups_data.keys())
    group_pairs = [(a, b) for i, a in enumerate(group_names) for b in group_names[i + 1 :]]
    P = len(group_pairs)

    # --- 各グループペアの全ユニーク組合せデッキ（非復元）を用意 ---
    pair_decks: Dict[Tuple[str, str], deque] = {}
    total_capacity = 0
    for ga, gb in group_pairs:
        deck = [(ia, ib) for ia, ib in product(range(len(groups_data[ga])), range(len(groups_data[gb])))]
        rng.shuffle(deck)  # seedに基づいて決定的にシャッフル
        pair_decks[(ga, gb)] = deque(deck)
        total_capacity += len(deck)

    if matches > total_capacity:
        raise ValueError(
            f"要求された試合数 matches={matches} は、ユニークに生成可能な最大数 {total_capacity} を超えています。"
        )

    # --- グループペア一様サンプリングを厳密に維持する割当て ---
    base = matches // P
    rem = matches % P
    # 余りremをランダムにrem個のペアへ+1（seedに依存）
    pair_order = group_pairs[:]  # コピー
    rng.shuffle(pair_order)
    need_per_pair = {pair: base for pair in group_pairs}
    for pair in pair_order[:rem]:
        need_per_pair[pair] += 1

    # --- 実行（各ペアから必要数だけ先頭を消化：ユニーク保証） ---
    log = []
    m = 0
    # ペアの出現順はシャッフルしつつ、必要数を1件ずつラウンドロビンで消化してもよいが、
    # ここでは単純に必要数分を順に積む → その後全体をシャッフルしても良い。
    scheduled_matches = []
    for pair in group_pairs:
        ga, gb = pair
        need = need_per_pair[pair]
        if need > len(pair_decks[pair]):
            raise ValueError(
                f"ペア {ga} vs {gb} に必要な試合数 {need} がデッキ残数 {len(pair_decks[pair])} を超えました。"
            )
        for _ in range(need):
            ia, ib = pair_decks[pair].popleft()
            scheduled_matches.append((ga, ia, gb, ib))

    # 全体の順序も乱択（seed依存、結果の再現性を保ちつつ偏りを避ける）
    rng.shuffle(scheduled_matches)

    for ga, ia, gb, ib in tqdm(scheduled_matches, desc="Judging matches"):
        a_key, b_key = (ga, ia), (gb, ib)
        a_text, b_text = groups_data[ga][ia], groups_data[gb][ib]

        # 左右の表示順はランダム
        if rng.random() < 0.5:
            left_text, right_text = a_text, b_text
            invert = False
        else:
            left_text, right_text = b_text, a_text
            invert = True

        # 判定
        winner = judge_pair(left_text, right_text, temperature=temperature)
        if winner == "left":
            sa = 1.0 if not invert else 0.0
        elif winner == "right":
            sa = 0.0 if not invert else 1.0
        else:
            sa = 0.5  # tieを使う場合

        # Elo更新（従来通り）
        ra, rb = R[a_key], R[b_key]
        ra2, rb2 = update_elo(ra, rb, sa, K)
        R[a_key], R[b_key] = ra2, rb2

        log.append(
            {
                "match": m,
                "group_a": ga,
                "idx_a": ia,
                "group_b": gb,
                "idx_b": ib,
                "winner_raw": winner,
                "sa_effective": sa,
                "ra_before": ra,
                "rb_before": rb,
                "ra_after": ra2,
                "rb_after": rb2,
            }
        )
        m += 1

    group_ratings = {g: sum(R[(g, i)] for i in range(len(lst))) / len(lst) for g, lst in groups_data.items()}

    return {
        "item_ratings": {f"{g}:{i}": R[(g, i)] for g, lst in groups_data.items() for i in range(len(lst))},
        "group_ratings": group_ratings,
        "matches": log,
    }


# -------------------------- CLI --------------------------
def main():
    ap = argparse.ArgumentParser(description="Minimal Elo ranking with LLM-as-judge (using gpt-4o-2024-05-13)")
    ap.add_argument("--matches", type=int, default=1200)
    ap.add_argument("--K", type=float, default=16.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="results/elo")
    ap.add_argument(
        "--workshop-file", type=str, default="", help="Path to a file containing the workshop description."
    )
    args = ap.parse_args()

    global WORKSHOP_DESC
    with open(args.workshop_file, "r") as f:
        WORKSHOP_DESC = f.read()

    idea_num = 80
    proposed_ideas = load_json("../results/qd/elites/gen_50.json")[:idea_num]
    proposed_literature_ideas = load_json("../results/qd_semantic_scholar/elites/gen_50.json")[:idea_num]

    existing_ideas = load_json("ideas/polya_urn_model.json")[:idea_num]
    existing_literature_ideas = load_json("ideas/polya_urn_model_with_semanticscholar.json")[: len(proposed_ideas)]

    groups = {
        "Reflection-only": existing_ideas,
        "Literature-informed": existing_literature_ideas,
        "Proposed": proposed_ideas,
        "Proposed-semantic": proposed_literature_ideas,
    }

    res = run(groups, matches=args.matches, K=args.K, seed=args.seed, temperature=args.temperature)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    with open(f"{args.out_prefix}_group_ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "elo_mean"])
        for g, r in sorted(res["group_ratings"].items(), key=lambda x: -x[1]):
            w.writerow([g, f"{r:.2f}"])
    with open(f"{args.out_prefix}_item_ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_idx", "elo"])
        for k, r in sorted(res["item_ratings"].items(), key=lambda x: -x[1]):
            w.writerow([k, f"{r:.2f}"])
    with open(f"{args.out_prefix}_matches.json", "w", encoding="utf-8") as f:
        json.dump(res["matches"], f, ensure_ascii=False, indent=2)

    print("Group Elo (mean):")
    for g, r in sorted(res["group_ratings"].items(), key=lambda x: -x[1]):
        print(f"  {g}: {r:.1f}")


if __name__ == "__main__":
    main()
