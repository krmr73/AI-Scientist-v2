"""
Minimal pairwise judging logger (group-level)
- Elo 計算は行わず、試合結果のみを保存
- 入力: 既存コードと同じ（プロジェクト内の JSON 群）
- サンプリング: グループペアを指定可能（A:* / *:B / A:B を複数可）。
  未出ペアを均等に選び、その中からアイデア同士を非復元でサンプリング
- Judge: LLM-as-a-judge が {left,right} を返す（tie は使わない想定だが後方互換で left/right/tie を許容）
- 出力: 単純な試合ログ JSON（*_matches.json）のみ

Usage 例:
  python elo.py \
    --matches 1200 --seed 42 --out-prefix results/elo \
    --workshop-file workshop.md \
    --pair "Reflection-only:*" --pair "Reflection-only:Literature-informed"
"""

from __future__ import annotations

import argparse
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
pairwise_evaluation_system_prompt = (
    "You are an expert reviewer for an academic workshop. Your role is to evaluate and "
    "compare research ideas submitted to the workshop. You must provide an impartial and "
    "expert judgment based on the criteria of novelty, feasibility, and relevance to the workshop theme."
)

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
    """ワークショップスタイルのプロンプト。'A is better'/'B is better' を left/right にマップ。"""
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
    if "a is better" in txt or re.fullmatch(r"a", txt):
        return "left"
    if "b is better" in txt or re.fullmatch(r"b", txt):
        return "right"
    if "tie" in txt:
        return "tie"
    raise ValueError(f"Unexpected judge output: {content}")


# -------------------- 既出ペアの扱い ---------------------


def norm_key(ga, ia, gb, ib):
    # (A,i)-(B,j) と (B,j)-(A,i) を同一視
    return (ga, ia, gb, ib) if (ga, ia, gb, ib) <= (gb, ib, ga, ia) else (gb, ib, ga, ia)


def load_seen_pairs(paths):
    """過去 matches.json（複数可）から既出ペア集合を作る"""
    seen = set()
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            ms = json.load(f)
        for m in ms:
            ga, ia = m["group_a"], m["idx_a"]
            gb, ib = m["group_b"], m["idx_b"]
            seen.add(norm_key(ga, ia, gb, ib))
    return seen


# ------------------ グループペアのフィルタ ------------------


def build_group_pairs(all_groups: List[str], pair_filters: List[str]) -> List[Tuple[str, str]]:
    """
    pair_filters の例:
      - "A:*"  -> A vs それ以外すべて
      - "*:B"  -> 全て vs B
      - "A:B"  -> A vs B だけ
    複数指定可。重複は除去。順序は (min, max) の無向扱い。
    指定が無い場合は全組合せ（A<B）。
    """
    gset = set(all_groups)
    pairs = set()

    def add_pair(x, y):
        if x == y:
            return
        a, b = (x, y) if x < y else (y, x)
        if a in gset and b in gset:
            pairs.add((a, b))

    if not pair_filters:
        # デフォルト：全組
        for i, a in enumerate(all_groups):
            for b in all_groups[i + 1 :]:
                add_pair(a, b)
        return sorted(pairs)

    for f in pair_filters:
        f = f.strip()
        if ":" not in f:
            # 単独名は無視（安全側）
            continue
        left, right = f.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left == "*" and right == "*":
            # 全組
            for i, a in enumerate(all_groups):
                for b in all_groups[i + 1 :]:
                    add_pair(a, b)
        elif left == "*" and right != "*":
            for a in all_groups:
                add_pair(a, right)
        elif left != "*" and right == "*":
            for b in all_groups:
                add_pair(left, b)
        else:
            add_pair(left, right)

    return sorted(pairs)


# ----------------------- Main logic ----------------------


def run(
    groups_data: Dict[str, List[str]],
    matches: int,
    seed: int,
    temperature: float,
    seen_pairs: set | None = None,
    group_pair_filters: List[str] | None = None,
) -> Dict:
    rng = random.Random(seed)
    seen_pairs = set() if seen_pairs is None else set(seen_pairs)

    # アイテム一覧
    items = []
    for g, lst in groups_data.items():
        for i, t in enumerate(lst):
            items.append((g, i, t))

    group_names = sorted(groups_data.keys())
    # フィルタでペア集合を決定
    group_pairs = build_group_pairs(group_names, group_pair_filters or [])

    # 各グループペアのデッキを「未出カードだけ」で作る（非復元）
    pair_decks = {}
    total_capacity = 0
    for ga, gb in group_pairs:
        all_pairs = [(ia, ib) for ia, ib in product(range(len(groups_data[ga])), range(len(groups_data[gb])))]
        remaining = [(ia, ib) for (ia, ib) in all_pairs if norm_key(ga, ia, gb, ib) not in seen_pairs]
        rng.shuffle(remaining)  # seedに基づく決定的シャッフル
        pair_decks[(ga, gb)] = deque(remaining)
        total_capacity += len(remaining)

    if total_capacity == 0:
        raise RuntimeError("未出のカードが残っていません（全て既出、またはフィルタが厳しすぎます）。")
    if matches > total_capacity:
        print(
            f"[WARN] 要求matches={matches} > 未出容量={total_capacity}。"
            f"生成できるのは最大 {total_capacity} 試合まで。"
        )
        matches = total_capacity

    active_pairs = [p for p in group_pairs if pair_decks[p]]

    log = []
    for m in tqdm(range(matches), desc="Judging matches"):
        ga, gb = rng.choice(active_pairs)
        ia, ib = pair_decks[(ga, gb)].popleft()

        # 既出登録
        seen_pairs.add(norm_key(ga, ia, gb, ib))

        a_text, b_text = groups_data[ga][ia], groups_data[gb][ib]

        # 左右ランダム化
        if rng.random() < 0.5:
            left_group, left_idx, left_text = ga, ia, a_text
            right_group, right_idx, right_text = gb, ib, b_text
            invert = False
        else:
            left_group, left_idx, left_text = gb, ib, b_text
            right_group, right_idx, right_text = ga, ia, a_text
            invert = True

        winner = judge_pair(left_text, right_text, temperature=temperature)

        # ログのみ（Elo は更新しない）
        match_rec = {
            "match": m,
            "group_a": ga,
            "idx_a": ia,
            "group_b": gb,
            "idx_b": ib,
            "left_group": left_group,
            "left_idx": left_idx,
            "right_group": right_group,
            "right_idx": right_idx,
            "winner_raw": winner,  # 'left' / 'right' / 'tie'
            "winner_group": (left_group if winner == "left" else right_group if winner == "right" else "tie"),
        }
        log.append(match_rec)

        # デッキが空になったら除外
        if not pair_decks[(ga, gb)]:
            active_pairs = [p for p in active_pairs if pair_decks[p]]
            if not active_pairs and m < matches - 1:
                break

    return {"matches": log}


# -------------------------- CLI --------------------------


def main():
    ap = argparse.ArgumentParser(description="Pairwise judging logger (no Elo)")
    ap.add_argument("--matches", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="results/elo")
    ap.add_argument(
        "--workshop-file", type=str, default="", help="Path to a file containing the workshop description."
    )
    ap.add_argument(
        "--seen-matches",
        type=str,
        nargs="*",
        default=[],
        help="過去の *_matches.json を指定すると、そこに含まれるペアを除外して追加生成します。",
    )
    ap.add_argument(
        "--pair",
        type=str,
        action="append",
        default=[],
        help='対戦させるグループペアを指定（複数可）。例: "Reflection-only:*", "Reflection-only:Proposed", "*:Literature-informed"',
    )
    args = ap.parse_args()

    global WORKSHOP_DESC
    if args.workshop_file:
        with open(args.workshop_file, "r", encoding="utf-8") as f:
            WORKSHOP_DESC = f.read()

    # --- あなたのプロジェクトのデータ読み込み（既存のまま） ---
    idea_num = 80
    proposed_ideas = load_json("../results/qd/elites/gen_50.json")[:idea_num]
    proposed_literature_ideas = load_json("../results/qd_semantic_scholar/elites/gen_50.json")[:idea_num]

    existing_ideas = load_json("ideas/polya_urn_model.json")[:idea_num]
    existing_literature_ideas = load_json("ideas/polya_urn_model_with_semanticscholar.json")[: len(proposed_ideas)]

    # 使いたいグループをここで選ぶ
    groups = {
        "Reflection-only": existing_ideas,
        "Literature-informed": existing_literature_ideas,
        "Proposed": proposed_ideas,
        "Proposed-semantic": proposed_literature_ideas,
    }

    seen_pairs = load_seen_pairs(args.seen_matches) if args.seen_matches else set()

    res = run(
        groups_data=groups,
        matches=args.matches,
        seed=args.seed,
        temperature=args.temperature,
        seen_pairs=seen_pairs,
        group_pair_filters=args.pair,  # ← ここでペア指定を反映
    )

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    with open(f"{args.out_prefix}_matches.json", "w", encoding="utf-8") as f:
        json.dump(res["matches"], f, ensure_ascii=False, indent=2)

    print(f"Saved matches to {args.out_prefix}_matches.json")


if __name__ == "__main__":
    main()
