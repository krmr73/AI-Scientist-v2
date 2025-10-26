import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

Participant = Tuple[str, int]  # (group, idx)

INPUT_PATH = "results/matches_process/combined_matches_latest.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="試合データ(JSON)から (group, idx) ごとの勝率を集計し、各グループの上位K件を表示します。"
    )
    # 各グループ内の上位K件
    p.add_argument("--topk", type=int, default=5, help="各グループ内の上位K件を表示 (既定: 5)")
    p.add_argument("--min-games", type=int, default=2, help="集計対象とする最低試合数 (既定: 2)")
    p.add_argument(
        "--output",
        default="",
        help="CSVとして保存する場合の出力パス（例: per_group_results.csv）。未指定なら標準出力のみ。",
    )
    return p.parse_args()


def load_data(path: str) -> List[Dict[str, Any]]:
    if path == "-":
        import sys

        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def key_left(rec: Dict[str, Any]) -> Participant:
    return (str(rec["group_a"]), int(rec["idx_a"]))


def key_right(rec: Dict[str, Any]) -> Participant:
    return (str(rec["group_b"]), int(rec["idx_b"]))


def winner_key(rec: Dict[str, Any]) -> Participant:
    w = rec.get("winner_raw")
    if w == "left":
        return key_left(rec)
    elif w == "right":
        return key_right(rec)
    # 想定外の値は例外にする（必要なら 'draw' 等の扱いを拡張）
    raise ValueError(f"winner_raw が不正です: {w!r}")


def accumulate(records: List[Dict[str, Any]]):
    """参加者ごとの勝ち数と試合数を集計"""
    wins: Dict[Participant, int] = defaultdict(int)
    games: Dict[Participant, int] = defaultdict(int)

    for i, rec in enumerate(records):
        for k in ("group_a", "idx_a", "group_b", "idx_b", "winner_raw"):
            if k not in rec:
                raise KeyError(f"{i}件目のレコードに {k} がありません")

        left = key_left(rec)
        right = key_right(rec)

        games[left] += 1
        games[right] += 1

        wkey = winner_key(rec)
        wins[wkey] += 1

    return wins, games


def compute_rows(wins: Dict[Participant, int], games: Dict[Participant, int], min_games: int):
    """(group, idx) 単位のメトリクス行を作成"""
    rows = []
    for p in games.keys():
        g = games[p]
        if g < min_games:
            continue
        w = wins.get(p, 0)
        win_rate = w / g if g > 0 else 0.0
        rows.append(
            {
                "group": p[0],
                "idx": p[1],
                "wins": w,
                "losses": g - w,
                "games": g,
                "win_rate": win_rate,
            }
        )

    # 基本の並び順: 勝率↓ → 試合数↓ → 勝利数↓ → group↑ → idx↑
    rows.sort(key=lambda r: (-r["win_rate"], -r["games"], -r["wins"], r["group"], r["idx"]))
    return rows


def select_topk_per_group(rows: List[Dict[str, Any]], topk: int) -> Dict[str, List[Dict[str, Any]]]:
    """グループごとに上位K件を抽出"""
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    # すでに rows はグローバルでソート済みだが、念のため各グループ内でも同じキーでソート
    for g in by_group:
        by_group[g].sort(key=lambda r: (-r["win_rate"], -r["games"], -r["wins"], r["idx"]))
        by_group[g] = by_group[g][:topk]

    return by_group


def print_per_group_table(by_group: Dict[str, List[Dict[str, Any]]]):
    from math import isfinite

    for group in sorted(by_group.keys()):
        items = by_group[group]
        if not items:
            continue
        print(f"\n=== Group: {group} ===")
        print("rank_in_group\tidx\twins\tlosses\tgames\twin_rate")
        for i, r in enumerate(items, 1):
            wr = r["win_rate"]
            wr_str = f"{wr:.3f}" if isfinite(wr) else "NaN"
            print(f"{i}\t{r['idx']}\t{r['wins']}\t{r['losses']}\t{r['games']}\t{wr_str}")


def print_idea_per_group(groups: Dict[str, List[Dict[str, Any]]], by_group: Dict[str, List[Dict[str, Any]]]):
    """
    by_group の各 (group, idx) に対応するアイデアを groups[group] の idx 番目から取り出して出力。
    idx はアイデアリスト中の順序（0始まり）として扱う。
    - 主要フィールド: Title, Short Hypothesis, Abstract
    - フォールバック: 辞書内の最初の文字列値
    """
    PREFERRED_FIELDS = ["Title", "Short Hypothesis", "Abstract"]
    JOIN_WITH = " — "
    TRUNCATE = 200  # 表示を切り詰める上限文字数（0以下で無効）

    def extract_text(item: Dict[str, Any]) -> str:
        """指定キーを順に探し、文字列を連結して返す。"""
        if isinstance(item, dict):
            parts = []
            for k in PREFERRED_FIELDS:
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            if parts:
                return JOIN_WITH.join(parts)
            # フォールバック: 最初の文字列値
            for v in item.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return str(item)
        return str(item)

    for group in sorted(by_group.keys()):
        top_items = by_group[group]
        if not top_items:
            continue

        print(f"\n=== Ideas for Group: {group} ===")
        print("rank_in_group\tidx\tidea")

        idea_list = groups.get(group)
        if not isinstance(idea_list, list):
            print("(no idea list for this group)")
            continue

        for i, r in enumerate(top_items, 1):
            idx = r["idx"]  # リスト内の順番（0始まり）でアクセス
            if 0 <= idx < len(idea_list):
                idea_entry = idea_list[idx]
                idea_text = extract_text(idea_entry)
            else:
                idea_text = "(not found)"

            if TRUNCATE > 0 and len(idea_text) > TRUNCATE:
                idea_text = idea_text[:TRUNCATE] + "…"

            print(f"{i}\t{idx}\t{idea_text}")


def maybe_write_csv_per_group(by_group: Dict[str, List[Dict[str, Any]]], path: str):
    if not path:
        return
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # CSVはグループごとに rank_in_group を付けてフラットに出力
        w.writerow(["group", "rank_in_group", "idx", "wins", "losses", "games", "win_rate"])
        for group in sorted(by_group.keys()):
            items = by_group[group]
            for i, r in enumerate(items, 1):
                w.writerow([group, i, r["idx"], r["wins"], r["losses"], r["games"], f"{r['win_rate']:.6f}"])
    print(f"\nCSVを書き出しました: {path}")


def main():
    # --- あなたのプロジェクトのデータ読み込み（既存のまま） ---
    idea_num = 80
    proposed_ideas = load_data("../results/qd/elites/gen_50.json")[:idea_num]
    proposed_literature_ideas = load_data("../results/qd_semantic_scholar/elites/gen_50.json")[:idea_num]

    existing_ideas = load_data("ideas/polya_urn_model.json")[:idea_num]
    existing_literature_ideas = load_data("ideas/polya_urn_model_with_semanticscholar.json")[: len(proposed_ideas)]

    groups = {
        "Reflection-only": existing_ideas,
        "Literature-informed": existing_literature_ideas,
        "Proposed": proposed_ideas,
        "Proposed-semantic": proposed_literature_ideas,
    }

    args = parse_args()
    records = load_data(INPUT_PATH)
    wins, games = accumulate(records)
    rows = compute_rows(wins, games, args.min_games)
    by_group = select_topk_per_group(rows, args.topk)
    print_per_group_table(by_group)
    maybe_write_csv_per_group(by_group, args.output)

    print_idea_per_group(groups, by_group)


if __name__ == "__main__":
    main()
