import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

Participant = Tuple[str, int]  # (group, idx)


INPUT_PATH = "results/matches_process/combined_matches_latest.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="試合データ(JSON)から (group, idx) ごとの勝率を集計し、上位10件を表示します。"
    )
    p.add_argument("--topk", type=int, default=10, help="上位K件を表示 (既定: 10)")
    p.add_argument("--min-games", type=int, default=2, help="集計対象とする最低試合数 (既定: 1)")
    p.add_argument(
        "--output",
        default="",
        help="CSVとして保存する場合の出力パス（例: results.csv）。未指定なら標準出力のみ。",
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
    wins: Dict[Participant, int] = defaultdict(int)
    games: Dict[Participant, int] = defaultdict(int)

    for i, rec in enumerate(records):
        # 必要キーの存在チェック（欠けていたらスキップ/例外にする）
        for k in ("group_a", "idx_a", "group_b", "idx_b", "winner_raw"):
            if k not in rec:
                raise KeyError(f"{i}件目のレコードに {k} がありません")

        left = key_left(rec)
        right = key_right(rec)

        # 総試合数を両者にカウント
        games[left] += 1
        games[right] += 1

        # 勝者に勝ちをカウント
        wkey = winner_key(rec)
        wins[wkey] += 1

    return wins, games


def compute_ranking(wins: Dict[Participant, int], games: Dict[Participant, int], min_games: int):
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

    # 並び順: 勝率降順 → 総試合数降順 → 勝利数降順 → group 昇順 → idx 昇順
    rows.sort(key=lambda r: (-r["win_rate"], -r["games"], -r["wins"], r["group"], r["idx"]))
    return rows


def print_table(rows, topk: int):
    from math import isfinite

    top = rows[:topk]
    # 見やすいテキスト表
    print("rank\tgroup\tidx\twins\tlosses\tgames\twin_rate")
    for i, r in enumerate(top, 1):
        wr = r["win_rate"]
        wr_str = f"{wr:.3f}" if isfinite(wr) else "NaN"
        print(f"{i}\t{r['group']}\t{r['idx']}\t{r['wins']}\t{r['losses']}\t{r['games']}\t{wr_str}")


def maybe_write_csv(rows, path: str, topk: int):
    if not path:
        return
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "group", "idx", "wins", "losses", "games", "win_rate"])
        for i, r in enumerate(rows[:topk], 1):
            w.writerow([i, r["group"], r["idx"], r["wins"], r["losses"], r["games"], f"{r['win_rate']:.6f}"])
    print(f"\nCSVを書き出しました: {path}")


def main():
    args = parse_args()
    records = load_data(INPUT_PATH)
    wins, games = accumulate(records)
    rows = compute_ranking(wins, games, args.min_games)
    print_table(rows, args.topk)
    maybe_write_csv(rows, args.output, args.topk)


if __name__ == "__main__":
    main()
