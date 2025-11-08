import json
import os
from collections import Counter
from pathlib import Path


def norm_key(ga, ia, gb, ib):
    return (ga, ia, gb, ib) if (ga, ia, gb, ib) <= (gb, ib, ga, ia) else (gb, ib, ga, ia)


def load_matches_any(path: str):
    """配列形式 [ ... ] と { 'matches': [ ... ] } の両方に対応"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "matches" in data and isinstance(data["matches"], list):
        return data["matches"]
    if isinstance(data, list):
        return data
    raise ValueError(f"未知のJSON形式です（配列 or {{'matches': [...]}} を想定）: {path}")


# 入力
base = load_matches_any("../results/elo__matches.json")
add = load_matches_any("../results/combined_add_matches.json")

# 重複除去
seen = set()
unique_matches = []
for match in base + add:  # ← ここを add + base にすると「追加ログ優先」に変わる
    try:
        key = norm_key(match["group_a"], match["idx_a"], match["group_b"], match["idx_b"])
    except KeyError as e:
        # 必須キーがない壊れレコード
        print(f"[WARN] 必須キー欠落でスキップ: {e} | record={match}")
        continue
    if key not in seen:
        seen.add(key)
        unique_matches.append(match)

print(f"統合前: {len(base) + len(add)} 試合")
print(f"統合後（重複除去後）: {len(unique_matches)} 試合")

# 保存
out_path = Path("../results/combined_matches_latest.json")
os.makedirs(out_path.parent, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(unique_matches, f, ensure_ascii=False, indent=2)


def keys_from(matches):
    ks = []
    for m in matches:
        try:
            ks.append(norm_key(m["group_a"], m["idx_a"], m["group_b"], m["idx_b"]))
        except KeyError:
            continue
    return ks


# 統計
kb = keys_from(base)
ka = keys_from(add)
ub, ua = set(kb), set(ka)
dup_in_base = len(kb) - len(ub)
dup_in_add = len(ka) - len(ua)
overlap = len(ub & ua)

print(f"[base]   total={len(base)}, unique={len(ub)}, dup_inside={dup_in_base}")
print(f"[add ]   total={len(add)},  unique={len(ua)}, dup_inside={dup_in_add}")
print(f"[between] overlap(base∩add)={overlap}")

# 各グループの試合数（無向ペアのユニーク後）


group_counter = Counter()
for m in unique_matches:
    group_counter[m.get("group_a")] += 1
    group_counter[m.get("group_b")] += 1
print("各グループの試合数:", dict(group_counter))
