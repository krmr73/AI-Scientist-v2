import json
from collections import Counter

# ログ読み込み
with open("results/elo__matches.json", "r", encoding="utf-8") as f:
    base = json.load(f)
with open("results/combined_add_matches.json", "r", encoding="utf-8") as f:
    add = json.load(f)

# # 空の追加ログを想定
# add = []

# 重複除去用セット
seen = set()
unique_matches = []

# ベースと追加を合わせて1本のリストとして扱う
for match in base + add:
    ga, ia = match["group_a"], match["idx_a"]
    gb, ib = match["group_b"], match["idx_b"]

    # ペアの順序を正規化（A vs B と B vs A を同一とみなす）
    if (ga, ia, gb, ib) > (gb, ib, ga, ia):
        key = (gb, ib, ga, ia)
    else:
        key = (ga, ia, gb, ib)

    if key not in seen:
        seen.add(key)
        unique_matches.append(match)

print(f"統合前: {len(base) + len(add)} 試合")
print(f"統合後（重複除去後）: {len(unique_matches)} 試合")

# 保存
with open("results/combined_matches_latest.json", "w", encoding="utf-8") as f:
    json.dump(unique_matches, f, ensure_ascii=False, indent=2)


def norm_key(ga, ia, gb, ib):
    return (ga, ia, gb, ib) if (ga, ia, gb, ib) <= (gb, ib, ga, ia) else (gb, ib, ga, ia)


def keys_from(matches):
    ks = []
    for m in matches:
        ks.append(norm_key(m["group_a"], m["idx_a"], m["group_b"], m["idx_b"]))
    return ks


# 1) 各ログ内のユニーク数と内部重複
kb = keys_from(base)
ka = keys_from(add)
ub, ua = set(kb), set(ka)
dup_in_base = len(kb) - len(ub)
dup_in_add = len(ka) - len(ua)

# 2) ログ間の重なり
overlap = len(ub & ua)

print(f"[base]   total={len(base)}, unique={len(ub)}, dup_inside={dup_in_base}")
print(f"[add ]   total={len(add)},  unique={len(ua)}, dup_inside={dup_in_add}")
print(f"[between] overlap(base∩add)={overlap}")

# 各グループの試合数
group_counter = Counter()
for m in unique_matches:
    group_counter[m["group_a"]] += 1
    group_counter[m["group_b"]] += 1
print("各グループの試合数:", dict(group_counter))
