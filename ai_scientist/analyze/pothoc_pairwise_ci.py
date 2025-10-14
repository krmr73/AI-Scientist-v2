import collections
import json

path = "results/elo_combined_matches.json"
matches = json.load(open(path, "r"))

pair_n = collections.Counter()
pair_w = collections.Counter()

for m in matches:
    a, b = m["group_a"], m["group_b"]
    key = tuple(sorted([a, b]))
    pair_n[key] += 1
    pair_w[key] += m["sa_effective"]  # 引分=0.5扱い

print("Pairwise coverage & 95% CI (sorted A vs B, A視点勝率)")
for g1, g2 in sorted(pair_n):
    n = pair_n[(g1, g2)]
    p = pair_w[(g1, g2)] / n
    se = (p * (1 - p) / n) ** 0.5
    lo, hi = p - 1.96 * se, p + 1.96 * se
    flag = "OK" if (hi < 0.5 or lo > 0.5) else "UNCERTAIN"
    print(f"{g1} vs {g2}: n={n}, p={p:.3f}, 95%CI=({lo:.3f},{hi:.3f}) [{flag}]")

print(f"Total matches: {len(matches)}")


# groupごとのelo値を計算
def compute_elo(matches, k=32, initial_elo=1500):
    elos = collections.defaultdict(lambda: initial_elo)

    for m in matches:
        a, b = m["group_a"], m["group_b"]
        sa, sb = m["sa_effective"], 1 - m["sa_effective"]

        ea = 1 / (1 + 10 ** ((elos[b] - elos[a]) / 400))
        eb = 1 / (1 + 10 ** ((elos[a] - elos[b]) / 400))

        elos[a] += k * (sa - ea)
        elos[b] += k * (sb - eb)

    return elos


elos = compute_elo(matches)
print("\nElo ratings:")
for g, e in sorted(elos.items()):
    print(f"  {g}: {e:.1f}")
mean_elo = sum(elos.values()) / len(elos)
print(f"Mean Elo: {mean_elo:.1f}")
print("Elo differences from mean:")
for g, e in sorted(elos.items()):
    print(f"  {g}: {e - mean_elo:+.1f}")
