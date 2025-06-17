import glob
import os
import re

from PIL import Image

# 対象のベースパス
base_path = "results/images"
output_dir = "results/gifs"
os.makedirs(output_dir, exist_ok=True)

# 評価軸
criterias = ["Feasibility", "Interestingness", "Novelty"]

# GIFのフレームレート（1枚あたりの表示時間, 単位: ミリ秒）
frame_duration = 200


def extract_generation(filename):
    match = re.search(r"gen_(\d+)_", filename)
    return int(match.group(1)) if match else -1


for crit in criterias:
    pattern = os.path.join(base_path, f"gen_*_{crit}.png")
    files = sorted(glob.glob(pattern), key=extract_generation)

    if not files:
        print(f"[{crit}] 対象ファイルが見つかりませんでした。")
        continue

    # 画像を読み込む
    images = [Image.open(f) for f in files]

    # GIFファイルの出力先
    gif_path = os.path.join(output_dir, f"{crit}.gif")

    # GIFとして保存
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,
        loop=0,
    )

    print(f"[{crit}] GIFを保存しました: {gif_path}")
