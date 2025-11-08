import random
from typing import Any, Dict, Optional, Tuple


class SimpleGridArchive:
    def __init__(self, name: str, bc1_range: Tuple[int, int], bc2_range: Tuple[int, int], resolution: int):
        """
        QDアーカイブ（2次元グリッド）
        各セルには最高スコアの解（idea_id, quality, measures）を保持
        """
        self.name = name
        self.grid: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.bc1_range = bc1_range
        self.bc2_range = bc2_range
        self.resolution = resolution

    def _discretize(self, bc1: float, bc2: float) -> Tuple[int, int]:
        """
        BC値をグリッド座標に変換
        """
        x = int((bc1 - self.bc1_range[0]) / (self.bc1_range[1] - self.bc1_range[0]) * (self.resolution - 1))
        y = int((bc2 - self.bc2_range[0]) / (self.bc2_range[1] - self.bc2_range[0]) * (self.resolution - 1))
        return max(0, min(x, self.resolution - 1)), max(0, min(y, self.resolution - 1))

    def add(self, idea_id: int, quality: float, measures: Tuple[float, float]):
        """
        指定座標にアイデアを登録（より高スコアであれば上書き）
        """
        x, y = self._discretize(*measures)
        key = (x, y)
        current = self.grid.get(key)
        if current is None or quality > current["quality"]:
            self.grid[key] = {"idea_id": idea_id, "quality": quality, "measures": measures}

    def get_idea_at(self, measures: Tuple[float, float]) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """
        指定座標（BC値）にアイデアが存在するか確認し、存在すればその内容とそのqualityスコアを返す。
        存在しなければ (None, None) を返す。

        Returns:
            (idea_dict, quality_value) or (None, None)
        """
        x, y = self._discretize(*measures)
        key = (x, y)
        current = self.grid.get(key)
        print(f"current: {current}")

        if current is None:
            return None, None

        quality = current.get("quality")
        print(f"quality: {quality}")
        return current, quality

    def overwrite_idea_at(self, idea_id: int, quality: float, measures: Tuple[float, float]):
        """
        指定座標にアイデアを強制的に上書きする（スコアに関係なく）。
        """
        x, y = self._discretize(*measures)
        self.grid[(x, y)] = {"idea_id": idea_id, "quality": quality, "measures": measures}

    def sample_elites(self, n: int):
        """
        グリッドに存在するエリートからランダムにn件をサンプリング
        """
        return random.sample(list(self.grid.values()), min(n, len(self.grid)))

    def all_idea_ids(self):
        """
        登録されている全アイデアのIDを返す
        """
        return [entry["idea_id"] for entry in self.grid.values()]

    def get_grid(self):
        """
        可視化用などにグリッド全体を返す
        """
        return self.grid
