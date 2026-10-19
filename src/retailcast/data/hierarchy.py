"""Hierarchy construction: summing matrix S and aggregation of a bottom-level panel.

Levels follow the M5 convention: total -> store -> category(within store) -> item x store.
Any list of grouping columns works; ``levels`` are ordered coarse -> fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Hierarchy:
    bottom_ids: list[str]
    node_ids: list[str]  # all nodes in order: total, level-1 ..., bottom
    level_of: dict[str, str]
    S: np.ndarray  # (n_nodes, n_bottom) summing matrix
    bottom_to_nodes: dict[str, list[str]]

    @property
    def n_bottom(self) -> int:
        return len(self.bottom_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def nodes_at(self, level: str) -> list[str]:
        return [n for n in self.node_ids if self.level_of[n] == level]

    def aggregate(self, bottom: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a (date x bottom_id) wide frame to all nodes (date x node_id)."""
        b = bottom.reindex(columns=self.bottom_ids).fillna(0.0).to_numpy(dtype=float)
        agg = b @ self.S.T
        return pd.DataFrame(agg, index=bottom.index, columns=self.node_ids)


def build_hierarchy(
    meta: pd.DataFrame,
    levels: list[list[str]] | None = None,
    bottom_col: str = "series_id",
) -> Hierarchy:
    """Build the summing matrix from a per-series metadata frame.

    Parameters
    ----------
    meta
        One row per bottom series with the grouping columns.
    levels
        Each entry is a list of columns defining one aggregation level, coarse -> fine.
        Default: ``[[], ["store_id"], ["store_id", "cat_id"]]`` (plus the bottom level).
    """
    if levels is None:
        levels = [[], ["store_id"], ["store_id", "cat_id"]]
    meta = meta.drop_duplicates(bottom_col).reset_index(drop=True)
    bottom_ids = meta[bottom_col].astype(str).tolist()
    n_b = len(bottom_ids)

    node_ids: list[str] = []
    level_of: dict[str, str] = {}
    blocks: list[np.ndarray] = []

    for cols in levels:
        lvl_name = "total" if not cols else "/".join(cols)
        if not cols:
            keys = pd.Series(["total"] * n_b)
        else:
            keys = meta[cols].astype(str).agg("|".join, axis=1)
            keys = lvl_name + "=" + keys
        uniq = list(dict.fromkeys(keys.tolist()))
        block = np.zeros((len(uniq), n_b), dtype=np.float32)
        idx = {k: i for i, k in enumerate(uniq)}
        for j, k in enumerate(keys):
            block[idx[k], j] = 1.0
        blocks.append(block)
        node_ids.extend(uniq)
        level_of.update({u: lvl_name for u in uniq})

    blocks.append(np.eye(n_b, dtype=np.float32))
    node_ids.extend(bottom_ids)
    level_of.update({b: "bottom" for b in bottom_ids})
    S = np.vstack(blocks)

    bottom_to_nodes: dict[str, list[str]] = {}
    for j, b in enumerate(bottom_ids):
        bottom_to_nodes[b] = [node_ids[i] for i in np.nonzero(S[:, j])[0]]

    return Hierarchy(bottom_ids, node_ids, level_of, S, bottom_to_nodes)
