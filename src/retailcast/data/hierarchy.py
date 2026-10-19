"""Hierarchy construction: summing matrix S and aggregation of a bottom-level panel.

Levels follow the M5 convention: total -> store -> category(within store) -> item x store.
Any list of grouping columns works; ``levels`` are ordered coarse -> fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Hierarchy:
    bottom_ids: list[str]
    node_ids: list[str]  # all nodes in order: total, level-1 ..., bottom
    level_of: dict[str, str]
    S: np.ndarray  # (n_nodes, n_bottom) summing matrix
    bottom_to_nodes: dict[str, list[str]]
    node_keys: dict[str, dict[str, str]] = field(default_factory=dict)  # grouping values per node

    @property
    def upper_ids(self) -> list[str]:
        return [n for n in self.node_ids if self.level_of[n] != "bottom"]

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
    node_keys: dict[str, dict[str, str]] = {}

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
        for u in uniq:
            j = keys.tolist().index(u)
            node_keys[u] = {c: str(meta.loc[j, c]) for c in cols}

    blocks.append(np.eye(n_b, dtype=np.float32))
    node_ids.extend(bottom_ids)
    level_of.update({b: "bottom" for b in bottom_ids})
    S = np.vstack(blocks)

    bottom_to_nodes: dict[str, list[str]] = {}
    for j, b in enumerate(bottom_ids):
        bottom_to_nodes[b] = [node_ids[i] for i in np.nonzero(S[:, j])[0]]

    for j, b in enumerate(bottom_ids):
        node_keys[b] = {c: str(meta.loc[j, c]) for c in meta.columns if c != bottom_col}
    return Hierarchy(bottom_ids, node_ids, level_of, S, bottom_to_nodes, node_keys)


def upper_panel(panel: pd.DataFrame, hier: Hierarchy) -> pd.DataFrame:
    """Long panel of the non-bottom nodes (aggregated demand, volume-weighted price, promo share).

    Upper nodes get ``series_id = item_id = node_id``; grouping columns that the node does not
    fix are set to ``"ALL"`` so the same feature pipeline and models can be applied.
    """
    y = panel.pivot(index="date", columns="series_id", values="y").fillna(0.0)
    y = y.reindex(columns=hier.bottom_ids)
    upper = hier.upper_ids
    y_n = hier.aggregate(y)[upper]
    if "sell_price" in panel and panel["sell_price"].notna().any():
        p = panel.pivot(index="date", columns="series_id", values="sell_price").reindex(
            columns=hier.bottom_ids
        )
        p = p.ffill().bfill().fillna(0.0)
        rev = hier.aggregate(y * p)[upper]
        n_sub = pd.Series(hier.S.sum(axis=1), index=hier.node_ids)[upper]
        price_n = (rev / y_n.replace(0.0, np.nan)).fillna(hier.aggregate(p)[upper] / n_sub)
    else:
        price_n = pd.DataFrame(np.nan, index=y.index, columns=upper)
    promo = panel.pivot(index="date", columns="series_id", values="promo").reindex(
        columns=hier.bottom_ids
    )
    promo_share = (
        hier.aggregate(promo.fillna(0).astype(float))[upper]
        / pd.Series(hier.S.sum(axis=1), index=hier.node_ids)[upper]
    )
    rows = []
    for node in upper:
        keys = hier.node_keys.get(node, {})
        rows.append(
            pd.DataFrame(
                {
                    "series_id": node,
                    "item_id": node,
                    "cat_id": keys.get("cat_id", "ALL"),
                    "store_id": keys.get("store_id", "ALL"),
                    "date": y_n.index,
                    "y": y_n[node].to_numpy(dtype="float32"),
                    "sell_price": price_n[node].to_numpy(dtype="float32"),
                    "promo": (promo_share[node] >= 0.5).to_numpy().astype("int8"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
