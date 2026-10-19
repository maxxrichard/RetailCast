import numpy as np

from retailcast.data.hierarchy import build_hierarchy
from retailcast.models.base import to_wide


def test_summing_matrix_structure(panel):
    meta = panel.drop_duplicates("series_id")[["series_id", "item_id", "cat_id", "store_id"]]
    h = build_hierarchy(meta)
    n_b = 24
    assert h.S.shape == (1 + 2 + 6 + n_b, n_b)
    assert h.level_of["total"] == "total"
    assert len(h.nodes_at("store_id")) == 2
    assert len(h.nodes_at("store_id/cat_id")) == 6
    # bottom block is identity, total row is all ones
    np.testing.assert_array_equal(h.S[0], np.ones(n_b))
    np.testing.assert_array_equal(h.S[-n_b:], np.eye(n_b))


def test_aggregate_is_coherent(panel):
    meta = panel.drop_duplicates("series_id")[["series_id", "item_id", "cat_id", "store_id"]]
    h = build_hierarchy(meta)
    wide = to_wide(panel)
    nodes = h.aggregate(wide)
    np.testing.assert_allclose(nodes["total"], wide.sum(axis=1))
    for s in ["STORE_0", "STORE_1"]:
        cols = [c for c in wide.columns if c.endswith(s)]
        np.testing.assert_allclose(nodes[f"store_id={s}"], wide[cols].sum(axis=1))
