"""Offline tests for incremental.py (partition diff / upsert / date-delta helpers)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incremental import append_new_dates, incremental_merge, partition_diff  # noqa: E402


# ── partition_diff ───────────────────────────────────────────────────────────
def test_partition_diff_single_key_added_removed_changed():
    prev = pd.DataFrame({"k": ["a", "b", "c"], "v": [1, 2, 3]})
    curr = pd.DataFrame({"k": ["b", "c", "d"], "v": [2, 30, 4]})
    out = partition_diff(prev, curr, "k")
    assert out == {"added": ["d"], "removed": ["a"], "changed": ["c"],
                   "n_added": 1, "n_removed": 1, "n_changed": 1}


def test_partition_diff_identical_frames_reports_nothing():
    df = pd.DataFrame({"k": [1, 2], "v": ["x", "y"]})
    out = partition_diff(df, df.copy(), "k")
    assert out["added"] == [] and out["removed"] == [] and out["changed"] == []
    assert (out["n_added"], out["n_removed"], out["n_changed"]) == (0, 0, 0)


def test_partition_diff_composite_key_returns_tuples_sorted():
    prev = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "v": [1, 2, 3]})
    curr = pd.DataFrame({"a": [1, 3], "b": ["y", "z"], "v": [20, 40]})
    out = partition_diff(prev, curr, ["a", "b"])
    assert out["added"] == [(3, "z")]
    assert out["removed"] == [(1, "x"), (2, "x")]
    assert out["changed"] == [(1, "y")]


def test_partition_diff_list_key_of_one_column():
    prev = pd.DataFrame({"k": [1, 2], "v": [1, 2]})
    curr = pd.DataFrame({"k": [2, 3], "v": [2, 3]})
    out = partition_diff(prev, curr, ["k"])
    assert (out["added"], out["removed"], out["changed"]) == ([3], [1], [])


def test_partition_diff_prev_empty_everything_added():
    prev = pd.DataFrame({"k": [], "v": []})
    curr = pd.DataFrame({"k": ["b", "a"], "v": [1, 2]})
    out = partition_diff(prev, curr, "k")
    assert out["added"] == ["a", "b"]          # sorted
    assert out["removed"] == [] and out["changed"] == []
    assert out["n_added"] == 2


def test_partition_diff_curr_empty_everything_removed():
    prev = pd.DataFrame({"k": ["b", "a"], "v": [1, 2]})
    curr = pd.DataFrame({"k": [], "v": []})
    out = partition_diff(prev, curr, "k")
    assert out["removed"] == ["a", "b"]
    assert out["added"] == [] and out["n_removed"] == 2


def test_partition_diff_both_empty():
    e = pd.DataFrame({"k": [], "v": []})
    out = partition_diff(e, e.copy(), "k")
    assert out == {"added": [], "removed": [], "changed": [],
                   "n_added": 0, "n_removed": 0, "n_changed": 0}


def test_partition_diff_nan_equal_nan_is_not_a_change():
    prev = pd.DataFrame({"k": [1, 2], "v": [float("nan"), 1.0]})
    curr = pd.DataFrame({"k": [1, 2], "v": [float("nan"), 2.0]})
    out = partition_diff(prev, curr, "k")
    assert out["changed"] == [2]


def test_partition_diff_multi_column_change_detected_in_any_column():
    prev = pd.DataFrame({"k": [1, 2], "a": [1, 1], "b": ["x", "x"]})
    curr = pd.DataFrame({"k": [1, 2], "a": [1, 1], "b": ["x", "y"]})
    assert partition_diff(prev, curr, "k")["changed"] == [2]


def test_partition_diff_mixed_type_keys_sorted_by_str():
    prev = pd.DataFrame({"k": [], "v": []})
    curr = pd.DataFrame({"k": [10, 9, 100], "v": [1, 2, 3]})
    # sorted by str(): '10' < '100' < '9'
    assert partition_diff(prev, curr, "k")["added"] == [10, 100, 9]


# ── incremental_merge ────────────────────────────────────────────────────────
def test_merge_upserts_keeps_base_order_and_appends_new():
    base = pd.DataFrame({"k": ["c", "a", "b"], "v": [1, 2, 3]})
    new = pd.DataFrame({"k": ["b", "z", "a"], "v": [30, 40, 20]})
    out = incremental_merge(base, new, "k")
    assert out["k"].tolist() == ["c", "a", "b", "z"]
    assert out["v"].tolist() == [1, 20, 30, 40]
    assert list(out.index) == [0, 1, 2, 3]       # index reset


def test_merge_does_not_mutate_inputs():
    base = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]})
    new = pd.DataFrame({"k": ["a"], "v": [9]})
    b0, n0 = base.copy(), new.copy()
    incremental_merge(base, new, "k")
    pd.testing.assert_frame_equal(base, b0)
    pd.testing.assert_frame_equal(new, n0)


def test_merge_drop_removed_keeps_only_new_keys_in_base_order():
    base = pd.DataFrame({"k": ["c", "b", "a"], "v": [1, 2, 3]})
    new = pd.DataFrame({"k": ["a", "b"], "v": [30, 20]})
    out = incremental_merge(base, new, "k", drop_removed=True)
    assert out["k"].tolist() == ["b", "a"]
    assert out["v"].tolist() == [20, 30]


def test_merge_drop_removed_false_keeps_removed_rows():
    base = pd.DataFrame({"k": ["c", "b"], "v": [1, 2]})
    new = pd.DataFrame({"k": ["b"], "v": [20]})
    out = incremental_merge(base, new, "k")
    assert out["k"].tolist() == ["c", "b"]


def test_merge_empty_base_returns_independent_copy_of_new():
    base = pd.DataFrame({"k": [], "v": []})
    new = pd.DataFrame({"k": ["a"], "v": [1]})
    out = incremental_merge(base, new, "k")
    pd.testing.assert_frame_equal(out, new)
    out.loc[0, "v"] = 99
    assert new.loc[0, "v"] == 1


def test_merge_empty_new_returns_independent_copy_of_base():
    base = pd.DataFrame({"k": ["a"], "v": [1]})
    new = pd.DataFrame({"k": [], "v": []})
    out = incremental_merge(base, new, "k")
    pd.testing.assert_frame_equal(out, base)
    out.loc[0, "v"] = 99
    assert base.loc[0, "v"] == 1


def test_merge_empty_new_ignores_drop_removed():
    # documented early return: an empty `new` never wipes the base
    base = pd.DataFrame({"k": ["a"], "v": [1]})
    new = pd.DataFrame({"k": [], "v": []})
    assert len(incremental_merge(base, new, "k", drop_removed=True)) == 1


def test_merge_composite_key():
    base = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "v": [1, 2, 3]})
    new = pd.DataFrame({"a": [1, 3], "b": ["y", "z"], "v": [20, 40]})
    out = incremental_merge(base, new, ["a", "b"])
    assert out[["a", "b", "v"]].values.tolist() == [
        [1, "x", 1], [1, "y", 20], [2, "x", 3], [3, "z", 40]]


def test_merge_column_order_of_new_does_not_matter():
    base = pd.DataFrame({"k": ["a", "b"], "v": [1, 2], "w": [1.5, 2.5]})
    new = pd.DataFrame({"w": [9.5], "v": [7], "k": ["b"]})
    out = incremental_merge(base, new, "k")
    row = out[out["k"] == "b"].iloc[0]
    assert (row["v"], row["w"]) == (7, 9.5)


# ── append_new_dates ─────────────────────────────────────────────────────────
def test_append_new_dates_only_strictly_newer_rows():
    base = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "v": [1, 2]})
    new = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                        "v": [10, 20, 3]})
    out = append_new_dates(base, new)
    assert out["v"].tolist() == [1, 2, 3]          # equal-date rows are NOT re-added
    assert list(out.index) == [0, 1, 2]


def test_append_new_dates_string_dates_are_coerced():
    base = pd.DataFrame({"Date": ["2024-01-02"], "v": [1]})
    new = pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"], "v": [5, 6]})
    out = append_new_dates(base, new)
    assert out["v"].tolist() == [1, 6]


def test_append_new_dates_nothing_new_returns_base_rows():
    base = pd.DataFrame({"Date": pd.to_datetime(["2024-01-05"]), "v": [1]})
    new = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "v": [9]})
    assert append_new_dates(base, new)["v"].tolist() == [1]


def test_append_new_dates_empty_base_returns_copy_of_new():
    base = pd.DataFrame({"Date": [], "v": []})
    new = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "v": [1]})
    out = append_new_dates(base, new)
    pd.testing.assert_frame_equal(out, new)
    out.loc[0, "v"] = 5
    assert new.loc[0, "v"] == 1


def test_append_new_dates_custom_date_column():
    base = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"]), "v": [1]})
    new = pd.DataFrame({"d": pd.to_datetime(["2024-01-02"]), "v": [2]})
    assert append_new_dates(base, new, date_col="d")["v"].tolist() == [1, 2]


def test_append_new_dates_per_key_cutoffs_and_unknown_key():
    base = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-05", "2024-01-02"]),
        "sym": ["A", "B"], "v": [1, 2]})
    new = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-03", "2024-01-03", "2024-01-05", "2024-01-01"]),
        "sym": ["A", "B", "A", "C"], "v": [10, 20, 30, 40]})
    out = append_new_dates(base, new, key_col="sym")
    # A@01-03 (older than A's 01-05) dropped; A@01-05 equal -> dropped;
    # B@01-03 newer than B's 01-02 kept; C unknown -> kept even though old
    assert out["v"].tolist() == [1, 2, 20, 40]


def test_append_new_dates_key_col_empty_new_is_noop():
    base = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02"]), "sym": ["A"], "v": [1]})
    new = pd.DataFrame({"Date": pd.to_datetime([]), "sym": [], "v": []})
    out = append_new_dates(base, new, key_col="sym")
    assert out["v"].tolist() == [1]


def test_append_new_dates_empty_new_without_key_is_noop():
    base = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02"]), "v": [1]})
    new = pd.DataFrame({"Date": pd.to_datetime([]), "v": []})
    assert append_new_dates(base, new)["v"].tolist() == [1]
