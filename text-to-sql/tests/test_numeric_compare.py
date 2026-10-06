"""The epsilon matcher on its own: one-to-one matching, the disk index path, and its budgets.

`compare_results` reaches this module with rows already reduced to typed cell keys, so these tests build the same
keys with `sqlite_exec._normalize_row`.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_to_sql_recipe import numeric_compare, sqlite_exec

MATCH = "results match with absolute numeric tolerance 1e-6"
MISMATCH = "result values or row cardinalities differ"


def keys(*rows):
    return [sqlite_exec._normalize_row(tuple(row)) for row in rows]


def compare(expected, actual, method="set", **kwargs):
    return numeric_compare.compare_numeric_rows(keys(*expected), keys(*actual), method=method, **kwargs)


class MaterializedMatchTest(unittest.TestCase):
    def test_exact_sets_match_without_an_index(self):
        self.assertEqual(compare([(1, "a"), (2, "b")], [(2, "b"), (1, "a")]), (True, MATCH))

    def test_cardinality_is_exact_even_within_epsilon(self):
        """Two gold rows 5e-7 apart are two rows: one predicted row cannot stand for both."""
        self.assertEqual(compare([(1.0,), (1.0000005,)], [(1.0000002,)]), (False, MISMATCH))

    def test_single_values_within_epsilon_match(self):
        self.assertTrue(compare([(0.3,)], [(0.1 + 0.2,)])[0])
        self.assertFalse(compare([(0.3,)], [(0.300002,)])[0])

    def test_a_subset_may_not_be_larger_than_the_gold_set(self):
        self.assertFalse(compare([(1,)], [(1,), (2,)], "subset,>=,1")[0])


class IndexedMatchTest(unittest.TestCase):
    """Rows that are close but not identical go through the on-disk matcher."""

    def test_near_equal_scalar_populations_match_one_to_one(self):
        expected = [(1.0,), (2.0,), (3.0,)]
        actual = [(1.0000004,), (2.0000004,), (2.9999996,)]
        self.assertEqual(compare(expected, actual), (True, MATCH))
        self.assertEqual(compare(expected, [(1.0000004,), (2.0000004,), (3.0000020,)]), (False, MISMATCH))

    def test_multiset_matching_is_injective(self):
        self.assertTrue(compare([(1.0,), (1.0000005,)], [(1.0000002,), (1.0000002,)], "multiset")[0])
        self.assertFalse(compare([(1.0,), (5.0,)], [(1.0000002,), (1.0000002,)], "multiset")[0])

    def test_augmenting_paths_reassign_an_earlier_match(self):
        """A greedy first fit gives the first predicted row the first gold row and strands the second; the
        matcher must move the first one over instead of rejecting a valid one-to-one matching."""
        expected = [("t", 0.0, 0.0), ("t", 0.0, 1.5e-6)]
        actual = [("t", 0.0, 0.75e-6), ("t", 0.0, 0.0000000001)]
        self.assertEqual(compare(expected, actual), (True, MATCH))

    def test_text_cells_must_match_exactly_within_a_bucket(self):
        self.assertFalse(compare([("a", 1.0), ("b", 2.0)], [("a", 1.0000001), ("c", 2.0)])[0])

    def test_subsets_inject_predicted_rows_into_gold_rows(self):
        expected = [(1.0,), (2.0,), (3.0,)]
        self.assertTrue(compare(expected, [(1.0000001,), (3.0000001,)], "subset,=,2")[0])
        self.assertFalse(compare(expected, [(1.0000001,), (4.0,)], "subset,=,2")[0])

    def test_ordered_results_compare_position_by_position(self):
        self.assertTrue(compare([(1.0,), (2.0,)], [(1.0000001,), (2.0000001,)], "list")[0])
        self.assertFalse(compare([(1.0,), (2.0,)], [(2.0000001,), (1.0000001,)], "ordered")[0])
        self.assertFalse(compare([(1.0,), (2.0,)], [(1.0,)], "list")[0])


class BudgetTest(unittest.TestCase):
    EXPECTED = tuple((float(i),) for i in range(50))
    ACTUAL = tuple((float(i) + 5e-7,) for i in range(50))

    def test_time_budget(self):
        with self.assertRaisesRegex(numeric_compare.NumericComparisonError, "exceeded its time budget"):
            compare(self.EXPECTED, self.ACTUAL, timeout_seconds=0)

    def test_disk_budget(self):
        with self.assertRaisesRegex(numeric_compare.NumericComparisonError, "exceeded its disk budget"):
            compare(self.EXPECTED, self.ACTUAL, max_index_bytes=1)

    def test_index_lives_under_the_given_root(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertTrue(compare(self.EXPECTED, self.ACTUAL, temp_root=root)[0])
            self.assertEqual(list(Path(root).iterdir()), [])


class MethodValidationTest(unittest.TestCase):
    def test_unknown_and_malformed_methods_are_rejected(self):
        for method, message in (
            ("jaccard", "unsupported SQL grading method"),
            ("bird_set", "unsupported SQL grading method"),
            ("subset", "unsupported SQL grading method"),
            ("subset,>,1", "invalid subset grading method"),
            ("subset,>=", "invalid subset grading method"),
            ("subset,>=,many", "invalid subset grading method"),
        ):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, message):
                compare([(1,)], [(1,)], method)


if __name__ == "__main__":
    unittest.main()
