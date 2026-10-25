"""Name linkage between the shortage list's firm units and the FDA inspection dataset (analysis/inspections.py)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.inspections import match_firm, match_firm_first_hit  # noqa: E402

NAMES = {"EUGIA SPECIALITIES", "MYLAN", "HOSPIRA", "INTAS", "STERI", "MEDEFIL INCORPORATED", "SUN", "SUN RISE", "PFIZER", "BAXTER"}


def test_exact_match_first():
    assert match_firm("HOSPIRA", NAMES) == ({"HOSPIRA", "PFIZER"}, "exact+parent")   # own name and the Pfizer parent
    assert match_firm("SUN", NAMES) == ({"SUN"}, "exact")                 # exact match even for a short name


def test_parent_match_through_corporate_group():
    assert match_firm("ACCORD", NAMES) == ({"INTAS"}, "parent")           # Accord Healthcare -> Intas
    hits, rule = match_firm("EUGIA", NAMES)                                 # Eugia -> Aurobindo is not in NAMES, so the prefix rule applies
    assert rule == "prefix" and hits == {"EUGIA SPECIALITIES"}


def test_rules_are_a_union_not_a_precedence():
    names = NAMES | {"ACCORD", "AUROBINDO"}
    hits, rule = match_firm("ACCORD", names)                                # Accord's own establishments AND Intas's
    assert hits == {"ACCORD", "INTAS"} and rule == "exact+parent"
    assert match_firm_first_hit("ACCORD", names) == {"ACCORD"}             # the first-hit rule would have stopped at Accord
    hits, rule = match_firm("EUGIA", names)                                 # Eugia's own establishments AND Aurobindo's
    assert hits == {"EUGIA SPECIALITIES", "AUROBINDO"} and rule == "parent+prefix"


def test_word_prefix_both_directions_and_length_guard():
    assert match_firm("MYLAN INSTITUTIONAL", NAMES) == ({"MYLAN"}, "prefix")
    assert match_firm("MEDEFIL", NAMES) == ({"MEDEFIL INCORPORATED"}, "prefix")
    assert match_firm("STERIMAX", NAMES) == (set(), None)                   # STERI is a substring, not a word prefix
    assert match_firm("SUN", {"SUN RISE"}) == (set(), None)                # three-letter names match only exactly
    assert match_firm("", NAMES) == (set(), None)
