"""Firm units (analysis/common.py): the marketer-of-record unit merges a company's labeler codes and counts a presentation
reported by a contract manufacturer under the company whose NDC it carries."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.common import load_cells, normalise_company, PROC  # noqa: E402

HAVE_DATA = os.path.exists(os.path.join(PROC, "presentations.csv"))


def test_normalise_company_merges_spellings():
    assert normalise_company("Meitheal Pharmaceuticals Inc.") == normalise_company("Meitheal Pharmaceuticals, Inc.") == "MEITHEAL"
    assert normalise_company("Hospira, Inc., a Pfizer Company") == "HOSPIRA"


@pytest.mark.skipif(not HAVE_DATA, reason="processed data not built")
def test_marketer_unit_partitions_presentations():
    for unit in ("company", "marketer", "labeler"):
        cells, pres, _, _ = load_cells(unit)
        assert cells["n_presentations"].sum() == len(pres)                       # every presentation in exactly one cell
        assert not cells.duplicated(["firm_unit", "drug"]).any()
    cm, pm, _, _ = load_cells("marketer")
    # every presentation's marketer is the normalised NDC labeler name, whatever company reported it
    assert (pm["firm_unit"] == pm["firm_name"].fillna(pm["company_name"]).astype(str).map(normalise_company)).all()
