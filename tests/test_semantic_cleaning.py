import math

import pandas as pd

from bt5151_credit_risk.semantic_cleaning import allowed_cleaning_primitives
from bt5151_credit_risk.semantic_cleaning import fill_numeric_by_group_then_global
from bt5151_credit_risk.semantic_cleaning import multi_hot_membership
from bt5151_credit_risk.semantic_cleaning import parse_age_series
from bt5151_credit_risk.semantic_cleaning import parse_duration_months
from bt5151_credit_risk.semantic_cleaning import parse_dirty_numeric


def test_parse_dirty_numeric_handles_currency_and_artifacts():
    series = pd.Series(["$1,200.50", "45_", None, "bad"])

    parsed = parse_dirty_numeric(series)

    assert parsed.iloc[0] == 1200.50
    assert parsed.iloc[1] == 45.0
    assert math.isnan(parsed.iloc[2])
    assert math.isnan(parsed.iloc[3])


def test_parse_age_and_duration_months_preserve_signal():
    ages = parse_age_series(pd.Series(["23", "45_", "-500", None]))
    months = parse_duration_months(pd.Series(["22 Years and 7 Months", "3 Years", "18", None]))

    assert ages.tolist()[:3] == [23.0, 45.0, -500.0]
    assert math.isnan(ages.iloc[3])
    assert months.tolist()[:3] == [271.0, 36.0, 18.0]
    assert math.isnan(months.iloc[3])


def test_multi_hot_membership_merges_duplicates_and_tracks_missing():
    frame, missing = multi_hot_membership(
        pd.Series([None, "Home Loan, Personal Loan", " and Home Loan , Home Loan "]),
        column_name="Type_of_Loan",
    )

    assert missing.tolist() == [1, 0, 0]
    assert list(frame.columns) == [
        "Type_of_Loan_Home Loan",
        "Type_of_Loan_Personal Loan",
    ]
    assert frame["Type_of_Loan_Home Loan"].tolist() == [0, 1, 1]
    assert frame["Type_of_Loan_Personal Loan"].tolist() == [0, 1, 0]


def test_fill_numeric_by_group_then_global_fills_group_first():
    values = pd.Series([10.0, None, 30.0, None])
    groups = pd.Series(["A", "A", "B", "C"])

    filled = fill_numeric_by_group_then_global(values, groups, default=99.0)

    assert filled.tolist() == [10.0, 10.0, 30.0, 20.0]


def test_allowed_cleaning_primitives_exposes_expected_catalog():
    catalog = allowed_cleaning_primitives()

    assert "parse_dirty_numeric" in catalog
    assert "parse_age_series" in catalog
    assert "parse_duration_months" in catalog
    assert "multi_hot_membership" in catalog
    assert "fill_numeric_by_group_then_global" in catalog
