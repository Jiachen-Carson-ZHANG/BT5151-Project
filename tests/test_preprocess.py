import pandas as pd
import pytest

from bt5151_credit_risk.preprocess import preprocess_credit_data


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        [
            {
                "ID": "0x1",
                "Customer_ID": "CUS_A",
                "Name": "Alice",
                "SSN": "111-11-1111",
                "Credit_Score": "Good",
                "Age": 25,
                "Outstanding_Debt": 1000.0,
            },
            {
                "ID": "0x2",
                "Customer_ID": "CUS_A",
                "Name": "Alice",
                "SSN": "111-11-1111",
                "Credit_Score": "Standard",
                "Age": 26,
                "Outstanding_Debt": 1500.0,
            },
            {
                "ID": "0x3",
                "Customer_ID": "CUS_B",
                "Name": "Bob",
                "SSN": "222-22-2222",
                "Credit_Score": "Poor",
                "Age": 40,
                "Outstanding_Debt": 3000.0,
            },
            {
                "ID": "0x4",
                "Customer_ID": "CUS_B",
                "Name": "Bob",
                "SSN": "222-22-2222",
                "Credit_Score": "Poor",
                "Age": 41,
                "Outstanding_Debt": 3200.0,
            },
        ]
    )


def test_preprocess_drops_identifier_columns(sample_frame):
    result = preprocess_credit_data(sample_frame)
    assert "ID" not in result.feature_frame.columns
    assert "Customer_ID" not in result.feature_frame.columns
    assert "Name" not in result.feature_frame.columns
    assert "SSN" not in result.feature_frame.columns
    assert "Credit_Score" not in result.feature_frame.columns


def test_preprocess_uses_group_split(sample_frame):
    result = preprocess_credit_data(sample_frame)
    train_groups = set(result.train_groups)
    test_groups = set(result.test_groups)
    assert train_groups.isdisjoint(test_groups)
