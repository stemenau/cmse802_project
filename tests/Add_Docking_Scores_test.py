import os
import pandas as pd
import pytest
import numpy as np
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the function
from Add_Docking_Scores import add_docking_scores


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def main_csv(tmp_path):
    """Creates a main CSV with sample data."""
    df = pd.DataFrame({
        'Canonical SMILES': ['C', 'CCO', 'CCC', 'CCN'],
        'Name': ['methane', 'ethanol', 'propane', 'ethylamine']
    })
    csv_file = tmp_path / "data_preprocessed.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def docking_scores_csv(tmp_path):
    """Creates a docking scores CSV with Index mapping."""
    df = pd.DataFrame({
        'Index': [0, 1, 3],  # deliberately skip 2 to test NaN handling
        'S': [-5.2, -6.0, -4.7]
    })
    csv_file = tmp_path / "docking_scores.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


# ----------------------------------------------------------------------
# Tests for add_docking_scores()
# ----------------------------------------------------------------------

def test_add_docking_scores_adds_column(tmp_path, main_csv, docking_scores_csv):
    """Check that the 'Docking Score' column is added and values match."""
    os.chdir(tmp_path)  # isolate output
    df_updated = add_docking_scores(main_csv, docking_scores_csv)

    # Check type
    assert isinstance(df_updated, pd.DataFrame)
    assert 'Docking Score' in df_updated.columns

    # Check that scores are assigned correctly
    expected_scores = pd.Series([-5.2, -6.0, -4.7], index=[0,1,3])
    for idx, score in expected_scores.items():
        assert df_updated.loc[idx, 'Docking Score'] == score

    # Row 2 should be dropped because no docking score
    assert 2 not in df_updated.index

    # Output file created
    assert os.path.exists("data_complete.csv")


def test_add_docking_scores_all_rows_missing(tmp_path, main_csv):
    """If docking_scores CSV has no matching indices, all rows should be dropped."""
    docking_csv = tmp_path / "empty_docking.csv"
    pd.DataFrame({'Index':[10,11], 'S':[-5.0, -4.0]}).to_csv(docking_csv, index=False)

    os.chdir(tmp_path)
    df_result = add_docking_scores(main_csv, docking_csv)

    assert df_result.empty
    assert os.path.exists("data_complete.csv")


def test_add_docking_scores_handles_non_sequential_indices(tmp_path, main_csv):
    """Docking scores with non-sequential indices should align correctly."""
    docking_csv = tmp_path / "nonseq_docking.csv"
    pd.DataFrame({'Index':[3,0], 'S':[-4.7, -5.2]}).to_csv(docking_csv, index=False)

    os.chdir(tmp_path)
    df_result = add_docking_scores(main_csv, docking_csv)

    # Only indices 0 and 3 should exist
    assert set(df_result.index) == {0,3}
    assert df_result.loc[0, 'Docking Score'] == -5.2
    assert df_result.loc[3, 'Docking Score'] == -4.7


def test_add_docking_scores_preserves_other_columns(tmp_path, main_csv, docking_scores_csv):
    """Ensure original columns remain unchanged after adding docking scores."""
    os.chdir(tmp_path)
    df_result = add_docking_scores(main_csv, docking_scores_csv)

    assert 'Canonical SMILES' in df_result.columns
    assert 'Name' in df_result.columns