import os
import pandas as pd
import pytest
from rdkit import Chem

# Import functions from your module
from Data_Prep import (
    get_canonical_smiles,
    clean_data,
    csv_to_sdf,
    generate_descriptors,
)

# ----------------------------------------------------------------------
# Fixtures and setup
# ----------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a temporary CSV file with valid and invalid SMILES for testing."""
    df = pd.DataFrame({
        'SMILES': ['C', 'CCO', 'invalid_smiles', None, 'C'],  # duplicate + invalid + missing
        'Name': ['methane', 'ethanol', 'bad', 'none', 'duplicate']
    })
    csv_file = tmp_path / "pfas_sample.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def cleaned_csv(tmp_path):
    """Creates a minimal valid cleaned CSV with canonical SMILES."""
    df = pd.DataFrame({'Canonical SMILES': ['C', 'CCO']})
    csv_file = tmp_path / "data_cleaned.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


# ----------------------------------------------------------------------
# Tests for get_canonical_smiles()
# ----------------------------------------------------------------------

def test_get_canonical_smiles_valid():
    """Valid SMILES should return canonical representation."""
    result = get_canonical_smiles("CCO")
    assert result == "CCO", "Canonical SMILES for ethanol should be 'CCO'"


def test_get_canonical_smiles_invalid():
    """Invalid SMILES should return None."""
    assert get_canonical_smiles("not_a_smiles") is None


def test_get_canonical_smiles_non_string():
    """Non-string input should return None."""
    assert get_canonical_smiles(123) is None


# ----------------------------------------------------------------------
# Tests for clean_data()
# ----------------------------------------------------------------------

def test_clean_data_creates_expected_output(tmp_path, sample_csv):
    """Check that clean_data removes invalid entries, duplicates, and outputs expected CSV."""
    os.chdir(tmp_path)  # Ensure test isolation
    df_cleaned = clean_data(sample_csv)

    # Assertions on returned DataFrame
    assert isinstance(df_cleaned, pd.DataFrame)
    assert 'Canonical SMILES' in df_cleaned.columns
    assert all(df_cleaned['Canonical SMILES'].apply(lambda x: isinstance(x, str)))
    assert len(df_cleaned) == 2  # Only 'C' and 'CCO' remain after cleaning

    # Check that output file exists
    assert os.path.exists("data_cleaned.csv")


# ----------------------------------------------------------------------
# Tests for csv_to_sdf()
# ----------------------------------------------------------------------

def test_csv_to_sdf_success(tmp_path, cleaned_csv):
    """Check that valid CSV produces an SDF with correct molecule count."""
    os.chdir(tmp_path)
    n_written = csv_to_sdf(cleaned_csv)
    assert n_written == 2, "Expected 2 molecules written to SDF"
    assert os.path.exists("data_cleaned.sdf")


def test_csv_to_sdf_missing_column(tmp_path):
    """Should raise ValueError if 'Canonical SMILES' column missing."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": ["C", "CCO"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        csv_to_sdf(bad_csv)


# ----------------------------------------------------------------------
# Tests for generate_descriptors()
# ----------------------------------------------------------------------

def test_generate_descriptors_adds_expected_columns(tmp_path, cleaned_csv):
    """Generated descriptors should add expected numeric columns."""
    os.chdir(tmp_path)
    df_desc = generate_descriptors(cleaned_csv)

    expected_cols = {
        'Molecular Weight', 'LogP', 'Labute ASA', 'TPSA',
        'Num H Donors', 'Num H Acceptors', 'Num Rotatable Bonds', 'Num Rings'
    }

    # Columns added correctly
    assert expected_cols.issubset(set(df_desc.columns))
    # All descriptor columns have numeric values
    assert all(pd.api.types.is_numeric_dtype(df_desc[col]) for col in expected_cols)
    # Output file created
    assert os.path.exists("data_preprocessed.csv")


def test_generate_descriptors_handles_invalid_smiles(tmp_path):
    """Ensure invalid SMILES are handled gracefully and descriptors still computed for valid entries."""
    df = pd.DataFrame({'Canonical SMILES': ['C', 'invalid']})
    csv_path = tmp_path / "invalid_test.csv"
    df.to_csv(csv_path, index=False)

    os.chdir(tmp_path)
    df_result = generate_descriptors(csv_path)

    # Only one valid molecule should have numeric descriptor values
    valid_mask = df_result['Canonical SMILES'] == 'C'
    assert df_result.loc[valid_mask, 'Molecular Weight'].notna().all()
    assert df_result.loc[~valid_mask, 'Molecular Weight'].isna().all()
    assert os.path.exists("data_preprocessed.csv")