import os
import pandas as pd
import pytest
from rdkit import Chem
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import functions from your sampling module
from Get_Samples import get_samples, csv_to_sdf


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a temporary CSV file with canonical SMILES for testing."""
    df = pd.DataFrame({
        'Canonical SMILES': ['C', 'CCO', 'CCC', 'CCN', 'CCCl'],
        'Name': ['methane', 'ethanol', 'propane', 'ethylamine', 'chloroethane']
    })
    csv_file = tmp_path / "data_cleaned.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


# ----------------------------------------------------------------------
# Tests for get_samples()
# ----------------------------------------------------------------------

def test_get_samples_returns_dataframe(tmp_path, sample_csv):
    """Check that get_samples returns a DataFrame with correct number of rows."""
    os.chdir(tmp_path / "tmp_dir")  # isolate output
    os.makedirs("../data", exist_ok=True)  # ensure output directory exists

    df_sampled = get_samples(sample_csv, n_samples=3, random_state=123)

    assert isinstance(df_sampled, pd.DataFrame)
    assert len(df_sampled) == 3
    assert 'Canonical SMILES' in df_sampled.columns
    # Check reproducibility
    df_sampled2 = get_samples(sample_csv, n_samples=3, random_state=123)
    pd.testing.assert_frame_equal(df_sampled, df_sampled2)


def test_get_samples_output_file_created(tmp_path, sample_csv):
    """Check that CSV file is written to expected location."""
    os.chdir(tmp_path / "tmp_dir")
    os.makedirs("../data", exist_ok=True)
    get_samples(sample_csv, n_samples=2)
    assert os.path.exists("../data/data_samples.csv")


def test_get_samples_more_than_available(tmp_path, sample_csv):
    """Sampling more rows than available should raise an error."""
    os.chdir(tmp_path / "tmp_dir")
    os.makedirs("../data", exist_ok=True)
    with pytest.raises(ValueError):
        get_samples(sample_csv, n_samples=100)  # only 5 rows exist


# ----------------------------------------------------------------------
# Tests for csv_to_sdf()
# ----------------------------------------------------------------------

@pytest.fixture
def sample_csv_with_properties(tmp_path):
    """Temporary CSV with SMILES and extra columns for SDF testing."""
    df = pd.DataFrame({
        'Canonical SMILES': ['C', 'CCO', 'invalid'],
        'Property': ['A', 'B', 'C']
    })
    csv_file = tmp_path / "data_samples.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_csv_to_sdf_writes_correct_number_of_molecules(tmp_path, sample_csv_with_properties):
    """Valid SMILES should be converted to SDF; invalid ignored."""
    os.chdir(tmp_path)
    n_written = csv_to_sdf(sample_csv_with_properties)
    assert n_written == 2  # 'C' and 'CCO' only
    assert os.path.exists("data_samples.sdf")

    # Check that molecules have properties set
    suppl = Chem.SDMolSupplier("data_samples.sdf")
    for mol in suppl:
        assert mol is not None
        assert mol.HasProp("Property")


def test_csv_to_sdf_missing_smiles_column(tmp_path):
    """Should raise ValueError if 'Canonical SMILES' column missing."""
    df = pd.DataFrame({'wrong': ['C', 'CCO']})
    csv_file = tmp_path / "bad.csv"
    df.to_csv(csv_file, index=False)
    os.chdir(tmp_path)

    with pytest.raises(ValueError):
        csv_to_sdf(csv_file)


def test_csv_to_sdf_with_empty_csv(tmp_path):
    """Empty CSV should write zero molecules."""
    df = pd.DataFrame({'Canonical SMILES': []})
    csv_file = tmp_path / "empty.csv"
    df.to_csv(csv_file, index=False)
    os.chdir(tmp_path)

    n_written = csv_to_sdf(csv_file)
    assert n_written == 0
    assert os.path.exists("data_samples.sdf")