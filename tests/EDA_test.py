import os
import pandas as pd
import pytest

# Import functions from your EDA module
from EDA import load_data, plot_histograms, plot_pairplots, plot_spearman_correlation


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a temporary CSV with numeric and non-numeric columns."""
    df = pd.DataFrame({
        'Canonical SMILES': ['C', 'CCO', 'CCC'],
        'Molecular Weight': [16.0, 46.1, 44.1],
        'LogP': [0.5, -0.2, 1.0],
        'Num H Donors': [0, 1, 0]
    })
    csv_file = tmp_path / "data_complete.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def numeric_data(sample_csv):
    """Load numeric-only data from sample CSV."""
    return load_data(sample_csv)


# ----------------------------------------------------------------------
# Tests for load_data()
# ----------------------------------------------------------------------

def test_load_data_only_numeric_columns(numeric_data):
    """Ensure load_data returns only numeric columns."""
    assert 'Canonical SMILES' not in numeric_data.columns
    assert set(numeric_data.columns) == {'Molecular Weight', 'LogP', 'Num H Donors'}


def test_load_data_returns_dataframe(numeric_data):
    """Ensure the output is a pandas DataFrame."""
    import pandas as pd
    assert isinstance(numeric_data, pd.DataFrame)


# ----------------------------------------------------------------------
# Tests for plotting functions
# ----------------------------------------------------------------------

def test_plot_histograms_creates_png(tmp_path, numeric_data):
    """Ensure plot_histograms generates a PNG file."""
    os.chdir(tmp_path)
    plot_histograms(numeric_data)
    assert os.path.exists("histograms.png")


def test_plot_pairplots_creates_png(tmp_path, numeric_data):
    """Ensure plot_pairplots generates a PNG file."""
    os.chdir(tmp_path)
    plot_pairplots(numeric_data)
    assert os.path.exists("pairplots.png")


def test_plot_spearman_correlation_creates_png(tmp_path, numeric_data):
    """Ensure plot_spearman_correlation generates a PNG file."""
    os.chdir(tmp_path)
    plot_spearman_correlation(numeric_data)
    assert os.path.exists("spearman_correlation.png")


def test_plot_functions_handle_small_dataset(tmp_path):
    """Check that plotting functions do not fail for a single-row dataset."""
    os.chdir(tmp_path)
    df_small = pd.DataFrame({
        'Molecular Weight': [16.0],
        'LogP': [0.5],
        'Num H Donors': [0]
    })
    # Should not raise errors
    plot_histograms(df_small)
    plot_pairplots(df_small)
    plot_spearman_correlation(df_small)
    # All files created
    assert os.path.exists("histograms.png")
    assert os.path.exists("pairplots.png")
    assert os.path.exists("spearman_correlation.png")