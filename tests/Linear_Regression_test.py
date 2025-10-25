import os
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import your class
from Linear_Regression import LinearRegressionModel


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV with numeric features and target variable."""
    df = pd.DataFrame({
        'Canonical SMILES': ['C', 'CCO', 'CCC', 'CCN', 'CCl'],
        'Molecular Weight': [16, 46, 44, 45, 50],
        'LogP': [0.5, -0.2, 1.0, 0.1, 0.8],
        'TPSA': [0, 20, 0, 5, 10],
        'Target': [1.2, 2.5, 2.0, 2.2, 2.8]
    })
    csv_file = tmp_path / "data_complete.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def model(tmp_path, sample_csv):
    """Initialize LinearRegressionModel."""
    os.chdir(tmp_path)  # isolate outputs
    return LinearRegressionModel(sample_csv)


# ----------------------------------------------------------------------
# Test initialization
# ----------------------------------------------------------------------

def test_model_initialization(model, sample_csv):
    """Ensure attributes are set correctly on initialization."""
    assert model.data_path == sample_csv
    assert isinstance(model.model, LinearRegression)
    assert model.X_train is None
    assert model.X_test is None
    assert model.y_train is None
    assert model.y_test is None
    assert model.y_pred is None


# ----------------------------------------------------------------------
# Test data loading and preprocessing
# ----------------------------------------------------------------------

def test_load_and_preprocess_data_sets_attributes(model):
    """load_and_preprocess_data should split data into train/test."""
    model.load_and_preprocess_data()
    assert model.X_train is not None
    assert model.X_test is not None
    assert model.y_train is not None
    assert model.y_test is not None
    # Ensure 'Canonical SMILES' is dropped
    assert 'Canonical SMILES' not in model.X_train.columns


# ----------------------------------------------------------------------
# Test training and prediction
# ----------------------------------------------------------------------

def test_train_model_sets_model(model):
    """train_model should fit the model without errors."""
    model.load_and_preprocess_data()
    model.train_model()
    assert hasattr(model.model, 'coef_')  # model is trained
    assert hasattr(model.model, 'intercept_')


def test_evaluate_model_returns_predictions(model):
    """evaluate_model should generate predictions and print metrics."""
    model.load_and_preprocess_data()
    model.train_model()
    model.evaluate_model()
    assert model.y_pred is not None
    # y_pred shape matches y_test
    assert len(model.y_pred) == len(model.y_test)


# ----------------------------------------------------------------------
# Test plotting functions
# ----------------------------------------------------------------------

def test_plot_results_creates_png(model):
    """plot_results should generate a PNG file."""
    model.load_and_preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.plot_results()
    assert os.path.exists('lr_actual_vs_predicted.png')


def test_explain_model_creates_shap_png(model):
    """explain_model should generate SHAP summary PNG file."""
    model.load_and_preprocess_data()
    model.train_model()
    model.explain_model()
    assert os.path.exists('lr_shap_summary.png')


# ----------------------------------------------------------------------
# Test end-to-end workflow
# ----------------------------------------------------------------------

def test_full_workflow(tmp_path, sample_csv):
    """Run full workflow: load, train, evaluate, plot, explain."""
    os.chdir(tmp_path)
    lr_model = LinearRegressionModel(sample_csv)
    lr_model.load_and_preprocess_data()
    lr_model.train_model()
    lr_model.evaluate_model()
    lr_model.plot_results()
    lr_model.explain_model()
    # Check files
    assert os.path.exists('lr_actual_vs_predicted.png')
    assert os.path.exists('lr_shap_summary.png')
    # Check predictions
    assert lr_model.y_pred is not None
    assert len(lr_model.y_pred) == len(lr_model.y_test)