"""
This module implements a linear regression model to predict a target variable from a dataset.

It includes data loading, preprocessing, model training, evaluation, and explanation using SHAP.
Plots are generated to visualize actual vs predicted values and feature importance by SHAP values.
"""

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import shap

class LinearRegressionModel:
    def __init__(self, data_path):
        """
        Initialize a Linear Regression model instance and prepare data attributes.

        This constructor sets up the initial configuration for a regression model
        by storing the path to the input dataset, initializing a scikit-learn
        `LinearRegression` model, and creating placeholders for training and test
        datasets as well as prediction outputs.

        Parameters
        ----------
        data_path : str
            Path to the CSV file or dataset containing the input data
            for model training and evaluation.

        Attributes
        ----------
        data_path : str
            The path to the dataset used for training and testing.
        model : sklearn.linear_model.LinearRegression
            The initialized linear regression model.
        X_train : pandas.DataFrame or numpy.ndarray, optional
            The feature matrix for training (initialized as None).
        X_test : pandas.DataFrame or numpy.ndarray, optional
            The feature matrix for testing (initialized as None).
        y_train : pandas.Series or numpy.ndarray, optional
            The target variable for training (initialized as None).
        y_test : pandas.Series or numpy.ndarray, optional
            The target variable for testing (initialized as None).
        y_pred : numpy.ndarray, optional
            The predicted target values from the model (initialized as None).

        Notes
        -----
        - Data loading and preprocessing are expected to occur in a separate method.
        - The model is not fitted during initialization.
        """
        self.data_path = data_path
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_and_preprocess_data(self):
        """
        Load the dataset from the specified file path and prepare it for modeling.

        This method reads the dataset from the path provided during initialization,
        removes the 'Canonical SMILES' column (which is non-numeric and not used
        as a feature), separates the features and target variable, and splits the
        data into training and testing sets for model development.

        The resulting datasets are stored as instance attributes for later use
        in training, evaluation, and prediction.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Attributes Updated
        ------------------
        X_train : pandas.DataFrame
            Feature matrix used for model training.
        X_test : pandas.DataFrame
            Feature matrix used for model testing.
        y_train : pandas.Series
            Target values corresponding to the training set.
        y_test : pandas.Series
            Target values corresponding to the test set.

        Notes
        -----
        - The target variable is assumed to be the **last column** in the dataset.
        - The data is split using an 80/20 train-test ratio with a fixed
        random seed (`random_state=42`) for reproducibility.
        - The 'Canonical SMILES' column is dropped to avoid including
        string-based molecular identifiers in numerical modeling.
        """
        # Load data
        data = pd.read_csv(self.data_path)

        # Omit 'Canonical SMILES' column
        data = data.drop(columns=['Canonical SMILES'])
        
        # Assign features and target variable
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        """
        Train the linear regression model on the prepared training data.

        This method fits the initialized `LinearRegression` model using
        the training feature matrix (`X_train`) and corresponding target
        values (`y_train`). The trained model is stored as an instance
        attribute and can be used for prediction and further analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The training and testing data must be loaded and split prior
        to calling this method (see `load_and_preprocess_data`).
        - The fitted model is stored in `self.model` and can be accessed
        for evaluation, prediction, or SHAP explanation.
        """
        # Train the linear regression model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained model on the test dataset and print performance metrics.

        This method generates predictions for the test feature matrix (`X_test`)
        and computes two standard regression evaluation metrics:
        Mean Squared Error (MSE) and the coefficient of determination (R² score).
        The results are printed to the console and stored internally for later use.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Attributes Updated
        ------------------
        y_pred : numpy.ndarray
            The predicted target values for the test set.

        Notes
        -----
        - The model must be trained before evaluation.
        - Printed metrics provide a quick overview of model accuracy and fit quality.
        """
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        print(f'Mean Squared Error: {mse}')
        print(f'R² Score: {r2}')

    def plot_results(self):
        """
        Generate and save a scatter plot comparing actual vs. predicted values.

        This method visualizes model performance by plotting the predicted
        values against the actual target values for the test dataset.
        A reference diagonal line is included to indicate perfect prediction.
        The plot also displays MSE and R² metrics for quick interpretation
        and is saved as `'lr_actual_vs_predicted.png'` in the working directory.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method assumes that `evaluate_model()` has been called to
        generate predictions (`self.y_pred`).
        - Model evaluation metrics are annotated directly on the plot in relative 
        axes coordinates.
        """
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.8)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        # Print MSE and R^2 on the plot
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        plt.text(0.75, 0.25, f'MSE: {mse:.2f}\nR²: {r2:.2f}', fontsize=12, 
                 verticalalignment='top', transform=plt.gca().transAxes)
        plt.savefig('lr_actual_vs_predicted.png')
        plt.close() # Ensure that formatting does not carry over for SHAP plot

    def explain_model(self):
        """
        Generate SHAP (SHapley Additive exPlanations) values to interpret model predictions.

        This method applies SHAP to compute feature importance values for
        the trained linear regression model using the test dataset.
        It then produces a SHAP summary plot visualizing the contribution
        of each feature to the model’s predictions and saves the figure
        as `'lr_shap_summary.png'`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - SHAP provides a consistent, model-agnostic framework for interpreting
        feature contributions in predictive models.
        - The explainer is initialized with the trained model and training data,
        while SHAP values are computed for the test data.
        - The summary plot is saved automatically to the working directory.
        """
        # Use SHAP to explain the model predictions
        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(self.X_test)
        
        # Plot SHAP summary
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.savefig('lr_shap_summary.png')

def main():
    # Initialize model with data path
    data_path = 'data_complete.csv'
    lr_model = LinearRegressionModel(data_path)
    
    # Load and preprocess data
    lr_model.load_and_preprocess_data()
    
    # Train the model
    lr_model.train_model()
    
    # Evaluate the model
    lr_model.evaluate_model()
    
    # Plot results
    lr_model.plot_results()
    
    # Explain model predictions
    lr_model.explain_model()

if __name__ == '__main__':
    main()