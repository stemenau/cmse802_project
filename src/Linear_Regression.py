# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import shap

class LinearRegressionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_and_preprocess_data(self):
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
        # Train the linear regression model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')

    def plot_results(self):
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.7)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.savefig('lr_actual_vs_predicted.png')

    def explain_model(self):
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