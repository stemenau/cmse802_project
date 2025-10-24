# Import necessary packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
def load_data(csv):
    data = pd.read_csv(csv)
    return data

# Histograms of feature values - visualize distributions of each feature
def plot_histograms(data):
    data.hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.savefig('../results/samples_histograms.png')

# Pairplots - visually identify any relationships between features
def plot_pairplots(data):
    sns.pairplot(data)
    plt.savefig('../results/samples_pairplots.png')

# Spearman correlation matrix - identify monotonic relationships between features
def plot_spearman_correlation(data):
    corr = data.corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Spearman Correlation Matrix')
    plt.savefig('../results/samples_spearman_correlation.png')

# Main function to run EDA
def main(): 
    data = load_data('pfas_samples_final.csv')
    plot_histograms(data)
    plot_pairplots(data)
    plot_spearman_correlation(data)