"""
Module for exploratory data analysis (EDA) of a dataset.

This module contains functions to perform various EDA tasks such as
plotting histograms, pairplots, and Spearman correlation matrices.
The module outputs a PNG for each type of analysis.

Author: Audrey Stemen (stemenau@msu.edu)
Date: October 2025
"""
# Import necessary packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
def load_data(csv):
    data = pd.read_csv(csv)
    # Exclude 'Canonical SMILES' column for numerical analysis
    data = data.select_dtypes(include=['number'])
    return data

# Histograms of feature values - visualize distributions of each feature
def plot_histograms(data):
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    n_cols = len(data.columns)
    colors = sns.color_palette("rocket", n_colors=n_cols)
    for i, col in enumerate(data.columns):
        sns.histplot(data[col], bins=30, color=colors[i], ax=ax[i // 4, i % 4])
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=300)

# Pairplots - visually identify any relationships between features
def plot_pairplots(data):
    sns.pairplot(data)
    plt.tight_layout()
    plt.savefig('pairplots.png', dpi=300)

# Spearman correlation matrix - identify monotonic relationships between features
def plot_spearman_correlation(data):
    corr = data.corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True)
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.savefig('spearman_correlation.png', dpi=300)

def main(): 
    data = load_data('data_complete.csv')
    plot_histograms(data)
    plot_pairplots(data)
    plot_spearman_correlation(data)

if __name__ == "__main__":
    main()