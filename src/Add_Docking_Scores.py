"""
Module for adding calculated molecular docking scores to a dataset from a separate CSV file.

This module contains a function for adding docking scores as a new column to a dataset.
The module outputs the updated dataset to a CSV file.

Author: Audrey Stemen (stemenau@msu.edu)
Date: October 2025
"""
# Import necessary packages
import pandas as pd
import numpy as np

def add_docking_scores(csv, docking_scores_csv, output_csv):
    """
    Adds docking scores from a secondary CSV file to a main dataset.

    This function reads two CSV files: one containing the main data ('csv')
    and another containing docking scores (default 'docking_scores_csv') calculated from
    molecular docking simulations. The function aligns the rows using the 'Index'
    column from the docking score file and inserts the corresponding docking scores
    into the main dataset as a new column named 'Docking Score'. Rows for which no
    docking score is found will be assigned NaN values in the 'Docking Score' column 
    and subsequently dropped. The updated dataset is then saved as a new file named 
    'data_complete.csv' unless otherwise specified.

    Parameters
    ----------
    csv : str
        Path to the main CSV file containing PFAS sample data.
    docking_scores_csv : str
        Path to the CSV file containing docking scores and an 'Index' column 
        that maps to the rows in the main dataset.
    output_csv : str
        Name of output CSV file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original sample data with an added 
        'Docking Score' column.

    Notes
    -----
    The output CSV file (default 'data_complete.csv') is saved in the current 
    working directory. Ensure that the 'Index' column in the docking scores 
    file corresponds correctly to the row indices in the main dataset.
    """
    # Read the input CSV files
    data = pd.read_csv(csv)
    docking_scores = pd.read_csv(docking_scores_csv)

    # Add docking scores to the main dataframe as a new column
    for i, j in zip(range(len(data)), docking_scores['Index']):
        data.loc[j,'Docking Score'] = docking_scores.loc[i,'S']

    # Drop rows with NaN docking scores
    data = data.dropna(subset=['Docking Score'])

    # Save the updated dataframe to a new CSV file
    data.to_csv(output_csv, index=False)

    return data

def main():
    # Add docking scores to the sample data
    updated_data = add_docking_scores('../data/data_preprocessed.csv', '../data/docking_scores.csv')

if __name__ == "__main__":
    main()