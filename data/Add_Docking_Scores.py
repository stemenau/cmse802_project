# Import necessary packages
import pandas as pd
import numpy as np

def add_docking_scores(csv, docking_scores_csv):
    """
    Adds docking scores from a secondary CSV file to a main sample dataset.

    This function reads two CSV files: one containing PFAS sample data (`csv`)
    and another containing docking scores (`docking_scores_csv`). The function
    aligns the rows using the 'Index' column from the docking score file and 
    inserts the corresponding docking scores into the main dataset as a new 
    column named 'Docking Score'. The updated dataset is then saved as a new 
    file named 'pfas_samples_scores.csv'.

    Parameters
    ----------
    csv : str
        Path to the main CSV file containing PFAS sample data.
    docking_scores_csv : str
        Path to the CSV file containing docking scores and an 'Index' column 
        that maps to the rows in the main dataset.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original sample data with an added 
        'Docking Score' column.

    Notes
    -----
    The output CSV file ('pfas_samples_scores.csv') is saved in the current 
    working directory. Ensure that the 'Index' column in the docking scores 
    file corresponds correctly to the row indices in the main dataset.
    """
    # Read the input CSV files
    df = pd.read_csv(csv)
    docking_scores = pd.read_csv(docking_scores_csv)

    # Add docking scores to the main dataframe as a new column
    for i, j in zip(range(len(df)), docking_scores['Index']):
        df.loc[j,'Docking Score'] = docking_scores.loc[i,'S']

    # Save the updated dataframe to a new CSV file
    df.to_csv('pfas_samples_scores.csv', index=False)

    return df

# Add docking scores to the samples CSV
add_docking_scores('pfas_samples.csv', 'samples_docking_scores.csv')