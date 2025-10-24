"""
Module for selecting a sample subset of a dataset.

This module contains functions for selecting a random sample of entries from a dataset.
The module outputs sample data to CSV and SDF formats for further analysis.

Author: Audrey Stemen (stemenau@msu.edu)
Date: October 2025
"""
# Import necessary packages
import pandas as pd
from rdkit import Chem

def get_samples(csv, n_samples=1000, random_state=42):
    """
    Randomly sample a subset of entries from a cleaned dataset.

    This function reads a cleaned CSV file of compounds, randomly 
    selects a specified number of entries, and saves the subset to a new 
    CSV file named 'data_samples.csv'.

    Parameters
    ----------
    csv : str
        Path to the input CSV file containing the cleaned dataset.
    n_samples : int, default=1000
        Number of samples to randomly select from the dataset.
    random_state : int, default=42
        Random seed for reproducibility of the sampling process.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the randomly sampled entries from the dataset.

    Notes
    -----
    - The output file 'data_samples.csv' is saved in the current working directory.
    - The sampling process is reproducible due to the fixed random seed.
    """
    # Read the cleaned PFAS dataset
    data = pd.read_csv(csv)

    # Randomly sample n_samples entries
    sampled_data = data.sample(n=n_samples, random_state=random_state)

    # Export the sampled entries to a new CSV file
    sampled_data.to_csv('../data/data_samples.csv', index=False)

    return sampled_data

def csv_to_sdf(csv):
    """
    Convert a CSV file containing SMILES strings into an SDF file.

    This function reads a CSV file ('data_samples.csv') 
    containing a column named 'Canonical SMILES', converts each valid 
    SMILES string to an RDKit molecule, and writes all molecules to 
    an SDF file.

    Parameters
    ----------
    csv : str
        Path to the input CSV file containing canonical SMILES strings.

    Returns
    -------
    int
        The number of molecules successfully written to the SDF file.

    Notes
    -----
    - Only valid SMILES strings are converted and written to the SDF file.
    - Additional columns in the CSV (if present) are included as molecule
      properties in the SDF.
    - The output SDF file is written in the working directory unless an
      absolute path is provided.
    """
    # Read CSV file into a DataFrame
    data = pd.read_csv(csv)

    # Ensure the expected SMILES column exists
    if 'Canonical SMILES' not in data.columns:
        raise ValueError("Input CSV must contain a 'Canonical SMILES' column.")

    # Create SDF writer to edit SDF file within working directory
    writer = Chem.SDWriter('data_samples.sdf')
    n_written = 0

    # Convert SMILES to molecules and write to SDF
    for _, row in data.iterrows(): # Index does not matter here
        smiles = row['Canonical SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Add properties to the molecule
            for col in data.columns:
                if col != 'Canonical SMILES':
                    mol.SetProp(col, str(row[col]))
            writer.write(mol)
            n_written += 1

    writer.close()

    print(f"{n_written} molecules successfully written to 'data_samples.sdf'.")
    return n_written

def main():
    # Get a random sample of the cleaned data
    sampled_data = get_samples('../data/data_cleaned.csv')

    # Convert sampled CSV to SDF
    csv_to_sdf('../data/data_samples.csv')

if __name__ == "__main__":
    main()