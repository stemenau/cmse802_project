# Data: PFAS Master List of PFAS Substances (RETIRED) https://comptox.epa.gov/dashboard/chemical-lists/pfasmaster 
# Dataset downloaded as 'Chemical List pfasmaster-2025-09-13.csv'
"""
Module for data preprocessing and feature engineering.

This module contains functions for cleaning, transforming, and preparing data for modeling. 
It handles missing and duplicate values in the dataset and generates relevant molecular descriptors using RDKit.
The module outputs cleaned data with new features to CSV and SDF formats for further analysis.

Author: Audrey Stemen (stemenau@msu.edu)
Date: October 2025
"""
# Import necessary packages
import pandas as pd 
from rdkit import Chem
from rdkit.Chem import Descriptors

# Canonicalize SMILES strings and add as a new column
def get_canonical_smiles(smiles):
    """
    Convert a SMILES string to its canonical form using RDKit.

    This function takes a SMILES string, validates it, and returns its 
    canonical representation as defined by RDKit. If the input is not 
    a string or cannot be parsed into a valid molecule, the function 
    returns None.

    Parameters
    ----------
    smiles : str
        The SMILES string to canonicalize.

    Returns
    -------
    str or None
        The canonical SMILES string if valid; otherwise, None.

    Notes
    -----
    - Canonicalization ensures that equivalent molecules are represented
      by the same SMILES string.
    - Non-string inputs and invalid SMILES are safely handled by returning None.
    """
    if isinstance(smiles, str):  # Only process strings, line suggested by ChatGPT, GPT-5
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None  # Return None if invalid or not a string

def clean_data(csv):
    """
    Clean and preprocess a CSV file containing chemical data.

    This function reads a CSV file into a Pandas DataFrame, drops rows with
    missing SMILES strings, canonicalizes SMILES using `get_canonical_smiles()`, 
    removes extraneous columns, removes duplicate entries, and exports the cleaned
    dataset to a new CSV file named 'data_cleaned.csv'.

    Parameters
    ----------
    csv : str
        Path to the input CSV file containing PFAS compound data from the EPA PFAS Master List of PFAS Substances.

    Returns
    -------
    pandas.DataFrame
        A cleaned DataFrame containing only unique, valid, and canonicalized
        SMILES strings, along with relevant remaining columns.

    Notes
    -----
    - Columns removed include identifiers, names, and quality control metrics
      that are not required for downstream analysis.
    - The canonical SMILES column ensures consistency across equivalent 
      molecular structures.
    - The cleaned data is saved as 'data_cleaned.csv' in the working
      directory.

    See Also
    --------
    get_canonical_smiles : Converts SMILES strings to their canonical forms.
    """
    # Read CSV file into a Pandas DataFrame
    data_init = pd.read_csv(csv)

    # Remove rows with missing SMILES strings
    data = data_init.dropna(subset=['SMILES']).reset_index(drop=True)

    # Canonicalize SMILES strings and add as a new column
    data['Canonical SMILES'] = data['SMILES'].apply(get_canonical_smiles)

    # Remove all columns except 'Canonical SMILES'
    data = data[['Canonical SMILES']]

    # Remove duplicate SMILES entries
    data = data.drop_duplicates(subset=['Canonical SMILES'], keep='first').reset_index(drop=True)

    # Remove rows where canonical SMILES could not be generated
    data = data.dropna(subset=['Canonical SMILES']).reset_index(drop=True)

    # Save cleaned DataFrame to a new CSV file
    data.to_csv('data_cleaned.csv', index=False)

    return data

def csv_to_sdf(csv):
    """
    Convert a CSV file containing SMILES strings into an SDF file.

    This function reads a CSV file ('data_cleaned.csv') 
    containing a column named 'Canonical SMILES', converts each valid 
    SMILES string to an RDKit molecule, and writes all molecules to an SDF file 
    which can then be used for further analysis such as molecular docking.

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
    df = pd.read_csv(csv)

    # Ensure the expected SMILES column exists
    if 'Canonical SMILES' not in df.columns:
        raise ValueError("Input CSV must contain a 'Canonical SMILES' column.")

    # Create SDF writer to edit SDF file within working directory
    writer = Chem.SDWriter('data_cleaned.sdf')
    n_written = 0

    # Convert SMILES to molecules and write to SDF
    for _, row in df.iterrows(): # Index does not matter here
        smiles = row['Canonical SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Add properties to the molecule
            for col in df.columns:
                if col != 'Canonical SMILES':
                    mol.SetProp(col, str(row[col]))
            writer.write(mol)
            n_written += 1

    writer.close()

    print(f"{n_written} molecules successfully written to 'cleaned_pfas_list.sdf'.")
    return n_written

def generate_descriptors(csv):
    """
    Generate molecular descriptors for PFAS compounds from a CSV file using RDKit.

    This function reads a CSV file containing PFAS compounds with a column 
    named 'Canonical SMILES', computes a set of molecular descriptors for 
    each compound using RDKit, and saves the results to a new CSV file 
    named 'data_preprocessed.csv'.

    Parameters
    ----------
    csv : str
        Path to the input CSV file containing canonical SMILES strings.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original data along with computed 
        molecular descriptors.
    Notes
    -----
    - The output file 'data_preprocessed.csv' is saved in the current working directory.
    - Ensure that the RDKit library is installed in your Python environment.
    """
    # Read the input CSV file
    data = pd.read_csv(csv)

    # Generate descriptors for all molecules
    for i, smi in enumerate(data['Canonical SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            data.loc[i, 'Molecular Weight'] = Descriptors.MolWt(mol)
            data.loc[i, 'LogP'] = Descriptors.MolLogP(mol)
            data.loc[i, 'Labute ASA'] = Descriptors.LabuteASA(mol)
            data.loc[i, 'TPSA'] = Descriptors.TPSA(mol)
            data.loc[i, 'Num H Donors'] = Descriptors.NumHDonors(mol)
            data.loc[i, 'Num H Acceptors'] = Descriptors.NumHAcceptors(mol)
            data.loc[i, 'Num Rotatable Bonds'] = Descriptors.NumRotatableBonds(mol)
            data.loc[i, 'Num Rings'] = Descriptors.RingCount(mol)
        else:
            print(f"Invalid SMILES string at index {i}: {smi}")

    # Save the updated dataframe to a new CSV file
    data.to_csv('data_preprocessed.csv', index=False)

    return data

def main():
    # Clean the initial data
    cleaned_data = clean_data('Chemical List pfasmaster-2025-09-13.csv')

    # Convert cleaned CSV to SDF
    csv_to_sdf('data_cleaned.csv')

    # Generate descriptors for the cleaned data
    generate_descriptors('data_cleaned.csv')