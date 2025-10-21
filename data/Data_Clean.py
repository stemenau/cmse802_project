# Import necessary packages
import pandas as pd 
from rdkit import Chem

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
        return None  # return None if invalid or not a string

def clean_data(csv):
    """
    Clean and preprocess a CSV file containing chemical data.

    This function reads a CSV file into a Pandas DataFrame, removes 
    unnecessary columns, drops rows with missing SMILES strings, 
    canonicalizes SMILES using `get_canonical_smiles()`, removes 
    duplicate entries, and exports the cleaned dataset to a new CSV file 
    named `'cleaned_pfas_list.csv'`.

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
    - The cleaned data is saved as `'cleaned_pfas_list.csv'` in the working 
      directory.

    See Also
    --------
    get_canonical_smiles : Converts SMILES strings to their canonical forms.
    """
    # Read CSV file into a Pandas DataFrame
    data_init = pd.read_csv(csv)

    # Remove unnecessary columns
    data_trunc= data_init.drop(columns=['DTXSID','INCHIKEY','IUPAC NAME','INCHI STRING','MONOISOTOPIC MASS','QC Level','# ToxCast Active','Total Assays','% ToxCast Active'])
        
    # Remove rows with missing SMILES strings
    data_trunc = data_trunc.dropna(subset=['SMILES']).reset_index(drop=True)

    # Canonicalize SMILES strings and add as a new column
    data_trunc['Canonical SMILES'] = data_trunc['SMILES'].apply(get_canonical_smiles)

    # Remove duplicate SMILES entries
    data = data_trunc.drop_duplicates(subset=['Canonical SMILES'], keep='first').reset_index(drop=True)

    # Save cleaned DataFrame to a new CSV file
    data.to_csv('cleaned_pfas_list.csv', index=False)

    return data

# Use functions to clean data
clean_data('Chemical List pfasmaster-2025-09-13.csv')