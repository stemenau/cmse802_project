# Import necessary packages
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def generate_descriptors(csv):
    """
    Generate molecular descriptors for PFAS compounds from a CSV file using RDKit.

    This function reads a CSV file containing PFAS compounds with a column 
    named 'Canonical SMILES', computes a set of molecular descriptors for 
    each compound using RDKit, and saves the results to a new CSV file 
    named 'pfas_descriptors.csv'.

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
    - The output file 'pfas_descriptors.csv' is saved in the current working directory.
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
    data.to_csv('pfas_samples_descriptors.csv', index=False)

    return data

# Generate descriptors for the samples CSV
generate_descriptors('pfas_samples.csv')