# Get 100 sample entries from the cleaned PFAS dataset 'cleaned_pfas_list.csv'

# Import necessary packages
import pandas as pd

def get_samples(csv, n_samples=100, random_state=42):
    """
    Randomly sample a subset of entries from a cleaned PFAS dataset.

    This function reads a cleaned CSV file of PFAS compounds, randomly 
    selects a specified number of entries, and saves the subset to a new 
    CSV file named `'pfas_samples.csv'`.

    Parameters
    ----------
    csv : str
        Path to the input CSV file containing the cleaned PFAS dataset.
    n_samples : int, default=100
        Number of samples to randomly select from the dataset.
    random_state : int, default=42
        Random seed for reproducibility of the sampling process.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the randomly sampled entries from the dataset.

    Notes
    -----
    - The output file `'pfas_samples.csv'` is saved in the current working directory.
    - The sampling process is reproducible due to the fixed random seed.
    """
    # Read the cleaned PFAS dataset
    df = pd.read_csv(csv)

    # Randomly sample n_samples entries
    sampled_df = df.sample(n=n_samples, random_state=random_state)

    # Export the sampled entries to a new CSV file
    sampled_df.to_csv('pfas_samples.csv', index=False)

    return sampled_df

# Use function to get samples
get_samples('cleaned_pfas_list.csv')