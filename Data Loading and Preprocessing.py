###Data Loading and Preprocessing
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle

def load_data(file_path):
    """Load data from an Excel file."""
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print("File not found. Please ensure the file path is correct.")
        return None  # Or handle error as needed
    
def generate_fingerprints(df, smiles_column='smiles'):
    """Generate molecular fingerprints."""
    # Check for missing SMILES strings
    missing_smiles = df[smiles_column].isnull().sum()
    if missing_smiles > 0:
        print(f"Warning: {missing_smiles} missing SMILES strings.")
        df = df.dropna(subset=[smiles_column])  # Optionally handle missing values

    # Generate fingerprints
    try:
        df['AP_FP'] = [AllChem.GetHashedAtomPairFingerprintAsBitVect(Chem.MolFromSmiles(x)) for x in df[smiles_column]]
        print("Fingerprints generated successfully.")
        return df
    except Exception as e:
        print(f"Error generating fingerprints: {e}")
        return None  # Or handle error as needed
    
def save_processed_data(fingerprints, labels, fp_path='ap_fingerprints.pkl', label_path='labels.pkl'):
    """Save fingerprints and labels."""
    try:
        with open(fp_path, 'wb') as f:
            pickle.dump(fingerprints, f)
        with open(label_path, 'wb') as f:
            pickle.dump(labels, f)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

# Example usage:
df = load_data('LCK_ac.xlsx')

if df is not None:  # Only proceed if data is loaded successfully
    df = generate_fingerprints(df)
    
    if df is not None:  # Only proceed if fingerprints are generated successfully
        fingerprints = np.array([x for x in df['AP_FP']])
        labels = df['pIC50'].values
        save_processed_data(fingerprints, labels)
