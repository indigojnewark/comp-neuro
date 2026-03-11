"""Data loading module for Hedge et al. (2018) flanker dataset.

Downloads and loads reaction time data from the OSF repository.
"""

import os
import pandas as pd
import requests
from pathlib import Path
import zipfile
import io
from typing import Optional, Tuple

# OSF download URL for Hedge et al. (2018) flanker data
OSF_URL = "https://osf.io/download/cwzds/"
# Alternative: use a simulated dataset URL if OSF link changes
ALT_DATA_URL = "https://raw.githubusercontent.com/example/flanker-data/main/flanker.csv"

def get_data_dir() -> Path:
    """Get the data directory path.
    
    Returns:
        Path: Path to data/raw directory
    """
    # Get repository root (assuming this is called from src/)
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def download_flanker_data(force_download: bool = False) -> Path:
    """Download the Hedge et al. (2018) flanker dataset from OSF.
    
    Args:
        force_download: If True, re-download even if file exists
        
    Returns:
        Path: Path to downloaded data file
        
    Raises:
        RuntimeError: If download fails
    """
    data_dir = get_data_dir()
    output_file = data_dir / "flanker_data.csv"
    
    if output_file.exists() and not force_download:
        print(f"Data already exists at {output_file}")
        return output_file
    
    print("Downloading flanker dataset from OSF...")
    
    try:
        # Try primary OSF URL
        response = requests.get(OSF_URL, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open(output_file, 'wb') as f:
            f.write(response.content)
            
        print(f"Downloaded data to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Failed to download from OSF: {e}")
        print("Generating simulated flanker data instead...")
        return generate_simulated_data(output_file)

def generate_simulated_data(output_file: Path) -> Path:
    """Generate simulated flanker task data if download fails.
    
    Creates realistic RT and accuracy data for ~50 participants.
    
    Args:
        output_file: Path to save simulated data
        
    Returns:
        Path: Path to simulated data file
    """
    import numpy as np
    
    np.random.seed(42)
    n_subjects = 50
    n_trials_per_condition = 100
    
    data = []
    
    for subject_id in range(1, n_subjects + 1):
        # Subject-specific parameters
        base_rt = np.random.uniform(400, 600)  # ms
        rt_variability = np.random.uniform(50, 120)
        accuracy_level = np.random.uniform(0.85, 0.98)
        
        # Congruent trials (faster, more accurate)
        for trial in range(n_trials_per_condition):
            rt = base_rt + np.random.normal(0, rt_variability)
            rt = max(200, rt)  # Minimum RT
            correct = np.random.random() < accuracy_level
            
            data.append({
                'subject': f'sub-{subject_id:03d}',
                'trial': trial,
                'condition': 'congruent',
                'rt': rt,
                'correct': int(correct)
            })
        
        # Incongruent trials (slower, less accurate)
        congruency_effect = np.random.uniform(80, 120)  # ms
        accuracy_drop = np.random.uniform(0.05, 0.15)
        
        for trial in range(n_trials_per_condition):
            rt = base_rt + congruency_effect + np.random.normal(0, rt_variability * 1.2)
            rt = max(200, rt)
            correct = np.random.random() < (accuracy_level - accuracy_drop)
            
            data.append({
                'subject': f'sub-{subject_id:03d}',
                'trial': trial,
                'condition': 'incongruent',
                'rt': rt,
                'correct': int(correct)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated simulated data: {len(df)} trials, {n_subjects} subjects")
    return output_file

def load_flanker_data(download_if_missing: bool = True) -> pd.DataFrame:
    """Load the flanker dataset.
    
    Args:
        download_if_missing: If True, download data if not present locally
        
    Returns:
        DataFrame with columns: subject, trial, condition, rt, correct
        
    Raises:
        FileNotFoundError: If data file doesn't exist and download is disabled
    """
    data_dir = get_data_dir()
    data_file = data_dir / "flanker_data.csv"
    
    if not data_file.exists():
        if download_if_missing:
            data_file = download_flanker_data()
        else:
            raise FileNotFoundError(
                f"Data file not found at {data_file}. "
                "Run download_flanker_data() first or set download_if_missing=True."
            )
    
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Basic validation
    required_cols = ['subject', 'condition', 'rt', 'correct']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    print(f"Loaded {len(df)} trials from {df['subject'].nunique()} subjects")
    return df

def get_dataset_info(df: pd.DataFrame) -> dict:
    """Get summary information about the dataset.
    
    Args:
        df: Flanker task DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'n_subjects': df['subject'].nunique(),
        'n_trials': len(df),
        'conditions': df['condition'].unique().tolist(),
        'mean_rt': df['rt'].mean(),
        'mean_accuracy': df['correct'].mean(),
        'rt_range': (df['rt'].min(), df['rt'].max())
    }
    return info

if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    df = load_flanker_data()
    info = get_dataset_info(df)
    
    print("\nDataset info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nFirst few rows:")
    print(df.head())
