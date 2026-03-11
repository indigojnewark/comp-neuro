"""Data preprocessing for reaction time analysis."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

def remove_rt_outliers(df: pd.DataFrame, method: str = 'iqr', iqr_factor: float = 2.5) -> pd.DataFrame:
    """Remove RT outliers using IQR method per subject/condition."""
    df_clean = df.copy()
    
    def remove_group_outliers(group):
        Q1, Q3 = group['rt'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        return group[(group['rt'] >= lower) & (group['rt'] <= upper)]
    
    df_clean = df_clean.groupby(['subject', 'condition']).apply(remove_group_outliers)
    df_clean = df_clean.reset_index(drop=True)
    
    n_removed = len(df) - len(df_clean)
    print(f"Removed {n_removed} RT outliers ({100*n_removed/len(df):.1f}%)")
    return df_clean

def filter_correct_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only correct response trials."""
    df_correct = df[df['correct'] == 1].copy()
    print(f"Kept {len(df_correct)} correct trials ({100*len(df_correct)/len(df):.1f}%)")
    return df_correct

def exclude_low_accuracy_subjects(df: pd.DataFrame, min_accuracy: float = 0.7) -> pd.DataFrame:
    """Exclude subjects below accuracy threshold."""
    subject_acc = df.groupby('subject')['correct'].mean()
    good_subjects = subject_acc[subject_acc >= min_accuracy].index
    df_filtered = df[df['subject'].isin(good_subjects)].copy()
    n_excluded = df['subject'].nunique() - df_filtered['subject'].nunique()
    if n_excluded > 0:
        print(f"Excluded {n_excluded} subjects with accuracy < {min_accuracy}")
    return df_filtered

def preprocess_data(df: pd.DataFrame, remove_errors: bool = True,
                   remove_outliers: bool = True, min_accuracy: float = 0.7) -> Tuple[pd.DataFrame, Dict]:
    """Complete preprocessing pipeline."""
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Initial: {len(df)} trials, {df['subject'].nunique()} subjects\n")
    
    info = {'n_trials_initial': len(df), 'n_subjects_initial': df['subject'].nunique()}
    
    df_clean = exclude_low_accuracy_subjects(df, min_accuracy)
    if remove_errors:
        df_clean = filter_correct_trials(df_clean)
    if remove_outliers:
        df_clean = remove_rt_outliers(df_clean)
    
    info['n_trials_final'] = len(df_clean)
    info['n_subjects_final'] = df_clean['subject'].nunique()
    info['pct_retained'] = 100 * len(df_clean) / len(df)
    
    print(f"\nFinal: {info['n_trials_final']} trials, {info['n_subjects_final']} subjects")
    print(f"Retained {info['pct_retained']:.1f}% of data")
    print("="*60)
    return df_clean, info
