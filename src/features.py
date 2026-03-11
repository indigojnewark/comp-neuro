"""Feature extraction from RT distributions."""
import pandas as pd
import numpy as np
from scipy import stats

def extract_rt_features(df, group_by=['subject', 'condition']):
    def compute_features(group):
        rts = group['rt'].values
        return pd.Series({
            'mean_rt': np.mean(rts), 'median_rt': np.median(rts), 'std_rt': np.std(rts),
            'skew_rt': stats.skew(rts), 'kurtosis_rt': stats.kurtosis(rts),
            'iqr_rt': np.percentile(rts, 75) - np.percentile(rts, 25),
            'cv_rt': np.std(rts) / np.mean(rts), 'n_trials': len(rts),
            'accuracy': group['correct'].mean() if 'correct' in group.columns else None
        })
    return df.groupby(group_by).apply(compute_features).reset_index()

def pivot_condition_features(features_df):
    cong = features_df[features_df['condition']=='congruent'].add_suffix('_cong')
    incong = features_df[features_df['condition']=='incongruent'].add_suffix('_incong')
    merged = pd.merge(cong, incong, left_on='subject_cong', right_on='subject_incong')
    merged['congruency_effect'] = merged['mean_rt_incong'] - merged['mean_rt_cong']
    return merged
