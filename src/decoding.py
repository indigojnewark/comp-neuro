"""ML classifiers for behavioural decoding."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def prepare_features(features_df, target='condition'):
    exclude_cols = ['subject', 'condition', 'trial']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols and features_df[c].dtype in [np.float64, np.int64]]
    X, y = features_df[feature_cols].values, features_df[target].values
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}, Classes: {np.unique(y)}")
    return X, y, feature_cols

def train_classifiers(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    print("\n" + "="*60)
    print("CLASSIFIER TRAINING & EVALUATION")
    print("="*60)
    
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        
        results[name] = {'model': clf, 'accuracy': acc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
        print(f"\n{name}:")
        print(f"  Test Accuracy: {acc:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    print("="*60)
    return results, scaler, X_test_scaled, y_test
