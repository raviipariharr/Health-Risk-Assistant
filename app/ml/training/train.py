"""
STEP 5: MODEL TRAINING SCRIPT
================================
Run this once to train and save the model:
    python -m app.ml.training.train

What this script does:
  1. Generates (or loads) the training dataset
  2. Splits into train/test sets
  3. Builds a scikit-learn Pipeline:
       StandardScaler → RandomForestClassifier
  4. Runs 5-fold cross-validation to evaluate stability
  5. Trains the final model on all training data
  6. Evaluates on the held-out test set
  7. Saves the trained model + metadata to disk (joblib)
  8. Prints a full report so you understand what was learned

WHY A PIPELINE?
  sklearn Pipeline chains preprocessing + model into one object.
  This means:
    - Fit scaler only on training data (no data leakage)
    - At prediction time, scaler auto-transforms new data
    - Save/load one object instead of two

WHY RANDOM FOREST?
  - Handles mixed features (boolean + numeric) well
  - Naturally provides feature importances
  - Robust to outliers and doesn't need extensive tuning
  - Outputs calibrated probabilities (needed for confidence scores)
  - Interpretable: you can visualize individual trees

ALTERNATIVES you could swap in:
  - GradientBoostingClassifier (often better accuracy)
  - LogisticRegression (simpler, more interpretable)
  - SVM with RBF kernel (good for small datasets)
  - XGBoost / LightGBM (state-of-art for tabular data)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV

# Import our dataset generator
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from app.ml.training.dataset import (
    generate_dataset, ALL_FEATURE_NAMES, RISK_LABELS
)

# Where to save the trained model
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'risk_classifier.joblib')
META_PATH  = os.path.join(MODEL_DIR, 'model_metadata.json')


def build_pipeline() -> Pipeline:
    """
    Build the sklearn Pipeline.

    Steps:
      1. StandardScaler
         Normalizes numeric features to mean=0, std=1.
         Without this, age (18-85) would dominate boolean features (0/1).

      2. RandomForestClassifier
         - n_estimators=200: use 200 trees (more = better but slower)
         - max_depth=12: prevent overfitting (trees can't grow infinitely)
         - min_samples_leaf=5: each leaf needs ≥5 samples (regularization)
         - class_weight='balanced': auto-correct for class imbalance
         - n_jobs=-1: use all CPU cores for training
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ))
    ])


def extract_X_y(df: pd.DataFrame):
    """
    Split the dataframe into feature matrix X and label vector y.
    
    X shape: (n_samples, n_features)  → e.g. (3000, 36)
    y shape: (n_samples,)             → e.g. (3000,)  values: 0, 1, 2
    """
    X = df[ALL_FEATURE_NAMES].values.astype(float)
    y = df['risk_label'].values
    return X, y


def train():
    print("=" * 60)
    print("AI HEALTH RISK ASSISTANT — MODEL TRAINING")
    print("=" * 60)

    # ===========================
    # 1. GENERATE DATASET
    # ===========================
    print("\n[1/6] Generating synthetic training data...")
    df = generate_dataset(n_samples=3000, random_seed=42)

    X, y = extract_X_y(df)
    print(f"Feature matrix shape: {X.shape}  (samples × features)")
    print(f"Feature names: {ALL_FEATURE_NAMES[:5]}... +{len(ALL_FEATURE_NAMES)-5} more")

    # ===========================
    # 2. TRAIN / TEST SPLIT
    # ===========================
    print("\n[2/6] Splitting into train/test sets (80/20)...")

    # stratify=y ensures both splits have the same class distribution
    # test_size=0.2 = 20% held out for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")

    # ===========================
    # 3. CROSS-VALIDATION
    # ===========================
    print("\n[3/6] Running 5-fold cross-validation on training set...")
    print("(This tells us if the model generalises or just memorises)")

    pipeline = build_pipeline()

    # StratifiedKFold preserves class ratios across folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"CV accuracy per fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"Mean CV accuracy:  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # High mean + low std = model is stable and generalises well
    if cv_scores.std() > 0.05:
        print("⚠ High variance across folds — consider more data or simpler model")
    else:
        print("✓ Stable across folds — good sign")

    # ===========================
    # 4. TRAIN FINAL MODEL
    # ===========================
    print("\n[4/6] Training final model on full training set...")
    pipeline.fit(X_train, y_train)
    print("✓ Training complete")

    # ===========================
    # 5. EVALUATE ON TEST SET
    # ===========================
    print("\n[5/6] Evaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)  # shape: (n, 3)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.3f}")

    # ROC-AUC (one-vs-rest for multi-class)
    # Measures ability to rank positive over negative examples
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    print(f"ROC-AUC (weighted OVR): {auc:.3f}")

    # Per-class metrics
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Low Risk', 'Medium Risk', 'High Risk']
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("              Pred:Low  Pred:Med  Pred:High")
    for i, row_label in enumerate(['Actual:Low ', 'Actual:Med ', 'Actual:High']):
        print(f"  {row_label}   {cm[i][0]:8d}  {cm[i][1]:8d}  {cm[i][2]:9d}")

    # Feature importances (from the Random Forest)
    rf = pipeline.named_steps['classifier']
    importances = rf.feature_importances_
    feat_imp = sorted(
        zip(ALL_FEATURE_NAMES, importances),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop 10 most important features:")
    for name, imp in feat_imp[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {name:<35} {imp:.4f}  {bar}")

    # ===========================
    # 6. SAVE MODEL
    # ===========================
    print(f"\n[6/6] Saving model to {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    # Save metadata alongside model
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": "RandomForestClassifier",
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "features": ALL_FEATURE_NAMES,
        "n_features": len(ALL_FEATURE_NAMES),
        "classes": RISK_LABELS,
        "cv_mean_accuracy": float(round(cv_scores.mean(), 4)),
        "cv_std_accuracy": float(round(cv_scores.std(), 4)),
        "test_accuracy": float(round(accuracy, 4)),
        "test_roc_auc": float(round(auc, 4)),
        "top_features": [
            {"feature": name, "importance": float(round(imp, 4))}
            for name, imp in feat_imp[:15]
        ],
    }
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model saved: {MODEL_PATH}")
    print(f"✓ Metadata saved: {META_PATH}")
    print("\n" + "=" * 60)
    print("Training complete! You can now run the API.")
    print("The model will be auto-loaded by ml/classifier.py")
    print("=" * 60)

    return pipeline, metadata


if __name__ == "__main__":
    train()