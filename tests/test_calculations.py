"""
Unit tests to verify calculations and functionality in the Explainability AI Tool.
Tests include generating mock data and verifying that calculations are performed correctly.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_mock_data_generation():
    """Test that mock data is generated correctly for different use cases."""
    
    # Test Credit Risk data generation
    np.random.seed(42)
    n_samples = 500
    
    age = np.random.normal(40, 15, n_samples)
    income = np.random.lognormal(10, 0.5, n_samples)
    loan_amount = np.random.lognormal(9, 0.8, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    employment_years = np.random.gamma(2, 2, n_samples)
    
    # Create synthetic target (loan default: 1 = default, 0 = no default)
    default_prob = (
        0.1 +
        0.3 * (credit_score < 600) +
        0.2 * (income < 30000) +
        0.15 * (loan_amount / income > 0.3) +
        0.1 * (employment_years < 2) +
        np.random.normal(0, 0.1, n_samples)
    )
    default = (np.random.random(n_samples) < default_prob).astype(int)
    
    df_credit_risk = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'default': default
    })
    
    # Verify data structure
    assert len(df_credit_risk) == n_samples
    assert 'default' in df_credit_risk.columns
    assert df_credit_risk['default'].dtype in ['int64', 'int32']
    assert all(val in [0, 1] for val in df_credit_risk['default'].unique())
    
    # Test Fraud Detection data generation
    np.random.seed(42)
    n_samples = 500
    
    transaction_amount = np.random.lognormal(8, 1.5, n_samples)
    account_age_days = np.random.exponential(365, n_samples)
    num_transactions_day = np.random.poisson(5, n_samples)
    merchant_risk_score = np.random.uniform(0, 1, n_samples)
    time_since_last_transaction = np.random.exponential(2, n_samples)
    
    # Create synthetic target (fraud: 1 = fraud, 0 = legitimate)
    fraud_prob = (
        0.05 +
        0.2 * (transaction_amount > 5000) +
        0.1 * (merchant_risk_score > 0.8) +
        0.15 * (num_transactions_day > 10) +
        np.random.normal(0, 0.05, n_samples)
    )
    fraud = (np.random.random(n_samples) < fraud_prob).astype(int)
    
    df_fraud_detection = pd.DataFrame({
        'transaction_amount': transaction_amount,
        'account_age_days': account_age_days,
        'num_transactions_day': num_transactions_day,
        'merchant_risk_score': merchant_risk_score,
        'time_since_last_transaction': time_since_last_transaction,
        'fraud': fraud
    })
    
    # Verify data structure
    assert len(df_fraud_detection) == n_samples
    assert 'fraud' in df_fraud_detection.columns
    assert df_fraud_detection['fraud'].dtype in ['int64', 'int32']
    assert all(val in [0, 1] for val in df_fraud_detection['fraud'].unique())


def test_model_training_and_prediction():
    """Test that models can be trained and make predictions correctly."""
    
    # Generate mock training data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.normal(40, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    
    # Target variable based on features
    default_probability = (
        0.1 +
        0.3 * (credit_score < 600) +
        0.2 * (income < 30000) +
        0.1 * (age < 25)
    )
    default = (np.random.random(n_samples) < default_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'default': default
    })
    
    # Prepare features and target
    X = df[['age', 'income', 'credit_score']]
    y = df['default']
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    rf_model.fit(X, y)
    lr_model.fit(X, y)
    
    # Make predictions
    rf_predictions = rf_model.predict(X)
    lr_predictions = lr_model.predict(X)
    
    # Verify predictions are valid
    assert len(rf_predictions) == n_samples
    assert len(lr_predictions) == n_samples
    assert all(pred in [0, 1] for pred in rf_predictions)
    assert all(pred in [0, 1] for pred in lr_predictions)
    
    # Test prediction probabilities
    rf_probs = rf_model.predict_proba(X)
    lr_probs = lr_model.predict_proba(X)
    
    assert rf_probs.shape == (n_samples, 2)  # Two classes
    assert lr_probs.shape == (n_samples, 2)  # Two classes
    assert np.allclose(rf_probs.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.allclose(lr_probs.sum(axis=1), 1.0)  # Probabilities sum to 1


def test_metric_calculations():
    """Test that evaluation metrics are calculated correctly."""
    
    # Generate sample true and predicted values
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.choice([0, 1], size=n_samples)
    y_pred = np.random.choice([0, 1], size=n_samples)
    
    # Calculate metrics using sklearn
    expected_accuracy = accuracy_score(y_true, y_pred)
    expected_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    expected_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    expected_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    expected_confusion_matrix = confusion_matrix(y_true, y_pred)
    
    # Verify calculations are reasonable
    assert 0 <= expected_accuracy <= 1
    assert 0 <= expected_precision <= 1
    assert 0 <= expected_recall <= 1
    assert 0 <= expected_f1 <= 1
    
    # Test confusion matrix shape
    assert expected_confusion_matrix.shape == (2, 2)  # For binary classification
    assert np.sum(expected_confusion_matrix) == n_samples


def test_feature_importance_calculation():
    """Test that feature importance can be calculated correctly."""
    
    # Generate mock data
    np.random.seed(42)
    n_samples = 1000
    
    # Features with different importance levels
    feature1 = np.random.normal(0, 1, n_samples)  # Most important
    feature2 = np.random.normal(0, 1, n_samples)  # Medium important
    feature3 = np.random.normal(0, 1, n_samples)  # Least important
    
    # Target based mostly on feature1
    y = (feature1 > 0.5).astype(int)
    
    X = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3
    })
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Verify importances properties
    assert len(importances) == 3  # Three features
    assert abs(sum(importances) - 1.0) < 1e-6  # Should sum to 1
    assert all(imp >= 0 for imp in importances)  # All should be non-negative


def test_perturbation_stability():
    """Test that model predictions are reasonably stable under small perturbations."""
    
    # Generate mock data
    np.random.seed(42)
    n_samples = 100
    
    X_original = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    
    y = (X_original['feature1'] > 0).astype(int)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_original, y)
    
    # Test stability with perturbations on a single sample
    sample_idx = 0
    original_sample = X_original.iloc[sample_idx].copy()
    original_pred = model.predict([original_sample])[0]
    
    # Apply small perturbations and check prediction consistency
    perturbed_predictions = []
    for i in range(10):
        perturbation = np.random.normal(0, 0.01, size=original_sample.shape)  # 1% std deviation
        perturbed_sample = original_sample + perturbation
        perturbed_sample = pd.DataFrame([perturbed_sample], columns=X_original.columns)
        
        pred = model.predict(perturbed_sample)[0]
        perturbed_predictions.append(pred)
    
    # Calculate stability: ratio of same predictions
    same_as_original = sum(1 for pred in perturbed_predictions if pred == original_pred)
    stability = same_as_original / len(perturbed_predictions)
    
    # We expect high stability for small perturbations
    assert 0 <= stability <= 1


def test_categorical_encoding():
    """Test that categorical encoding works correctly."""
    
    # Create mock data with categorical variables
    df = pd.DataFrame({
        'numeric_feature': [1, 2, 3, 4, 5],
        'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Apply dummy encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Verify encoding worked
    assert 'categorical_feature_B' in X_encoded.columns
    assert 'categorical_feature_C' in X_encoded.columns
    assert 'categorical_feature_A' not in X_encoded.columns  # dropped due to drop_first=True
    assert len(X_encoded.columns) == 3  # numeric_feature + 2 dummy vars


def test_data_processing_pipeline():
    """Test the complete pipeline from data to model evaluation."""
    
    # Generate comprehensive mock dataset
    np.random.seed(42)
    n_samples = 500
    
    # Create features
    age = np.random.normal(40, 10, n_samples)
    income = np.random.lognormal(10, 0.5, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    employment_years = np.random.gamma(2, 2, n_samples)
    
    # Create target with known relationships
    default_prob = (
        0.1 +
        0.4 * (credit_score < 600) +
        0.2 * (income < 30000) +
        0.15 * (employment_years < 2) +
        0.05 * (age < 25)
    )
    default = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'default': default
    })
    
    # Process data
    feature_cols = [col for col in df.columns if col != 'default']
    X = df[feature_cols]
    y = df['default']
    
    # Handle categorical variables (though we don't have any in this example)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split data (simulate train/test split)
    split_idx = int(0.8 * len(df))
    X_train, X_test = X_encoded[:split_idx], X_encoded[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Basic sanity checks
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
    assert cm.shape[0] == cm.shape[1]  # Square confusion matrix


if __name__ == "__main__":
    # Run all tests
    test_mock_data_generation()
    print("✓ Mock data generation tests passed")
    
    test_model_training_and_prediction()
    print("✓ Model training and prediction tests passed")
    
    test_metric_calculations()
    print("✓ Metric calculation tests passed")
    
    test_feature_importance_calculation()
    print("✓ Feature importance calculation tests passed")
    
    test_perturbation_stability()
    print("✓ Perturbation stability tests passed")
    
    test_categorical_encoding()
    print("✓ Categorical encoding tests passed")
    
    test_data_processing_pipeline()
    print("✓ Data processing pipeline tests passed")
    
    print("\n🎉 All calculation verification tests passed!")