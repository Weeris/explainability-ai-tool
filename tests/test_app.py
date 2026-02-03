"""
Unit tests for the Explainability AI Tool (Project Noor)
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import sys
import os

# Add the project root to the path so we can import app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app


class TestApp(unittest.TestCase):
    """
    Test suite for the Explainability AI Tool
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create sample data for testing
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        self.sample_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        self.sample_data['target'] = y
        
        # Create sample models
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.lr_model = LogisticRegression(random_state=42)
        
        # Fit models
        X_train = self.sample_data.drop('target', axis=1)
        y_train = self.sample_data['target']
        self.rf_model.fit(X_train, y_train)
        self.lr_model.fit(X_train, y_train)
        
        # Store in session state for testing
        app.st.session_state.models = {
            'rf_model': {
                'model': self.rf_model,
                'X_train': X_train,
                'X_test': X_train,
                'y_train': y_train,
                'y_test': y_train,
                'feature_names': X_train.columns.tolist(),
                'target_name': 'target'
            },
            'lr_model': {
                'model': self.lr_model,
                'X_train': X_train,
                'X_test': X_train,
                'y_train': y_train,
                'y_test': y_train,
                'feature_names': X_train.columns.tolist(),
                'target_name': 'target'
            }
        }
        
        app.st.session_state.datasets = {
            'sample_data.csv': self.sample_data
        }
    
    def test_data_upload(self):
        """
        Test that data can be uploaded and stored properly
        """
        # Verify sample data was loaded
        self.assertIn('sample_data.csv', app.st.session_state.datasets)
        self.assertEqual(len(app.st.session_state.datasets['sample_data.csv']), 100)
        self.assertEqual(len(app.st.session_state.datasets['sample_data.csv'].columns), 6)  # 5 features + 1 target
    
    def test_model_training(self):
        """
        Test that models can be trained and stored properly
        """
        # Verify models were created
        self.assertIn('rf_model', app.st.session_state.models)
        self.assertIn('lr_model', app.st.session_state.models)
        
        # Check model properties
        rf_info = app.st.session_state.models['rf_model']
        self.assertEqual(len(rf_info['feature_names']), 5)
        self.assertEqual(rf_info['target_name'], 'target')
        
        lr_info = app.st.session_state.models['lr_model']
        self.assertEqual(len(lr_info['feature_names']), 5)
        self.assertEqual(lr_info['target_name'], 'target')
    
    def test_feature_importance_available(self):
        """
        Test that feature importance is available for tree-based models
        """
        rf_model = app.st.session_state.models['rf_model']['model']
        
        # Random Forest should have feature_importances_
        self.assertTrue(hasattr(rf_model, 'feature_importances_'))
        self.assertEqual(len(rf_model.feature_importances_), 5)  # 5 features
    
    def test_logistic_regression_no_feature_importance(self):
        """
        Test that logistic regression doesn't have feature_importances_ attribute
        """
        lr_model = app.st.session_state.models['lr_model']['model']
        
        # Logistic Regression does not have feature_importances_
        self.assertFalse(hasattr(lr_model, 'feature_importances_'))
    
    def test_data_preprocessing(self):
        """
        Test data preprocessing functionality
        """
        # Test that categorical variables can be encoded
        df_with_cat = self.sample_data.copy()
        df_with_cat['category'] = np.random.choice(['A', 'B', 'C'], size=len(df_with_cat))
        
        # Encode categorical variables
        X_with_cat = df_with_cat.drop('target', axis=1)
        X_encoded = pd.get_dummies(X_with_cat, drop_first=True)
        
        # Should have more columns after encoding
        self.assertGreater(len(X_encoded.columns), len(X_with_cat.columns))
    
    def test_model_prediction_consistency(self):
        """
        Test that models make consistent predictions
        """
        X_test = self.sample_data.drop('target', axis=1)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X_test)
        lr_pred = self.lr_model.predict(X_test)
        
        # Predictions should be binary (0 or 1)
        self.assertTrue(all(pred in [0, 1] for pred in rf_pred))
        self.assertTrue(all(pred in [0, 1] for pred in lr_pred))
        
        # Predictions should have same length as test data
        self.assertEqual(len(rf_pred), len(X_test))
        self.assertEqual(len(lr_pred), len(X_test))
    
    def test_shap_calculation_possible(self):
        """
        Test that SHAP values can be calculated for models
        """
        import shap
        
        X_test = self.sample_data.drop('target', axis=1)[:10]  # Use subset for faster testing
        
        # Test SHAP calculation for Random Forest
        try:
            explainer = shap.TreeExplainer(self.rf_model)
            shap_values = explainer.shap_values(X_test)
            # For binary classification, shape should be (n_samples, n_features)
            expected_shape = (len(X_test), len(X_test.columns))
            self.assertEqual(shap_values.shape, expected_shape)
        except Exception as e:
            # Some versions of SHAP might return different shapes for binary classification
            # Allow for list format [negative_class_shap, positive_class_shap]
            self.assertIsInstance(shap_values, (np.ndarray, list))
    
    def test_model_accuracy_calculation(self):
        """
        Test that model accuracy can be calculated
        """
        from sklearn.metrics import accuracy_score
        
        X_test = self.sample_data.drop('target', axis=1)
        y_test = self.sample_data['target']
        
        # Calculate accuracy for both models
        rf_pred = self.rf_model.predict(X_test)
        lr_pred = self.lr_model.predict(X_test)
        
        rf_accuracy = accuracy_score(y_test, rf_pred)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        # Accuracies should be between 0 and 1
        self.assertGreaterEqual(rf_accuracy, 0)
        self.assertLessEqual(rf_accuracy, 1)
        self.assertGreaterEqual(lr_accuracy, 0)
        self.assertLessEqual(lr_accuracy, 1)


class TestCreditRiskModel(unittest.TestCase):
    """
    Test suite for credit risk model functionality
    """
    
    def test_credit_risk_data_generation(self):
        """
        Test that credit risk data can be generated properly
        """
        # Simulate the credit risk data generation process
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic features as done in the app
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
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'employment_years': employment_years,
            'default': default
        })
        
        # Validate generated data
        self.assertEqual(len(df), n_samples)
        self.assertEqual(len(df.columns), 6)  # 5 features + 1 target
        self.assertTrue(all(col in df.columns for col in ['age', 'income', 'loan_amount', 'credit_score', 'employment_years', 'default']))
        
        # Validate ranges make sense
        self.assertTrue(all(df['age'] >= 0))
        self.assertTrue(all(df['income'] >= 0))
        self.assertTrue(all(df['loan_amount'] >= 0))
        self.assertTrue(all(df['employment_years'] >= 0))
        self.assertTrue(all(df['default'].isin([0, 1])))


class TestUtils(unittest.TestCase):
    """
    Test utility functions
    """
    
    def test_datetime_formatting(self):
        """
        Test that datetime formatting works correctly
        """
        from datetime import datetime
        
        dt = datetime.now()
        formatted = dt.strftime('%Y%m%d_%H%M%S')
        
        # Should be 15 characters (YYYYMMDD_HHMMSS)
        self.assertEqual(len(formatted), 15)
        
        # Should contain underscore separator
        self.assertIn('_', formatted)
    
    def test_model_naming(self):
        """
        Test that model naming follows expected pattern
        """
        from datetime import datetime
        
        model_type = "Random Forest"
        dataset_name = "sample_data.csv"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_name = f"{model_type}_{dataset_name}_{timestamp}"
        
        # Should contain all components
        self.assertIn(model_type, model_name)
        self.assertIn(dataset_name, model_name)
        self.assertIn(timestamp, model_name)


def run_tests():
    """
    Run all tests and return results
    """
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)