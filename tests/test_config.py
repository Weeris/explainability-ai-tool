"""
Unit tests for the configuration module
"""
import unittest
import os
import sys

# Add the project root to the path so we can import config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config


class TestConfig(unittest.TestCase):
    """
    Test suite for configuration settings
    """
    
    def test_base_config_properties(self):
        """
        Test that base configuration has expected properties
        """
        cfg = config.Config()
        
        # Check basic properties
        self.assertIsInstance(cfg.APP_NAME, str)
        self.assertIsInstance(cfg.VERSION, str)
        self.assertIsInstance(cfg.DEBUG, bool)
        self.assertIsInstance(cfg.SECRET_KEY, str)
        
        # Check numeric settings
        self.assertIsInstance(cfg.MAX_FEATURES_FOR_SHAP, int)
        self.assertIsInstance(cfg.SHAP_SAMPLE_SIZE, int)
        self.assertIsInstance(cfg.DIFFERENTIAL_PRIVACY_EPSILON, float)
        self.assertIsInstance(cfg.MAX_ROWS_FOR_UPLOAD, int)
        self.assertIsInstance(cfg.LOG_LEVEL, str)
        
        # Check lists
        self.assertIsInstance(cfg.SUPPORTED_MODELS, list)
        self.assertTrue(len(cfg.SUPPORTED_MODELS) > 0)
    
    def test_config_values(self):
        """
        Test that configuration values are reasonable
        """
        cfg = config.Config()
        
        # App name should not be empty
        self.assertGreater(len(cfg.APP_NAME), 0)
        
        # Version should follow semantic versioning pattern
        self.assertRegex(cfg.VERSION, r'^\d+\.\d+\.\d+$')
        
        # Sample sizes should be positive
        self.assertGreater(cfg.SHAP_SAMPLE_SIZE, 0)
        self.assertGreater(cfg.MAX_ROWS_FOR_UPLOAD, 0)
        
        # Epsilon should be positive
        self.assertGreater(cfg.DIFFERENTIAL_PRIVACY_EPSILON, 0)
    
    def test_environment_override(self):
        """
        Test that environment variables can override config values
        """
        # Temporarily set environment variable
        original_debug = os.environ.get('DEBUG')
        os.environ['DEBUG'] = 'True'
        
        # Reload config to pick up new environment variable
        import importlib
        importlib.reload(config)
        
        # Check that debug is now True
        cfg = config.Config()
        self.assertTrue(cfg.DEBUG)
        
        # Restore original value
        if original_debug is not None:
            os.environ['DEBUG'] = original_debug
        else:
            del os.environ['DEBUG']
    
    def test_configuration_types(self):
        """
        Test that configuration values have correct types
        """
        cfg = config.Config()
        
        # Boolean values
        self.assertIsInstance(cfg.DEBUG, bool)
        self.assertIsInstance(cfg.SESSION_COOKIE_SECURE, bool)
        
        # String values
        self.assertIsInstance(cfg.APP_NAME, str)
        self.assertIsInstance(cfg.VERSION, str)
        self.assertIsInstance(cfg.SECRET_KEY, str)
        self.assertIsInstance(cfg.DATABASE_URL, str)
        self.assertIsInstance(cfg.LOG_LEVEL, str)
        
        # Integer values
        self.assertIsInstance(cfg.MAX_FEATURES_FOR_SHAP, int)
        self.assertIsInstance(cfg.SHAP_SAMPLE_SIZE, int)
        self.assertIsInstance(cfg.MAX_ROWS_FOR_UPLOAD, int)
        
        # Float values
        self.assertIsInstance(cfg.DIFFERENTIAL_PRIVACY_EPSILON, float)
    
    def test_supported_models_list(self):
        """
        Test that supported models list contains expected models
        """
        cfg = config.Config()
        
        # Should contain at least one model
        self.assertGreater(len(cfg.SUPPORTED_MODELS), 0)
        
        # Should contain expected model types
        expected_models = [
            "RandomForestClassifier",
            "LogisticRegression",
            "GradientBoostingClassifier"
        ]
        
        for model in expected_models:
            if model in cfg.SUPPORTED_MODELS:
                break
        else:
            self.fail(f"Expected at least one of {expected_models} in SUPPORTED_MODELS")


class TestEnvironmentConfigs(unittest.TestCase):
    """
    Test environment-specific configurations
    """
    
    def test_development_config(self):
        """
        Test development configuration
        """
        dev_cfg = config.DevelopmentConfig()
        
        self.assertTrue(dev_cfg.DEBUG)
        self.assertEqual(dev_cfg.APP_NAME, config.Config.APP_NAME)
    
    def test_production_config(self):
        """
        Test production configuration
        """
        prod_cfg = config.ProductionConfig()
        
        self.assertFalse(prod_cfg.DEBUG)
        self.assertEqual(prod_cfg.APP_NAME, config.Config.APP_NAME)
    
    def test_testing_config(self):
        """
        Test testing configuration
        """
        test_cfg = config.TestingConfig()
        
        self.assertTrue(test_cfg.TESTING)
        self.assertTrue(test_cfg.DEBUG)
        self.assertEqual(test_cfg.APP_NAME, config.Config.APP_NAME)
    
    def test_config_by_name(self):
        """
        Test getting configuration by name
        """
        # Test getting each config type
        dev_cfg = config.get_config("development")
        self.assertIsInstance(dev_cfg, config.DevelopmentConfig)
        
        prod_cfg = config.get_config("production")
        self.assertIsInstance(prod_cfg, config.ProductionConfig)
        
        test_cfg = config.get_config("testing")
        self.assertIsInstance(test_cfg, config.TestingConfig)
        
        # Test default config
        default_cfg = config.get_config("default")
        self.assertIsInstance(default_cfg, config.DevelopmentConfig)
        
        # Test unknown config defaults to default
        unknown_cfg = config.get_config("unknown")
        self.assertIsInstance(unknown_cfg, config.DevelopmentConfig)


def run_tests():
    """
    Run all configuration tests
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