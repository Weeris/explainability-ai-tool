"""
Configuration file for the Explainability AI Tool
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Application settings
    APP_NAME = "Explainability AI Tool (Noor)"
    VERSION = "0.1.0"
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Security settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "True").lower() == "true"
    
    # Database settings (if needed)
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///explainability_tool.db")
    
    # Model settings
    MAX_FEATURES_FOR_SHAP = int(os.environ.get("MAX_FEATURES_FOR_SHAP", "50"))
    SHAP_SAMPLE_SIZE = int(os.environ.get("SHAP_SAMPLE_SIZE", "100"))
    
    # Privacy settings
    DIFFERENTIAL_PRIVACY_EPSILON = float(os.environ.get("DP_EPSILON", "1.0"))
    MAX_ROWS_FOR_UPLOAD = int(os.environ.get("MAX_ROWS_FOR_UPLOAD", "10000"))
    
    # Logging settings
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    
    # Model explainability settings
    SUPPORTED_MODELS = [
        "RandomForestClassifier",
        "LogisticRegression",
        "GradientBoostingClassifier",
        "XGBoost",
        "NeuralNetwork"
    ]

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration by name"""
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "default")
    return config_by_name.get(config_name, config_by_name["default"])