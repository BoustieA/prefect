#!/usr/bin/env python3
"""
Configuration settings for IA Continu Solution
Centralized configuration management
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "9000"))
API_VERSION = "2.0.0"

# Database Configuration
DATABASE_PATH = str(DATA_DIR / "ia_continu_solution.db")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_BACKEND_STORE_URI = os.getenv("MLFLOW_BACKEND_STORE_URI", f"sqlite:///{DATA_DIR}/mlflow.db")
MLFLOW_EXPERIMENT_NAME = "ia_continu_solution"

# Discord Configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Model Configuration
DEFAULT_MODEL_TYPE = "LogisticRegression"
MODEL_RANDOM_STATE = 42
MODEL_MAX_ITER = 1000

# Dataset Configuration
MIN_SAMPLES = 100
MAX_SAMPLES = 10000
DEFAULT_SAMPLES = 1000

# Monitoring Configuration
MONITORING_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 10  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"

def get_config():
    """Get configuration dictionary"""
    return {
        "api": {
            "host": API_HOST,
            "port": API_PORT,
            "version": API_VERSION,
            "debug": DEBUG
        },
        "database": {
            "path": DATABASE_PATH
        },
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "backend_store_uri": MLFLOW_BACKEND_STORE_URI,
            "experiment_name": MLFLOW_EXPERIMENT_NAME
        },
        "discord": {
            "webhook_url": DISCORD_WEBHOOK_URL
        },
        "model": {
            "type": DEFAULT_MODEL_TYPE,
            "random_state": MODEL_RANDOM_STATE,
            "max_iter": MODEL_MAX_ITER
        },
        "dataset": {
            "min_samples": MIN_SAMPLES,
            "max_samples": MAX_SAMPLES,
            "default_samples": DEFAULT_SAMPLES
        },
        "monitoring": {
            "interval": MONITORING_INTERVAL,
            "health_check_timeout": HEALTH_CHECK_TIMEOUT
        },
        "logging": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT
        },
        "environment": ENVIRONMENT
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    
    print("⚙️ IA Continu Solution Configuration")
    print("=" * 40)
    
    for section, settings in config.items():
        print(f"\n📋 {section.upper()}:")
        for key, value in settings.items():
            if key == "webhook_url" and value:
                # Mask webhook URL for security
                masked_value = value[:20] + "..." if len(value) > 20 else value
                print(f"   {key}: {masked_value}")
            else:
                print(f"   {key}: {value}")

if __name__ == "__main__":
    print_config()
