#!/usr/bin/env python3
"""
Comprehensive Unit Tests for IA Continu Solution API
Tests all endpoints, database integration, and ML pipeline functionality
"""

import pytest
import requests
import time
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database.db_manager import DatabaseManager
from monitoring.discord_notifier import DiscordNotifier


class TestAPIEndpoints:
    """Test suite for API endpoints"""

    @classmethod
    def setup_class(cls):
        """Setup for test class"""
        cls.base_url = "http://localhost:9000"
        cls.timeout = 30

        # Check if API is running
        try:
            response = requests.get(f"{cls.base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.fail(
                    "API server is not running. Start with: python src/api/main.py"
                )
        except requests.ConnectionError:
            pytest.fail("API server is not running. Start with: python src/api/main.py")

    def test_health_endpoint(self):
        """Test health endpoint returns 200 OK with correct format"""
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)

        assert response.status_code == 200, "Health endpoint should return 200"

        data = response.json()
        assert "status" in data, "Response should contain 'status' field"
        assert data["status"] == "ok", "Status should be 'ok'"
        assert "timestamp" in data, "Response should contain 'timestamp' field"
        assert "version" in data, "Response should contain 'version' field"
        assert data["version"] == "2.0.0", "Version should be '2.0.0'"

    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = requests.get(f"{self.base_url}/", timeout=self.timeout)

        assert response.status_code == 200, "Root endpoint should return 200"

        data = response.json()
        assert "message" in data, "Response should contain 'message' field"
        assert "version" in data, "Response should contain 'version' field"
        assert (
            "IA Continu Solution" in data["message"]
        ), "Message should contain project name"

    def test_generate_endpoint_valid_input(self):
        """Test dataset generation with valid input"""
        payload = {"samples": 150}

        response = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )

        assert (
            response.status_code == 200
        ), "Generate endpoint should work with valid input"

        data = response.json()
        assert "generation_id" in data, "Response should contain 'generation_id'"
        assert "samples_created" in data, "Response should contain 'samples_created'"
        assert "timestamp" in data, "Response should contain 'timestamp'"
        assert (
            data["samples_created"] == 150
        ), "Should create requested number of samples"
        assert isinstance(data["generation_id"], int), "Generation ID should be integer"

    def test_generate_endpoint_invalid_input(self):
        """Test dataset generation with invalid input"""
        # Test too few samples
        payload = {"samples": 50}
        response = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )
        assert response.status_code == 422, "Should reject too few samples"

        # Test too many samples
        payload = {"samples": 15000}
        response = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )
        assert response.status_code == 422, "Should reject too many samples"

        # Test invalid type
        payload = {"samples": "invalid"}
        response = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )
        assert response.status_code == 422, "Should reject non-integer samples"

    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input"""
        payload = {"features": [0.5, -0.3]}

        response = requests.post(
            f"{self.base_url}/predict", json=payload, timeout=self.timeout
        )

        assert (
            response.status_code == 200
        ), "Predict endpoint should work with valid input"

        data = response.json()
        assert "prediction" in data, "Response should contain 'prediction'"
        assert "model_version" in data, "Response should contain 'model_version'"
        assert "confidence" in data, "Response should contain 'confidence'"
        assert "timestamp" in data, "Response should contain 'timestamp'"

        # Validate prediction is binary
        assert data["prediction"] in [0, 1], "Prediction should be 0 or 1"

        # Validate confidence is between 0 and 1
        assert 0 <= data["confidence"] <= 1, "Confidence should be between 0 and 1"

    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid input"""
        # Test too few features
        payload = {"features": [1.0]}
        response = requests.post(
            f"{self.base_url}/predict", json=payload, timeout=self.timeout
        )
        assert response.status_code == 422, "Should reject too few features"

        # Test too many features
        payload = {"features": [1.0, 2.0, 3.0]}
        response = requests.post(
            f"{self.base_url}/predict", json=payload, timeout=self.timeout
        )
        assert response.status_code == 422, "Should reject too many features"

        # Test non-numeric features
        payload = {"features": ["a", "b"]}
        response = requests.post(
            f"{self.base_url}/predict", json=payload, timeout=self.timeout
        )
        assert response.status_code == 422, "Should reject non-numeric features"

        # Test missing features
        payload = {}
        response = requests.post(
            f"{self.base_url}/predict", json=payload, timeout=self.timeout
        )
        assert response.status_code == 422, "Should reject missing features"

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = requests.get(f"{self.base_url}/model/info", timeout=self.timeout)

        assert response.status_code == 200, "Model info endpoint should work"

        data = response.json()
        assert "model_version" in data, "Response should contain 'model_version'"
        assert "model_loaded" in data, "Response should contain 'model_loaded'"
        assert "model_type" in data, "Response should contain 'model_type'"
        assert "timestamp" in data, "Response should contain 'timestamp'"

        assert isinstance(data["model_loaded"], bool), "model_loaded should be boolean"

    def test_datasets_list_endpoint(self):
        """Test datasets list endpoint"""
        response = requests.get(f"{self.base_url}/datasets/list", timeout=self.timeout)

        assert response.status_code == 200, "Datasets list endpoint should work"

        data = response.json()
        assert "datasets" in data, "Response should contain 'datasets'"
        assert "total_datasets" in data, "Response should contain 'total_datasets'"

        assert isinstance(data["datasets"], list), "datasets should be a list"
        assert isinstance(
            data["total_datasets"], int
        ), "total_datasets should be integer"

        # If there are datasets, validate structure
        if data["datasets"]:
            dataset = data["datasets"][0]
            assert "generation_id" in dataset, "Dataset should have generation_id"
            assert "samples_count" in dataset, "Dataset should have samples_count"
            assert "created_at" in dataset, "Dataset should have created_at"

    def test_retrain_endpoint(self):
        """Test model retraining endpoint"""
        # First generate a dataset to ensure we have training data
        payload = {"samples": 200}
        gen_response = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )
        assert gen_response.status_code == 200, "Should generate dataset for training"

        # Wait a moment for database write
        time.sleep(1)

        # Test retraining
        response = requests.post(f"{self.base_url}/retrain", timeout=60)

        assert response.status_code == 200, "Retrain endpoint should work"

        data = response.json()
        assert "status" in data, "Response should contain 'status'"
        assert "model_version" in data, "Response should contain 'model_version'"
        assert "training_samples" in data, "Response should contain 'training_samples'"
        assert "accuracy" in data, "Response should contain 'accuracy'"
        assert "timestamp" in data, "Response should contain 'timestamp'"

        assert data["status"] == "success", "Status should be 'success'"
        assert 0 <= data["accuracy"] <= 1, "Accuracy should be between 0 and 1"
        assert data["training_samples"] > 0, "Should have training samples"

    def test_api_performance(self):
        """Test API response times"""
        # Test health endpoint performance
        start_time = time.time()
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        health_time = time.time() - start_time

        assert response.status_code == 200, "Health endpoint should work"
        assert (
            health_time < 2.0
        ), f"Health endpoint should respond in <2s, got {health_time:.3f}s"

        # Test prediction performance
        start_time = time.time()
        payload = {"features": [0.0, 0.0]}
        response = requests.post(
            f"{self.base_url}/predict", json=payload, timeout=self.timeout
        )
        predict_time = time.time() - start_time

        assert response.status_code == 200, "Predict endpoint should work"
        assert (
            predict_time < 1.0
        ), f"Predict endpoint should respond in <1s, got {predict_time:.3f}s"


class TestDatabaseIntegration:
    """Test suite for database integration"""

    @classmethod
    def setup_class(cls):
        """Setup for test class"""
        cls.db_manager = DatabaseManager("test_db.sqlite")

    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        import os

        if os.path.exists("test_db.sqlite"):
            os.remove("test_db.sqlite")

    def test_database_initialization(self):
        """Test database initialization"""
        assert (
            self.db_manager.init_database()
        ), "Database should initialize successfully"

    def test_database_health_check(self):
        """Test database health check"""
        health = self.db_manager.health_check()
        assert health["status"] == "healthy", "Database should be healthy"

    def test_store_and_retrieve_dataset(self):
        """Test storing and retrieving datasets"""
        generation_id = int(time.time())
        samples = [(0.1, 0.2, 1), (0.3, 0.4, 0), (0.5, 0.6, 1)]

        # Store dataset
        success = self.db_manager.store_dataset(
            generation_id, len(samples), 12, samples
        )
        assert success, "Should store dataset successfully"

        # Retrieve dataset
        result = self.db_manager.get_latest_dataset()
        assert result is not None, "Should retrieve dataset"

        retrieved_id, retrieved_samples = result
        assert retrieved_id == generation_id, "Should retrieve correct generation ID"
        assert len(retrieved_samples) == len(
            samples
        ), "Should retrieve correct number of samples"

    def test_store_and_retrieve_model(self):
        """Test storing and retrieving models"""
        version = f"test_v{int(time.time())}"
        accuracy = 0.95
        training_samples = 1000

        # Store model
        success = self.db_manager.store_model(version, accuracy, training_samples, True)
        assert success, "Should store model successfully"

        # Retrieve active model
        model = self.db_manager.get_active_model()
        assert model is not None, "Should retrieve active model"
        assert model["version"] == version, "Should retrieve correct model version"
        assert model["accuracy"] == accuracy, "Should retrieve correct accuracy"


class TestDiscordIntegration:
    """Test suite for Discord integration"""

    @classmethod
    def setup_class(cls):
        """Setup for test class"""
        cls.notifier = DiscordNotifier()
        cls.webhook_configured = bool(os.getenv("DISCORD_WEBHOOK_URL"))

    def test_discord_configuration(self):
        """Test Discord webhook configuration"""
        if not self.webhook_configured:
            pytest.skip("Discord webhook not configured")

        assert self.notifier.webhook_url is not None, "Webhook URL should be configured"

    def test_discord_notification(self):
        """Test Discord notification functionality"""
        if not self.webhook_configured:
            pytest.skip("Discord webhook not configured")

        # Test basic notification
        success = self.notifier.send_notification(
            "🧪 Test notification from unit tests", "Info"
        )
        assert success, "Should send Discord notification successfully"

    def test_discord_webhook_test(self):
        """Test Discord webhook test function"""
        if not self.webhook_configured:
            pytest.skip("Discord webhook not configured")

        success = self.notifier.test_webhook()
        assert success, "Discord webhook test should pass"


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)

    # Run pytest with verbose output
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short", "-x"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    exit(0 if success else 1)
