#!/usr/bin/env python3
"""
Complete System Tests for IA Continu Solution using pytest
Tests all components: API, Database, Discord, MLflow integration
"""

import pytest
import requests
import time
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database.db_manager import DatabaseManager
from monitoring.discord_notifier import DiscordNotifier

# Test configuration
API_BASE_URL = "http://localhost:9000"
TIMEOUT = 30


class TestSystemHealth:
    """Test system health and configuration"""

    def test_environment_variables(self):
        """Test that required environment variables are set"""
        discord_url = os.getenv("DISCORD_WEBHOOK_URL")
        assert discord_url is not None, "DISCORD_WEBHOOK_URL should be set"
        assert (
            "discord.com/api/webhooks" in discord_url
        ), "Should be a valid Discord webhook URL"

    def test_api_server_running(self):
        """Test that API server is accessible"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            assert (
                response.status_code == 200
            ), "API server should be running and healthy"
        except requests.ConnectionError:
            pytest.fail("API server is not running. Start with: python src/api/main.py")


class TestAPIEndpoints:
    """Test all API endpoints"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method"""
        self.base_url = API_BASE_URL

        # Ensure API is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API server is not running")
        except requests.ConnectionError:
            pytest.skip("API server is not running")

    def test_health_endpoint(self):
        """Test health endpoint returns correct format"""
        response = requests.get(f"{self.base_url}/health", timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{self.base_url}/", timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "IA Continu Solution" in data["message"]

    def test_generate_endpoint_valid(self):
        """Test dataset generation with valid parameters"""
        payload = {"samples": 150}
        response = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "generation_id" in data
        assert "samples_created" in data
        assert "timestamp" in data
        assert data["samples_created"] == 150
        assert isinstance(data["generation_id"], int)

    def test_generate_endpoint_invalid_samples(self):
        """Test dataset generation with invalid sample counts"""
        # Too few samples
        response = requests.post(
            f"{self.base_url}/generate", json={"samples": 50}, timeout=TIMEOUT
        )
        assert response.status_code == 422

        # Too many samples
        response = requests.post(
            f"{self.base_url}/generate", json={"samples": 15000}, timeout=TIMEOUT
        )
        assert response.status_code == 422

    def test_predict_endpoint_valid(self):
        """Test prediction with valid features"""
        payload = {"features": [0.5, -0.3]}
        response = requests.post(
            f"{self.base_url}/predict", json=payload, timeout=TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "prediction" in data
        assert "model_version" in data
        assert "confidence" in data
        assert "timestamp" in data

        # Validate prediction is binary
        assert data["prediction"] in [0, 1]
        assert 0 <= data["confidence"] <= 1

    def test_predict_endpoint_invalid_features(self):
        """Test prediction with invalid features"""
        # Too few features
        response = requests.post(
            f"{self.base_url}/predict", json={"features": [1.0]}, timeout=TIMEOUT
        )
        assert response.status_code == 422

        # Too many features
        response = requests.post(
            f"{self.base_url}/predict",
            json={"features": [1.0, 2.0, 3.0]},
            timeout=TIMEOUT,
        )
        assert response.status_code == 422

        # Non-numeric features
        response = requests.post(
            f"{self.base_url}/predict", json={"features": ["a", "b"]}, timeout=TIMEOUT
        )
        assert response.status_code == 422

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = requests.get(f"{self.base_url}/model/info", timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "model_version" in data
        assert "model_loaded" in data
        assert "model_type" in data
        assert "timestamp" in data
        assert isinstance(data["model_loaded"], bool)

    def test_datasets_list_endpoint(self):
        """Test datasets list endpoint"""
        response = requests.get(f"{self.base_url}/datasets/list", timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "datasets" in data
        assert "total_datasets" in data
        assert isinstance(data["datasets"], list)
        assert isinstance(data["total_datasets"], int)

    def test_retrain_endpoint(self):
        """Test model retraining endpoint"""
        # First generate a dataset to ensure training data
        gen_payload = {"samples": 200}
        gen_response = requests.post(
            f"{self.base_url}/generate", json=gen_payload, timeout=TIMEOUT
        )
        assert gen_response.status_code == 200

        # Wait for database write
        time.sleep(1)

        # Test retraining
        response = requests.post(f"{self.base_url}/retrain", timeout=60)

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_version" in data
        assert "training_samples" in data
        assert "accuracy" in data
        assert "timestamp" in data

        assert data["status"] == "success"
        assert 0 <= data["accuracy"] <= 1
        assert data["training_samples"] > 0


class TestDatabaseIntegration:
    """Test database functionality"""

    @pytest.fixture
    def db_manager(self):
        """Create a test database manager"""
        return DatabaseManager("test_db.sqlite")

    def test_database_initialization(self, db_manager):
        """Test database initialization"""
        assert db_manager.init_database()

    def test_database_health_check(self, db_manager):
        """Test database health check"""
        health = db_manager.health_check()
        assert health["status"] == "healthy"

    def test_store_and_retrieve_dataset(self, db_manager):
        """Test storing and retrieving datasets"""
        generation_id = int(time.time())
        samples = [(0.1, 0.2, 1), (0.3, 0.4, 0), (0.5, 0.6, 1)]

        # Store dataset
        success = db_manager.store_dataset(generation_id, len(samples), 12, samples)
        assert success

        # Retrieve dataset
        result = db_manager.get_latest_dataset()
        assert result is not None

        retrieved_id, retrieved_samples = result
        assert retrieved_id == generation_id
        assert len(retrieved_samples) == len(samples)

    def test_store_and_retrieve_model(self, db_manager):
        """Test storing and retrieving models"""
        version = f"test_v{int(time.time())}"
        accuracy = 0.95
        training_samples = 1000

        # Store model
        success = db_manager.store_model(version, accuracy, training_samples, True)
        assert success

        # Retrieve active model
        model = db_manager.get_active_model()
        assert model is not None
        assert model["version"] == version
        assert model["accuracy"] == accuracy

    def teardown_method(self):
        """Cleanup test database"""
        if os.path.exists("test_db.sqlite"):
            os.remove("test_db.sqlite")


class TestDiscordIntegration:
    """Test Discord webhook functionality"""

    @pytest.fixture
    def notifier(self):
        """Create Discord notifier"""
        return DiscordNotifier()

    def test_discord_configuration(self, notifier):
        """Test Discord webhook configuration"""
        assert notifier.webhook_url is not None, "Discord webhook should be configured"
        assert "discord.com/api/webhooks" in notifier.webhook_url

    def test_discord_notification_success(self, notifier):
        """Test successful Discord notification"""
        success = notifier.send_notification("🧪 Pytest test notification", "Succès")
        assert success, "Discord notification should be sent successfully"

    def test_discord_notification_failure(self, notifier):
        """Test failure Discord notification"""
        success = notifier.send_notification(
            "🔴 Pytest test failure notification", "Échec"
        )
        assert success, "Discord failure notification should be sent successfully"

    def test_discord_model_retrain_notification(self, notifier):
        """Test model retraining notification"""
        success = notifier.send_model_retrain_success("pytest_v1.0.0", 0.95, 1000)
        assert success, "Model retrain notification should be sent successfully"


class TestPerformance:
    """Test API performance"""

    def test_health_endpoint_performance(self):
        """Test health endpoint response time"""
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        response_time = time.time() - start_time

        assert response.status_code == 200
        assert (
            response_time < 2.0
        ), f"Health endpoint should respond in <2s, got {response_time:.3f}s"

    def test_prediction_performance(self):
        """Test prediction endpoint response time"""
        start_time = time.time()
        payload = {"features": [0.0, 0.0]}
        response = requests.post(
            f"{API_BASE_URL}/predict", json=payload, timeout=TIMEOUT
        )
        response_time = time.time() - start_time

        assert response.status_code == 200
        assert (
            response_time < 1.0
        ), f"Prediction should respond in <1s, got {response_time:.3f}s"


class TestMLPipeline:
    """Test complete ML pipeline"""

    def test_complete_ml_workflow(self):
        """Test the complete ML workflow: generate → predict → retrain → predict"""
        # Step 1: Generate dataset
        gen_response = requests.post(
            f"{API_BASE_URL}/generate", json={"samples": 300}, timeout=TIMEOUT
        )
        assert gen_response.status_code == 200
        gen_data = gen_response.json()
        generation_id = gen_data["generation_id"]

        # Step 2: Make initial prediction
        pred1_response = requests.post(
            f"{API_BASE_URL}/predict", json={"features": [0.5, -0.3]}, timeout=TIMEOUT
        )
        assert pred1_response.status_code == 200
        pred1_data = pred1_response.json()
        initial_model_version = pred1_data["model_version"]

        # Step 3: Retrain model
        time.sleep(1)  # Wait for database write
        retrain_response = requests.post(f"{API_BASE_URL}/retrain", timeout=60)
        assert retrain_response.status_code == 200
        retrain_data = retrain_response.json()
        new_model_version = retrain_data["model_version"]

        # Step 4: Make prediction with new model
        pred2_response = requests.post(
            f"{API_BASE_URL}/predict", json={"features": [0.5, -0.3]}, timeout=TIMEOUT
        )
        assert pred2_response.status_code == 200
        pred2_data = pred2_response.json()

        # Verify workflow
        assert retrain_data["status"] == "success"
        assert retrain_data["training_samples"] == 300
        assert new_model_version != initial_model_version  # Model version should change
        assert pred2_data["model_version"] == new_model_version  # Should use new model

        # Verify dataset is in database
        datasets_response = requests.get(
            f"{API_BASE_URL}/datasets/list", timeout=TIMEOUT
        )
        assert datasets_response.status_code == 200
        datasets_data = datasets_response.json()

        # Check if our generated dataset is in the list
        found_dataset = any(
            d["generation_id"] == generation_id for d in datasets_data["datasets"]
        )
        assert found_dataset, "Generated dataset should be found in database"


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
