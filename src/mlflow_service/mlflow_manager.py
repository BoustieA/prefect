#!/usr/bin/env python3
"""
MLflow Service Manager for IA Continu Solution
Handles MLflow server management and experiment tracking
"""

import os
import subprocess
import time
import requests
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowManager:
    """MLflow service manager"""
    
    def __init__(self, tracking_uri="http://localhost:5000", backend_store_uri="sqlite:///mlflow.db"):
        self.tracking_uri = tracking_uri
        self.backend_store_uri = backend_store_uri
        self.mlflow_process = None
        self.experiment_name = "ia_continu_solution"
        
        # Create MLflow directory
        self.mlflow_dir = Path("mlruns")
        self.mlflow_dir.mkdir(exist_ok=True)
        
    def start_mlflow_server(self, host="0.0.0.0", port=5000):
        """Start MLflow tracking server"""
        try:
            # Check if MLflow is already running
            if self.is_mlflow_running():
                logger.info("MLflow server is already running")
                return True
            
            logger.info(f"Starting MLflow server on {host}:{port}")
            
            # Start MLflow server
            cmd = [
                "mlflow", "server",
                "--backend-store-uri", self.backend_store_uri,
                "--default-artifact-root", "./mlruns",
                "--host", host,
                "--port", str(port)
            ]
            
            self.mlflow_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_retries = 30
            for i in range(max_retries):
                if self.is_mlflow_running():
                    logger.info("MLflow server started successfully")
                    return True
                time.sleep(1)
            
            logger.error("MLflow server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start MLflow server: {e}")
            return False
    
    def stop_mlflow_server(self):
        """Stop MLflow tracking server"""
        try:
            if self.mlflow_process:
                self.mlflow_process.terminate()
                self.mlflow_process.wait(timeout=10)
                logger.info("MLflow server stopped")
                return True
        except Exception as e:
            logger.error(f"Failed to stop MLflow server: {e}")
            return False
    
    def is_mlflow_running(self):
        """Check if MLflow server is running"""
        try:
            response = requests.get(f"{self.tracking_uri}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            return False
    
    def log_model_training(self, model, params, metrics, model_name="ia_continu_model"):
        """Log model training to MLflow"""
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=model_name
                )
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Model logged to MLflow with run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
            return None
    
    def get_latest_model(self, model_name="ia_continu_model"):
        """Get latest model from MLflow"""
        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            
            # Get latest version of the model
            latest_version = client.get_latest_versions(
                model_name, 
                stages=["Production", "Staging", "None"]
            )
            
            if latest_version:
                model_version = latest_version[0]
                model_uri = f"models:/{model_name}/{model_version.version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded model {model_name} version {model_version.version}")
                return model
            else:
                logger.warning(f"No model found with name {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            return None
    
    def get_experiment_runs(self, limit=10):
        """Get recent experiment runs"""
        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=limit,
                    order_by=["start_time DESC"]
                )
                
                run_data = []
                for run in runs:
                    run_data.append({
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "metrics": run.data.metrics,
                        "params": run.data.params
                    })
                
                return run_data
            else:
                logger.warning(f"Experiment {self.experiment_name} not found")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    def health_check(self):
        """Perform MLflow health check"""
        try:
            # Check if server is running
            if not self.is_mlflow_running():
                return {
                    "status": "unhealthy",
                    "message": "MLflow server is not running",
                    "tracking_uri": self.tracking_uri
                }
            
            # Check experiment setup
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment:
                return {
                    "status": "healthy",
                    "message": "MLflow server is running and experiment is configured",
                    "tracking_uri": self.tracking_uri,
                    "experiment_name": self.experiment_name,
                    "experiment_id": experiment.experiment_id
                }
            else:
                return {
                    "status": "warning",
                    "message": "MLflow server is running but experiment is not configured",
                    "tracking_uri": self.tracking_uri
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"MLflow health check failed: {str(e)}",
                "tracking_uri": self.tracking_uri
            }

def main():
    """Main MLflow service execution"""
    print("🔬 MLflow Service Manager")
    print("=" * 30)
    
    manager = MLflowManager()
    
    print("Choose an option:")
    print("1. Start MLflow server")
    print("2. Stop MLflow server")
    print("3. Check MLflow status")
    print("4. Setup experiment")
    print("5. Health check")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting MLflow server...")
            success = manager.start_mlflow_server()
            if success:
                print("✅ MLflow server started successfully")
                print(f"   Access UI at: {manager.tracking_uri}")
            else:
                print("❌ Failed to start MLflow server")
                
        elif choice == "2":
            print("\n⏹️ Stopping MLflow server...")
            success = manager.stop_mlflow_server()
            if success:
                print("✅ MLflow server stopped")
            else:
                print("❌ Failed to stop MLflow server")
                
        elif choice == "3":
            print("\n🔍 Checking MLflow status...")
            if manager.is_mlflow_running():
                print("✅ MLflow server is running")
            else:
                print("❌ MLflow server is not running")
                
        elif choice == "4":
            print("\n⚙️ Setting up experiment...")
            success = manager.setup_experiment()
            if success:
                print("✅ Experiment setup completed")
            else:
                print("❌ Failed to setup experiment")
                
        elif choice == "5":
            print("\n🏥 Performing health check...")
            health = manager.health_check()
            print(f"Status: {health['status']}")
            print(f"Message: {health['message']}")
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
