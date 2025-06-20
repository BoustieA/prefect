#!/usr/bin/env python3
"""
IA Continu Solution - Main Application Launcher
Professional entry point for the complete ML pipeline system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

<<<<<<< HEAD
import numpy as np

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import joblib
from sklearn.linear_model import LinearRegression
import os
from ia_continu_solution.utils.utilities import send_discord_embed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
=======
# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
>>>>>>> origin/jour1

from config.settings import get_config, print_config
from src.mlflow_service.mlflow_manager import MLflowManager
from src.monitoring.discord_notifier import DiscordNotifier
from src.database.db_manager import DatabaseManager

<<<<<<< HEAD

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning basic API information."""
    return {
        "message": "IA Continu Solution API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring services."""
    try:
        # Perform basic health checks here
        # You can add database connectivity, external service checks, etc.
=======
def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "fastapi", "uvicorn", "requests", "numpy", "sklearn",
        "mlflow", "pydantic", "joblib"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    
    try:
        # Import and run the API
        from src.api.main import app
        import uvicorn
        
        config = get_config()
        uvicorn.run(
            app, 
            host=config["api"]["host"], 
            port=config["api"]["port"],
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False

def start_mlflow_server():
    """Start MLflow tracking server"""
    print("🔬 Starting MLflow server...")
    
    manager = MLflowManager()
    success = manager.start_mlflow_server()
    
    if success:
        print("✅ MLflow server started successfully")
        print(f"   Access UI at: {manager.tracking_uri}")
        return True
    else:
        print("❌ Failed to start MLflow server")
        return False

def test_discord_webhook():
    """Test Discord webhook functionality"""
    print("📱 Testing Discord webhook...")
    
    notifier = DiscordNotifier()
    
    if not notifier.webhook_url:
        print("⚠️  Discord webhook not configured")
        print("   Set DISCORD_WEBHOOK_URL environment variable")
        return False
    
    success = notifier.test_webhook()
    
    if success:
        print("✅ Discord webhook test successful")
        return True
    else:
        print("❌ Discord webhook test failed")
        return False

def run_comprehensive_tests():
    """Run the complete test suite"""
    print("🧪 Running comprehensive test suite...")
    
    try:
        # Run the test suite
        from tests.test_api import run_comprehensive_tests
        success = run_comprehensive_tests()
        
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return success
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def check_system_health():
    """Check overall system health"""
    print("🏥 Checking system health...")
    
    health_status = {
        "database": False,
        "api": False,
        "mlflow": False,
        "discord": False
    }
    
    # Check database
    try:
        db_manager = DatabaseManager()
        health = db_manager.health_check()
        health_status["database"] = health["status"] == "healthy"
        print(f"   Database: {'✅' if health_status['database'] else '❌'}")
    except Exception as e:
        print(f"   Database: ❌ ({e})")
    
    # Check API
    try:
        import requests
        response = requests.get("http://localhost:9000/health", timeout=5)
        health_status["api"] = response.status_code == 200
        print(f"   API: {'✅' if health_status['api'] else '❌'}")
    except Exception:
        print("   API: ❌ (not running)")
    
    # Check MLflow
    try:
        manager = MLflowManager()
        health_status["mlflow"] = manager.is_mlflow_running()
        print(f"   MLflow: {'✅' if health_status['mlflow'] else '❌'}")
    except Exception:
        print("   MLflow: ❌ (not running)")
    
    # Check Discord
    try:
        notifier = DiscordNotifier()
        health_status["discord"] = bool(notifier.webhook_url)
        print(f"   Discord: {'✅' if health_status['discord'] else '⚠️'} {'(configured)' if health_status['discord'] else '(not configured)'}")
    except Exception:
        print("   Discord: ❌ (error)")
    
    return health_status

def main():
    """Main application launcher"""
    print("🤖 IA Continu Solution - Professional ML Pipeline")
    print("=" * 55)
    print()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("✅ All dependencies are installed")
    print()
    
    # Show configuration
    config = get_config()
    print(f"🌍 Environment: {config['environment']}")
    print(f"🔗 API will run on: http://{config['api']['host']}:{config['api']['port']}")
    print(f"🔬 MLflow UI: {config['mlflow']['tracking_uri']}")
    print(f"📱 Discord: {'Configured' if config['discord']['webhook_url'] else 'Not configured'}")
    print()
    
    # Main menu
    while True:
        print("Choose an option:")
        print("1. 🚀 Start API Server")
        print("2. 🔬 Start MLflow Server")
        print("3. 📱 Test Discord Webhook")
        print("4. 🧪 Run Comprehensive Tests")
        print("5. 🏥 Check System Health")
        print("6. ⚙️  Show Configuration")
        print("7. 📋 Start All Services")
        print("8. ❌ Exit")
>>>>>>> origin/jour1
        
        try:
            choice = input("\nEnter choice (1-8): ").strip()
            print()
            
            if choice == "1":
                start_api_server()
                
            elif choice == "2":
                start_mlflow_server()
                input("\nPress Enter to continue...")
                
            elif choice == "3":
                test_discord_webhook()
                input("\nPress Enter to continue...")
                
            elif choice == "4":
                run_comprehensive_tests()
                input("\nPress Enter to continue...")
                
            elif choice == "5":
                health = check_system_health()
                print(f"\nOverall Status: {'✅ Healthy' if all(health.values()) else '⚠️ Issues detected'}")
                input("\nPress Enter to continue...")
                
            elif choice == "6":
                print_config()
                input("\nPress Enter to continue...")
                
            elif choice == "7":
                print("🔄 Starting all services...")
                
                # Start MLflow first
                mlflow_success = start_mlflow_server()
                if mlflow_success:
                    time.sleep(2)  # Give MLflow time to start
                
                # Test Discord
                discord_success = test_discord_webhook()
                
                print(f"\n📊 Services Status:")
                print(f"   MLflow: {'✅' if mlflow_success else '❌'}")
                print(f"   Discord: {'✅' if discord_success else '⚠️'}")
                print(f"\n🚀 Starting API server (this will block)...")
                
                # Start API (this will block)
                start_api_server()
                
            elif choice == "8":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Operation cancelled")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPress Enter to continue...")





MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pkl")
DB_PATH = "sqlite:///"+os.getenv("DB_PATH", "/data/db.sqlite3")

orm_model=""
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#orm_model.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utilitaires
def get_model(path):#TODO
    #changer pour récupérer le dernier modèle.
    model = mlflow.pyfunc.load_model(model_uri="models:/my_model/latest")
    model = joblib.load(path)
    return model

def get_last_dataset(db):
    """Get last batch of data inserted"""
    index_col=-1
    max_index = db.query(func.max(getattr(orm_model, index_col))).scalar()
    return db.query(orm_model).filter(getattr(orm_model, index_col) == max_index).all()




# Route prédiction
@app.get("/predict")
async def predict(db: Session = Depends(get_db)):
    """predict output of last model for last batch of data"""
    try:
        prediction = model.predict(features)
        model = get_model(MODEL_PATH)
        dataset = get_last_dataset(db)
        features = [[row.feature1, row.feature2] for row in dataset]
        prediction = model.predict(features)
        if not (prediction.astype(bool)==prediction).all():
            logger.error(f"Output du modèle non binaire")
    except Exception as e:
        message = str(e)
        status = "Erreur"
        send_discord_embed(message, status)
    return {
        "prediction": list(prediction)
    }








if __name__ == "__main__":
    main()
