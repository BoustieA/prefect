#!/usr/bin/env python3
"""
Database Manager for IA Continu Solution
Handles SQLite database operations for datasets and models
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite database manager for IA Continu Solution"""
    
    def __init__(self, db_path: str = "data/ia_continu_solution.db"):
        self.db_path = db_path
        
        # Ensure data directory exists
        Path(db_path).parent.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def init_database(self) -> bool:
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=1000")
            conn.execute("PRAGMA temp_store=memory")
            cursor = conn.cursor()
            
            # Create datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_id INTEGER UNIQUE,
                    samples_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hour_generated INTEGER
                )
            """)
            
            # Create dataset_samples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_id INTEGER,
                    feature1 REAL,
                    feature2 REAL,
                    target INTEGER,
                    FOREIGN KEY (generation_id) REFERENCES datasets (generation_id)
                )
            """)
            
            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE,
                    accuracy REAL,
                    training_samples INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create experiments table for MLflow integration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    run_id TEXT,
                    model_version TEXT,
                    accuracy REAL,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def store_dataset(self, generation_id: int, samples_count: int, hour_generated: int, 
                     features_targets: List[Tuple[float, float, int]]) -> bool:
        """Store generated dataset in database"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            # Insert dataset metadata
            cursor.execute("""
                INSERT INTO datasets (generation_id, samples_count, hour_generated)
                VALUES (?, ?, ?)
            """, (generation_id, samples_count, hour_generated))
            
            # Insert samples
            for feature1, feature2, target in features_targets:
                cursor.execute("""
                    INSERT INTO dataset_samples (generation_id, feature1, feature2, target)
                    VALUES (?, ?, ?, ?)
                """, (generation_id, feature1, feature2, target))
            
            conn.commit()
            conn.close()
            logger.info(f"Dataset {generation_id} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store dataset: {e}")
            return False
    
    def get_latest_dataset(self) -> Optional[Tuple[int, List[Tuple[float, float, int]]]]:
        """Get the latest dataset for training"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            # Get latest generation_id
            cursor.execute("SELECT generation_id FROM datasets ORDER BY created_at DESC LIMIT 1")
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return None
            
            generation_id = result[0]
            
            # Get samples for this generation
            cursor.execute("""
                SELECT feature1, feature2, target 
                FROM dataset_samples 
                WHERE generation_id = ?
            """, (generation_id,))
            
            samples = cursor.fetchall()
            conn.close()
            
            return generation_id, samples
            
        except Exception as e:
            logger.error(f"Failed to get latest dataset: {e}")
            return None
    
    def store_model(self, version: str, accuracy: float, training_samples: int, 
                   is_active: bool = True) -> bool:
        """Store model information in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Set all models as inactive if this one is active
            if is_active:
                cursor.execute("UPDATE models SET is_active = FALSE")
            
            # Insert new model
            cursor.execute("""
                INSERT INTO models (version, accuracy, training_samples, is_active)
                VALUES (?, ?, ?, ?)
            """, (version, accuracy, training_samples, is_active))
            
            conn.commit()
            conn.close()
            logger.info(f"Model {version} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            return False
    
    def get_active_model(self) -> Optional[Dict]:
        """Get the currently active model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT version, accuracy, training_samples, created_at
                FROM models 
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "version": result[0],
                    "accuracy": result[1],
                    "training_samples": result[2],
                    "created_at": result[3]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get active model: {e}")
            return None
    
    def list_datasets(self, limit: int = 50) -> List[Dict]:
        """List all datasets"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT generation_id, samples_count, created_at, hour_generated
                FROM datasets 
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            datasets = []
            for row in cursor.fetchall():
                datasets.append({
                    "generation_id": row[0],
                    "samples_count": row[1],
                    "created_at": row[2],
                    "hour_generated": row[3]
                })
            
            conn.close()
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []
    
    def list_models(self, limit: int = 20) -> List[Dict]:
        """List all models"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT version, accuracy, training_samples, created_at, is_active
                FROM models 
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    "version": row[0],
                    "accuracy": row[1],
                    "training_samples": row[2],
                    "created_at": row[3],
                    "is_active": bool(row[4])
                })
            
            conn.close()
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def store_experiment(self, experiment_id: str, run_id: str, model_version: str, 
                        accuracy: float, parameters: str) -> bool:
        """Store MLflow experiment information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiments (experiment_id, run_id, model_version, accuracy, parameters)
                VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, run_id, model_version, accuracy, parameters))
            
            conn.commit()
            conn.close()
            logger.info(f"Experiment {run_id} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store experiment: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count datasets
            cursor.execute("SELECT COUNT(*) FROM datasets")
            datasets_count = cursor.fetchone()[0]
            
            # Count samples
            cursor.execute("SELECT COUNT(*) FROM dataset_samples")
            samples_count = cursor.fetchone()[0]
            
            # Count models
            cursor.execute("SELECT COUNT(*) FROM models")
            models_count = cursor.fetchone()[0]
            
            # Count experiments
            cursor.execute("SELECT COUNT(*) FROM experiments")
            experiments_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "datasets": datasets_count,
                "samples": samples_count,
                "models": models_count,
                "experiments": experiments_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def health_check(self) -> Dict:
        """Perform database health check"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            conn.close()
            
            if result and result[0] == 1:
                stats = self.get_database_stats()
                return {
                    "status": "healthy",
                    "message": "Database is accessible and functioning",
                    "stats": stats
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Database query failed"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database health check failed: {str(e)}"
            }

def main():
    """Main database manager execution"""
    print("🗄️ Database Manager")
    print("=" * 20)
    
    db_manager = DatabaseManager()
    
    print("Choose an option:")
    print("1. Database health check")
    print("2. Show database statistics")
    print("3. List datasets")
    print("4. List models")
    print("5. Get active model")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🏥 Performing health check...")
            health = db_manager.health_check()
            print(f"Status: {health['status']}")
            print(f"Message: {health['message']}")
            if 'stats' in health:
                print(f"Stats: {health['stats']}")
                
        elif choice == "2":
            print("\n📊 Database statistics...")
            stats = db_manager.get_database_stats()
            for key, value in stats.items():
                print(f"   {key.title()}: {value}")
                
        elif choice == "3":
            print("\n📋 Listing datasets...")
            datasets = db_manager.list_datasets(10)
            for dataset in datasets:
                print(f"   ID: {dataset['generation_id']}, Samples: {dataset['samples_count']}, Created: {dataset['created_at']}")
                
        elif choice == "4":
            print("\n🤖 Listing models...")
            models = db_manager.list_models(10)
            for model in models:
                status = "ACTIVE" if model['is_active'] else "inactive"
                print(f"   Version: {model['version']}, Accuracy: {model['accuracy']:.3f}, Status: {status}")
                
        elif choice == "5":
            print("\n🎯 Getting active model...")
            model = db_manager.get_active_model()
            if model:
                print(f"   Version: {model['version']}")
                print(f"   Accuracy: {model['accuracy']:.3f}")
                print(f"   Training Samples: {model['training_samples']}")
                print(f"   Created: {model['created_at']}")
            else:
                print("   No active model found")
                
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
