#!/usr/bin/env python3
"""
IA Continu Solution - Main API Service
FastAPI application with ML pipeline endpoints, MLflow integration, and Discord notifications
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sqlite3
import joblib
from datetime import datetime, timezone
import os
import logging
import requests
from pathlib import Path

from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IA Continu Solution - Day 2",
    description="ML API with complete Day 2 functionality",
    version="2.0.0"
)


Instrumentator().instrument(app).expose(app)

# Global variables
current_model = None
current_model_version = "v1.0.0"
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Database setup
DATABASE_PATH = "data/ia_continu_solution.db"

# Discord webhook configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_discord_notification(message: str, status: str = "Succès") -> bool:
    """Send notification to Discord webhook with Day 1 format"""
    if not DISCORD_WEBHOOK_URL:
        logger.info(f"Discord webhook not configured. Message: {message}")
        return False
    
    # Color mapping
    color_map = {
        "Succès": 5814783,    # Green
        "Échec": 15158332,    # Red
        "Avertissement": 16776960,  # Yellow
        "Info": 3447003       # Blue
    }
    
    color = color_map.get(status, 3447003)
    
    data = {
        "embeds": [{
            "title": "Résultats du pipeline",
            "description": message,
            "color": color,
            "fields": [{
                "name": "Status",
                "value": status,
                "inline": True
            }, {
                "name": "Timestamp",
                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "inline": True
            }],
            "footer": {
                "text": "IA Continu Solution - Day 2"
            }
        }]
    }
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=10)
        if response.status_code == 204:
            logger.info(f"✅ Discord notification sent: {message}")
            return True
        else:
            logger.warning(f"❌ Discord notification failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Discord notification error: {e}")
        return False
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, TIMESTAMP, func
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(Integer, unique=True)
    samples_count = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    hour_generated = Column(Integer)

class DatasetSample(Base):
    __tablename__ = "dataset_samples"
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(Integer)
    feature1 = Column(Float)
    feature2 = Column(Float)
    target = Column(Integer)

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, unique=True)
    accuracy = Column(Float)
    training_samples = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    is_active = Column(Boolean, default=False)

engine = create_engine(f"sqlite:///{DATABASE_PATH}", connect_args={"timeout": 30,"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_database():
    try:
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

init_database()

# Pydantic models
class GenerateRequest(BaseModel):
    samples: int = Field(default=1000, ge=100, le=10000)

class GenerateResponse(BaseModel):
    generation_id: int
    samples_created: int
    timestamp: str

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=2, max_items=2)

class PredictResponse(BaseModel):
    prediction: int
    model_version: str
    confidence: float
    timestamp: str

class RetrainResponse(BaseModel):
    status: str
    model_version: str
    training_samples: int
    accuracy: float
    timestamp: str

# Routes

@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "IA Continu Solution - Day 2 API", "version": "2.0.0"}

@app.get("/health")
def health_check():
    """Health check endpoint - returns 200 OK"""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    }

@app.post("/generate", response_model=GenerateResponse)
def generate_dataset(request: GenerateRequest):
    try:
        logger.info(f"Generating {request.samples} samples")
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        feature1 = np.random.normal(0, 1, request.samples)
        feature2 = np.random.normal(0, 1, request.samples)
        current_hour = datetime.now().hour

        if current_hour % 2 == 1:
            feature1 = feature1 - 0.5
            logger.info(f"Applied time-based modification (hour {current_hour} is odd)")

        linear_combination = 0.5 * feature1 + 0.3 * feature2
        target = (linear_combination > 0).astype(int)

        generation_id = int(datetime.now().timestamp())

        session = SessionLocal()
        try:
            # Insert dataset metadata
            dataset = Dataset(
                generation_id=generation_id,
                samples_count=request.samples,
                hour_generated=current_hour
            )
            session.add(dataset)
            session.flush()  # flush to assign id if needed

            # Insert samples in bulk for better performance
            samples = [
                DatasetSample(
                    generation_id=generation_id,
                    feature1=float(feature1[i]),
                    feature2=float(feature2[i]),
                    target=int(target[i])
                )
                for i in range(request.samples)
            ]
            session.bulk_save_objects(samples)
            session.commit()

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"Generated dataset with ID: {generation_id}")

        return GenerateResponse(
            generation_id=generation_id,
            samples_created=request.samples,
            timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset generation failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Make predictions using the latest trained model"""
    global current_model, current_model_version
    
    try:
        if current_model is None:
            # Train a simple model if none exists
            train_default_model()
        
        # Make prediction
        features = np.array(request.features).reshape(1, -1)
        prediction = current_model.predict(features)[0]
        
        # Get prediction probability for confidence
        if hasattr(current_model, 'predict_proba'):
            probabilities = current_model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.8  # Default confidence
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
        
        return PredictResponse(
            prediction=int(prediction),
            model_version=current_model_version,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
def get_model_info():
    """Get current model information"""
    global current_model, current_model_version

    return {
        "model_version": current_model_version,
        "model_loaded": current_model is not None,
        "model_type": "LogisticRegression" if current_model else None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/datasets/list")
def list_datasets():
    session = SessionLocal()
    try:
        datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).all()
        result = [{
            "generation_id": d.generation_id,
            "samples_count": d.samples_count,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "hour_generated": d.hour_generated
        } for d in datasets]

    finally:
        session.close()

    return {
        "datasets": result,
        "total_datasets": len(result)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
