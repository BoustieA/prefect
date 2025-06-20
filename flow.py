#!/usr/bin/env python3
"""
Prefect Flow - Drift Check Pipeline
Runs every 30 seconds, check accuracy, triggers retrain if < 0.5
"""

import os
import time
import requests
import logging
from pathlib import Path
from prefect import flow, task
from prefect.logging import get_run_logger
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
from mlflow.exceptions import MlflowException
import sqlite3
from collections import namedtuple

from sqlalchemy import create_engine, Column, Integer, Float, select, func
from sqlalchemy.orm import declarative_base, sessionmaker
from prometheus_client import start_http_server, Counter
import threading
from dotenv import load_dotenv

from utils.utilities import send_discord_embed




# Exemple simple de compteur custom
flow_runs_counter = Counter('prefect_flow_runs_total', 'Nombre de runs de flow')

def expose_metrics():
    start_http_server(8001)  # port exposé pour Prometheus

# Lancer le serveur dans un thread séparé

# Ensuite dans ton code, incrémente le compteur quand tu veux
def run_flow():
    flow_runs_counter.inc()
    # ton code de flow ici
    time.sleep(10)


# Configure logging for Prefect flow
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Setup file logging
file_handler = logging.FileHandler(logs_dir / "prefect_flow.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add file handler to root logger
logging.getLogger().addHandler(file_handler)

# Set environment variables

load_dotenv()  # charge les variables du fichier .env dans os.environ

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

# Set environment variables for Prefect
# PYTHONIOENCODING: évite les UnicodeDecodeError sous Windows
# PREFECT_API_URL: indique au SDK où se trouve l'API Prefect
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PREFECT_API_URL", PREFECT_API_URL)
os.environ.setdefault("MLFLOW_TRACKING_URI",MLFLOW_TRACKING_URI)

DATABASE_PATH = "data/ia_continu_solution.db"
PATH_MODEL="/app/models/"
FASTAPI_API_URL="http://fastapi_app"

Base = declarative_base()

class DatasetSample(Base):
    __tablename__ = "dataset_samples"

    id = Column(Integer, primary_key=True)
    generation_id = Column(Integer, nullable=False)
    feature1 = Column(Float, nullable=False)
    feature2 = Column(Float, nullable=False)
    target = Column(Integer, nullable=False)


@task(retries=2, retry_delay_seconds=10, name="Check model accuracy")
def check_accuracy():
    logger = get_run_logger()

    # Charger le modèle sauvegardé
    try:
        model = joblib.load(PATH_MODEL + "model.pkl")
    except FileNotFoundError:
        logger.warning("Aucun modèle local trouvé.")
        send_discord_embed(f"🚨 Aucun modèle local trouvé. - Training nécessaire")
        
        return {"status": "no_model"}

    data = get_last_dataset()
    if not data:
        logger.warning("Pas de données disponibles.")
        send_discord_embed(f"🚨 Pas de données disponibles. - Impossible d'entrainer")
        
        return {"status": "no_data"}

    X = [[row.feature1, row.feature2] for row in data]
    Y = [row.target for row in data]

    if not X or not Y:
        return {"status": "no_data"}

    # Prédiction et accuracy
    Y_pred = model.predict(X)
    current_accuracy = accuracy_score(Y, Y_pred)

    # Lire dernière accuracy de MLflow
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)

    if not runs or "accuracy" not in runs[0].data.metrics:
        logger.warning("Pas d'ancienne métrique d'accuracy trouvée.")
        send_discord_embed(f"🚨 Pas d'ancienne métrique d'accuracy trouvée. - Training nécessaire")
        return {"status": "no_previous_accuracy"}

    previous_accuracy = runs[0].data.metrics["accuracy"]

    logger.info(f"Accuracy actuelle: {current_accuracy:.3f} — Précédente: {previous_accuracy:.3f}")

    threshold = 0.01

    if current_accuracy < previous_accuracy-threshold:
        logger.warning(f"Dérive détectée ! Accuracy {current_accuracy:.3f} < dernière accuracy {previous_accuracy} - seuil {threshold}")
        send_discord_embed(f"🚨 Dérive du modèle détectée! Accuracy actuelle: {current_accuracy:.3f}, dernière accuracy: {previous_accuracy} seuil {threshold} - Retraining nécessaire")
        return {"status": "drift", "accuracy": current_accuracy}
    else:
        logger.info(f"Modèle OK ! Accuracy {current_accuracy:.3f} >= dernière accuracy {previous_accuracy} - seuil {threshold}")
        send_discord_embed(f"✅ Modèle performant! Accuracy: {current_accuracy:.3f}")
        return {"status": "ok", "accuracy": current_accuracy}
    

def save_last_model_mlflow_to_joblib(model):
    #model = mlflow.sklearn.load_model("models:/my_model/latest")
    joblib.dump(model, PATH_MODEL + "model.pkl")


def get_last_dataset(db_path: str = DATABASE_PATH, table_name: str = "dataset_samples"):
    """
    Récupère le dernier batch (max generation_id) et retourne la liste de DataRow.

    Args:
        db_path (str): chemin vers la base SQLite.
        table_name (str): nom de la table.

    Returns:
        List[DataRow]: liste des lignes du dernier batch.
    """

    engine = create_engine(f"sqlite:///{DATABASE_PATH}",
    connect_args={"timeout": 30, "check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        max_generation_id = session.query(func.max(DatasetSample.generation_id)).scalar()

        if max_generation_id is None:
            return []

        results = (
            session.query(DatasetSample)
            .filter(DatasetSample.generation_id == max_generation_id)
            .all()
        )

    return results



@task(retries=2, retry_delay_seconds=10, name="Train model")
def train_model():
    logger = get_run_logger()
    logger.info("Entraînement du modèle")
    model=LogisticRegression()
    data=get_last_dataset()
    if not data:
        logger.warning("📭 Aucune donnée trouvée, tentative de génération via l'API")
        try:
            response = requests.post(FASTAPI_API_URL+":8000/generate", json={"samples": 100})
            response.raise_for_status()
            logger.info("✅ Données générées avec succès")
        except Exception as e:
            logger.error(f"❌ Échec de génération de données : {e}")
            send_discord_embed("❌ Entraînement annulé : génération de données échouée.")
            return
        time.sleep(2)
        data = get_last_dataset()

        if not data:
            logger.error("❌ Toujours aucune donnée après génération")
            send_discord_embed("❌ Entraînement annulé : aucune donnée même après génération.")
            return

    logger.info("📊 Données disponibles, entraînement du modèle")
    X=[[X.feature1,X.feature2] for X in data]
    Y=[X.target for X in data]
    with mlflow.start_run():
        model.fit(X,Y)
        Y_pred=model.predict(X)
        accuracy=accuracy_score(Y,Y_pred)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics({"accuracy": accuracy})

        #run_id = run.info.run_id
        #model_uri = f"runs:/{run_id}/model"
        #mlflow.register_model(model_uri, "my_model")

    message = f"📈 Nouveau modèle entraîné — Accuracy : {accuracy:.3f}"
    logger.info(message)
    send_discord_embed(message, "Succès")
    
    save_last_model_mlflow_to_joblib(model)


@flow
def periodic_check():
    """Main flow that runs periodic checks"""
    
    logger = get_run_logger()
    logger.info("💡 Lancement de la vérification périodique...")
    
    run_flow()



    check_future = check_accuracy()#.submit()
    result = check_future#.result()  # attend le résultat

    if result["status"] in {"no_model", "no_previous_accuracy", "drift"}:
        logger.warning(f"⚠️ Entraînement nécessaire : {result['status']}")
        train_model()
    elif result["status"] == "ok":
        logger.info(f"✅ Modèle performant (accuracy: {result['accuracy']:.3f})")
    else:
        logger.warning(f"❌ État inattendu ou données manquantes : {result['status']}")

    logger.info(f"Periodic check completed: " + result['status'])
    return result

if __name__ == "__main__":
    # Wait for services to be ready
    threading.Thread(target=expose_metrics, daemon=True).start()
    print("Waiting for services to be ready...")
    time.sleep(30)
    # Send startup notification
    send_discord_embed("Pipeline de vérification démarrée")
    
    # Start the flow with 30-second intervals
    periodic_check.serve(
        name="drift-check-every-30s",
        interval=60)
