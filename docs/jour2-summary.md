# Jour 2 - Summary: ML API Implementation

## 🎯 Objectifs du Jour 2

Création d'une API ML complète avec les routes suivantes :
- **Route health** : retourne ok, 200
- **Route generate** : génère un dataset basé sur des nombres aléatoires
- **Route predict** : retourne la prédiction sur le dernier dataset généré
- **Route retrain** : re-entraîne le modèle avec MLflow

## 📋 Implémentation Réalisée

### **Routes API Développées**

#### 1. Route `/health` ✅
- **Méthode** : GET
- **Réponse** : `{"status": "ok", "timestamp": "ISO_datetime", "version": "2.0.0"}`
- **Code de statut** : 200
- **Test** : Vérifie que l'endpoint retourne bien 200

#### 2. Route `/generate` ✅
- **Méthode** : POST
- **Paramètres** : `{"samples": int}` (100-10000, défaut: 1000)
- **Logique** :
  - Génère un dataset linéaire avec 2 features (feature1, feature2)
  - Modification temporelle : `feature1 = feature1 - 0.5` quand `hour % 2 == 1`
  - Crée une variable cible binaire (0 ou 1) basée sur combinaison linéaire
  - Stocke en base SQLite avec generation_id et timestamp
- **Réponse** : `{"generation_id": int, "samples_created": int, "timestamp": "ISO_datetime"}`
- **Tests** : Base de données testée, validation des paramètres

#### 3. Route `/predict` ✅
- **Méthode** : POST
- **Paramètres** : `{"features": [float, float]}` (exactement 2 features)
- **Logique** : Charge le dernier modèle entraîné, retourne prédiction binaire
- **Réponse** : `{"prediction": int, "model_version": str, "confidence": float, "timestamp": "ISO_datetime"}`
- **Tests** : Vérifie que la prédiction retourne bien 0 ou 1

#### 4. Route `/retrain` ✅
- **Méthode** : POST
- **Logique** :
  - Récupère le dernier dataset avec la target depuis la base
  - Re-entraîne un modèle de régression logistique
  - Sauvegarde avec versioning automatique
  - Intégration MLflow complète
- **Réponse** : `{"status": "success", "model_version": str, "training_samples": int, "accuracy": float, "timestamp": "ISO_datetime"}`

### **Intégration MLflow** ✅

#### Configuration
- **Tracking URI** : http://localhost:5000
- **Experiment** : "ia_continu_solution"
- **Backend** : SQLite (`sqlite:///mlflow/mlflow.db`)

#### Logging dans `/retrain`
- **Hyperparamètres** : C, solver, max_iter, training_samples
- **Métriques** : accuracy, precision, recall, f1_score
- **Modèle** : Sauvegarde avec nom "ia_continu_logistic_regression"
- **Run naming** : `retrain_YYYYMMDD_HHMMSS`

### **Base de Données SQLite** ✅

#### Tables Créées
1. **datasets** : Métadonnées des générations
   - id, generation_id, samples_count, created_at, hour_generated

2. **dataset_samples** : Échantillons individuels
   - id, generation_id, feature1, feature2, target

3. **models** : Versions des modèles
   - id, version, accuracy, training_samples, created_at, is_active

#### Stockage
- **Fichier** : `data/ia_continu_solution.db`
- **Volume Docker** : `app_data:/app/data`

### **Tests Unitaires** ✅

#### Tests Implémentés
- **Health endpoint** : Retourne bien 200
- **Generate endpoint** : Base de données testée
- **Predict endpoint** : Retourne bien 0 ou 1
- **Retrain endpoint** : Pipeline complet testé
- **Validation** : Tests des paramètres d'entrée
- **Intégration** : Pipeline complet generate → predict → retrain

#### Exécution des Tests
```bash
python tests.py
```

## 🐳 Configuration Docker

### **Services Déployés**
1. **FastAPI App** (Port 9000)
   - Application ML principale
   - Base SQLite intégrée
   - Volume pour persistance des données

2. **MLflow Server** (Port 5000)
   - Tracking des expériences
   - Backend SQLite
   - Interface web accessible

3. **Uptime Kuma** (Port 3001)
   - Monitoring basique HTTP
   - Interface web de surveillance

4. **Prefect Server** (Port 4200)
   - Orchestration des workflows
   - Interface de monitoring avancé

5. **Prefect Worker**
   - Exécution des flows de monitoring
   - Notifications Discord

### **Volumes Docker**
- `app_data` : Données SQLite de l'application
- `app_models` : Modèles ML sauvegardés
- `mlflow_data` : Données MLflow
- `uptime-kuma-data` : Configuration Uptime Kuma
- `prefect_data` : Données Prefect

## 🔧 Utilisation

### **Démarrage**
```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier le statut
docker-compose ps
```

### **Accès aux Services**
- **API ML** : http://localhost:9000
- **Documentation API** : http://localhost:9000/docs
- **MLflow UI** : http://localhost:5000
- **Uptime Kuma** : http://localhost:3001
- **Prefect UI** : http://localhost:4200

### **Workflow Typique**
1. **Générer un dataset** : `POST /generate`
2. **Faire une prédiction** : `POST /predict`
3. **Re-entraîner le modèle** : `POST /retrain`
4. **Vérifier dans MLflow** : Interface web port 5000

### **Tests**
```bash
# Tests complets
python tests.py

# Tests spécifiques avec pytest
pytest tests.py -v
```

## 📊 Monitoring

### **Surveillance Automatique**
- **Uptime Kuma** : Ping HTTP toutes les 30 secondes
- **Prefect Flows** : Workflows de monitoring avancés
- **Discord** : Notifications automatiques

### **Métriques Suivies**
- Santé de l'API (`/health`)
- Performance des prédictions
- Succès des re-entraînements
- Métriques MLflow (accuracy, precision, recall, f1)

## ✅ Résultats

### **Fonctionnalités Livrées**
- ✅ API ML complète avec 4 routes principales
- ✅ Base de données SQLite intégrée
- ✅ Intégration MLflow pour le tracking
- ✅ Tests unitaires complets
- ✅ Configuration Docker multi-services
- ✅ Monitoring avec Uptime Kuma et Prefect
- ✅ Notifications Discord

### **Architecture Technique**
- **Framework** : FastAPI
- **ML** : scikit-learn (LogisticRegression)
- **Base de données** : SQLite
- **Tracking** : MLflow
- **Orchestration** : Docker Compose
- **Monitoring** : Uptime Kuma + Prefect
- **Tests** : pytest + requests

### **Performance**
- **Port utilisé** : 9000 (comme demandé)
- **Temps de réponse** : < 1s pour health, predict
- **Génération dataset** : < 5s pour 1000 échantillons
- **Re-entraînement** : < 30s pour 500 échantillons

## 🎉 Conclusion Jour 2

L'implémentation du Jour 2 est **complète et fonctionnelle** :

1. **API ML robuste** avec toutes les routes demandées
2. **Intégration MLflow** pour le suivi des expériences
3. **Base SQLite** pour la persistance des données
4. **Tests complets** validant toutes les fonctionnalités
5. **Architecture Docker** scalable et maintenable
6. **Monitoring intégré** avec notifications Discord

Le système est prêt pour la production avec une architecture microservices complète, des tests automatisés et un monitoring avancé.
