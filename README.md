# 🧠 Analyse du Bien-Être Étudiant & Prédiction du Risque de Santé Mentale

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.7%2B-017CEE?logo=apacheairflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)

> ⚠️ **Avertissement Éthique & Cadre Pédagogique :**  
> Ce projet est réalisé dans un cadre **strictement pédagogique**. Le modèle développé est un prototype d'exploration de données et d'aide à la décision globale. **Il ne doit en aucun cas être utilisé comme un outil de diagnostic médical, de dépistage clinique individuel, de décision disciplinaire ou de sélection d'étudiants.**

---

## 📋 Contexte du Projet

Un établissement d’enseignement souhaite analyser un jeu de données portant sur ses étudiants afin d’identifier les facteurs associés au bien-être psychologique et aux risques liés à la santé mentale, puis développer un prototype de classification supervisée.

Ce projet couvre l'intégralité du cycle de vie d'un projet **Data Science / MLOps** :
1. **Audit & Analyse Exploratoire (EDA)**
2. **Nettoyage, Feature Engineering & Prétraitement (Pipelines Anti-Data Leakage)**
3. **Entraînement, Comparaison & Optimisation de Modèles**
4. **Orchestration Automatisée avec Apache Airflow**
5. **Restitution Interactive via Streamlit**

---

## 🎯 Objectifs Métier

* **Détection des risques :** Identifier les profils présentant un risque accru de difficultés psychologiques.
* **Prévention :** Anticiper les besoins d’accompagnement au sein de l'établissement.
* **Analyse des facteurs :** Comprendre les variables académiques, sociales et personnelles les plus corrélées au niveau de risque.
* **Accompagnement :** Faciliter l'orientation globale vers des ressources de soutien adaptées.
* **Optimisation :** Aider à prioriser et dimensionner les actions de soutien étudiant.

---

## 🏗️ Architecture du Projet

```text
├── airflow/
│   └── dags/
│       └── student_mental_health_dag.py   # DAG d'orchestration du pipeline ML
├── app/
│   └── app.py                             # Application interactive Streamlit
├── data/
│   ├── raw/                               # Données brutes d'origine
│   └── processed/                         # Données nettoyées (Train / Test)
├── models/
│   └── best_model_pipeline.joblib         # Pipeline complet sauvegardé (Preprocessing + Modèle)
├── notebooks/
│   ├── 01_eda_audit.ipynb                 # Exploration, statistiques descriptives & corrélations
│   └── 02_modeling_tuning.ipynb           # Expérimentations, comparaison & GridSearchCV
├── src/
│   ├── data_prep.py                       # Nettoyage, gestion des outliers et split Train/Test
│   ├── preprocessing.py                   # Pipeline Scikit-Learn / Imblearn (SMOTE, Scaler, Encoder)
│   ├── train.py                           # Entraînement et optimisation des modèles
│   └── evaluate.py                        # Évaluation sur le jeu de test et métriques
├── requirements.txt                       # Dépendances Python du projet
└── README.md
```

---

## 🚀 Méthodologie & Étapes Réalisées

### Étape 1 — Chargement et Analyse Exploratoire des Données (EDA)
* Inspection de la structure du dataset (dimensions, types de variables, aperçus).
* Audit de qualité : détection des valeurs manquantes, doublons, incohérences métier et valeurs aberrantes.
* Analyse univariée et bivariée : statistiques descriptives et étude de la distribution de la variable cible (détection du déséquilibre des classes).
* Matrices de corrélation et identification des variables les plus fortement liées au risque psychologique.

### Étape 2 — Transformation et Préparation des Données
* **Séparation des données :** Split stratifié **80 % Entraînement / 20 % Test** réalisé *avant* toute transformation pour éviter toute fuite de données (*Data Leakage*).
* **Traitement des valeurs manquantes & aberrantes :** Imputation adaptée (médiane/moyenne pour le numérique, mode/constante pour le catégoriel) et traitement des outliers (méthode IQR / Z-score) avec justification documentée.
* **Encodage & Mise à l'échelle :** `OneHotEncoder` / `OrdinalEncoder` pour les variables catégorielles et `StandardScaler` / `RobustScaler` pour les variables numériques.
* **Gestion du déséquilibre :** Application de **SMOTE** (sur-échantillonnage) uniquement sur les plis d'entraînement via `imblearn.pipeline.Pipeline`.

### Étape 3 — Entraînement et Comparaison des Modèles
* Entraînement et comparaison sur les mêmes données d'au moins 3 algorithmes de classification (ex. : *Régression Logistique, Random Forest, XGBoost / Gradient Boosting*).
* Évaluation comparative via validation croisée sur des métriques adaptées au contexte : **Recall (Rappel)** (prioritaire pour ne pas manquer les cas à risque), **F1-Score**, **Précision**, **Accuracy**, **Matrice de confusion** et **ROC-AUC**.

### Étape 4 — Optimisation, Évaluation & Sérialisation
* Recherche d'hyperparamètres (`GridSearchCV` / `RandomizedSearchCV`) sur les modèles les plus prometteurs.
* Sélection du modèle final en adéquation avec l'objectif métier (arbitrage faux positifs / faux négatifs).
* Évaluation finale sur le jeu de test indépendant, analyse des erreurs et des limites du modèle.
* Exportation du pipeline complet (`preprocessing` + `modèle`) au format `.joblib`.

### Étape 5 — Orchestration (Airflow) & Visualisation (Streamlit)
* **Apache Airflow :** Conception d'un DAG planifié automatisant la chaîne complète (`préparation` $\rightarrow$ `prétraitement` $\rightarrow$ `entraînement` $\rightarrow$ `évaluation`), avec gestion des logs, alertes et suivi d'état.
* **Streamlit :** Interface web permettant de visualiser les performances du modèle (métriques, courbe ROC, matrice de confusion, importance des variables) et de simuler des prédictions sur de nouveaux profils.

---

## 🛠️ Installation & Utilisation

### 1. Cloner le dépôt et installer les dépendances
```bash
git clone https://github.com/votre-utilisateur/student-wellbeing-ml.git
cd student-wellbeing-ml

python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Lancer l'application interactive Streamlit
```bash
streamlit run app/app.py
```
L'interface sera accessible à l'adresse `http://localhost:8501`.

### 3. Exécuter le pipeline via Apache Airflow
1. Copier le fichier `airflow/dags/student_mental_health_dag.py` dans votre dossier `AIRFLOW_HOME/dags/`.
2. Démarrer le serveur web et le scheduler Airflow :
```bash
airflow standalone
```
3. Activer et déclencher le DAG **`student_mental_health_pipeline`** depuis l'interface Airflow (`http://localhost:8080`).

---

## 🛠️ Stack Technique

* **Langage :** Python 3.9+
* **Manipulation & Analyse de données :** Pandas, NumPy, Matplotlib, Seaborn, Plotly
* **Machine Learning & Prétraitement :** Scikit-learn, Imbalanced-learn (SMOTE), XGBoost, Joblib
* **Orchestration :** Apache Airflow
* **Application Web :** Streamlit

---

## 👤 Auteur

* **Oussama Lamlouki** — Développeur Full Stack (Web & IA)  
  * GitHub : [@Oussamaycode](https://github.com/Oussamaycode)
