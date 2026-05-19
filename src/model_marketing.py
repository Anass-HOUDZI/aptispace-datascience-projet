import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate_model(input_path: str):
    start_time = time.perf_counter()
    sns.set_theme(style="whitegrid", context="paper")
    
    logging.info(f"Chargement des données depuis {input_path}...")
    df = pd.read_parquet(input_path)
    
    cols_to_drop = ['ID', 'Dt_Customer', 'Year_Birth', 'Response', 'Income_Strata', 'Education']
    X = df.drop(columns=cols_to_drop)
    y = df['Response'].astype('int8')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logging.info(f"Dimensions Train: {X_train.shape} | Test: {X_test.shape}")

    logging.info("Entraînement du modèle RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    logging.info("Évaluation du modèle...")
    y_pred = rf_model.predict(X_test)
    
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Réalité', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_10 = importances.nlargest(10)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_10.values, y=top_10.index, hue=top_10.index, palette='mako', legend=False)
    plt.title('Importance des Variables (Top 10)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.show()
    
    logging.info(f"Processus terminé en {time.perf_counter() - start_time:.3f} secondes.")

if __name__ == "__main__":
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    in_path = base_dir / "data" / "processed" / "marketing_clean.parquet"
    train_and_evaluate_model(str(in_path))
