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
    
    # 1. INGESTION HAUTE PERFORMANCE (Fichier Parquet)
    logging.info(f"Chargement du fichier {input_path} en RAM...")
    df = pd.read_parquet(input_path)
    
    # 2. PRÉPARATION DES VECTEURS D'ENTRAÎNEMENT (X et y)
    # On détruit les variables qui ne sont pas des Features prédictives
    cols_to_drop = ['ID', 'Dt_Customer', 'Year_Birth', 'Response', 'Income_Strata', 'Education']
    X = df.drop(columns=cols_to_drop)
    y = df['Response'].astype('int8') # Target
    
    # 3. SPLIT STRATIFIÉ (80% Train / 20% Test)
    # 'stratify=y' garantit qu'il y aura bien 15% de "Oui" dans le Train ET dans le Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logging.info(f"Dimensions Train: {X_train.shape} | Dimensions Test: {X_test.shape}")

    # 4. ENTRAÎNEMENT DE L'ALGORITHME (Multi-threading CPU)
    logging.info("Entraînement du modèle RandomForest (class_weight='balanced')...")
    rf_model = RandomForestClassifier(
        n_estimators=300,        # Nombre d'arbres
        max_depth=10,            # Profondeur max pour éviter l'Overfitting
        class_weight='balanced', # Pénalise les erreurs sur la classe minoritaire (Oui)
        n_jobs=-1,               # Utilise tous les cœurs du CPU
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # 5. ÉVALUATION ET DIAGNOSTIC MÉTIER
    logging.info("Inférence sur le set de Test et génération du rapport...")
    y_pred = rf_model.predict(X_test)
    
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION MÉTIER")
    print("="*50)
    # Focus sur le F1-Score, le Recall et la Precision
    print(classification_report(y_test, y_pred))
    
    # 6. MATRICE DE CONFUSION
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matrice de Confusion (Test Set)')
    plt.xlabel('Prédiction Algorithmique')
    plt.ylabel('Réalité Métier (Ground Truth)')
    plt.show()
    
    # 7. FEATURE IMPORTANCE (L'explicabilité pour le métier)
    # Quelles sont les variables qui ont réellement déclenché l'achat ?
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_10 = importances.nlargest(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10.values, y=top_10.index, hue=top_10.index, palette='magma', legend=False)
    plt.title('Top 10 des Facteurs Déclencheurs d\'Achat (Feature Importance)')
    plt.xlabel('Poids dans la décision de l\'algorithme')
    plt.tight_layout()
    plt.show()
    
    logging.info(f"PROCESSUS MACHINE LEARNING TERMINÉ EN {time.perf_counter() - start_time:.3f} SECONDES.")

if __name__ == "__main__":
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    in_path = base_dir / "data" / "processed" / "marketing_clean.parquet"
    train_and_evaluate_model(str(in_path))
