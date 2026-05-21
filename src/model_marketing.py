import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_cnn_1d(timesteps=5, features=1) -> Model:
    """Architecture CNN 1D pour extraire les motifs séquentiels des campagnes."""
    seq_input = Input(shape=(timesteps, features), name='campaign_seq')
    x = Conv1D(filters=16, kernel_size=2, activation='relu')(seq_input)
    x = Conv1D(filters=32, kernel_size=2, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    embedding = Dense(8, activation='relu', name='temporal_embedding')(x)
    x = Dropout(0.2)(embedding)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def train_hybrid_pipeline(filepath: str):
    """Pipeline complet : Ingestion, Hybridation CNN+XGBoost, et Exports."""
    
    logging.info("Chargement du dataset...")
    df = pd.read_parquet(filepath)
    
    # Séparation des données
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    target_col = 'Response'
    exclude = set(campaign_cols + [target_col, 'ID', 'Dt_Customer', 'Education', 'Income_Strata'])
    tab_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]
    
    X_tab = df[tab_cols].values
    y = df[target_col].values
    X_seq = df[campaign_cols].values.reshape(-1, len(campaign_cols), 1)
    
    # Split
    X_tab_train, X_tab_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
        X_tab, X_seq, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # CNN 1D
    cnn = build_cnn_1d()
    cnn.fit(X_seq_train, y_train, epochs=15, batch_size=32, verbose=0)
    extractor = Model(inputs=cnn.input, outputs=cnn.get_layer('temporal_embedding').output)
    
    seq_features_train = extractor.predict(X_seq_train, verbose=0)
    seq_features_test = extractor.predict(X_seq_test, verbose=0)
    
    X_hybrid_train = np.hstack((X_tab_train, seq_features_train))
    X_hybrid_test = np.hstack((X_tab_test, seq_features_test))
    
    # XGBoost
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_model = XGBClassifier(scale_pos_weight=ratio, n_estimators=100, max_depth=4, eval_metric='auc')
    xgb_model.fit(X_hybrid_train, y_train)
    
    # Évaluation
    y_pred = xgb_model.predict(X_hybrid_test)
    print(classification_report(y_test, y_pred))

    # Persistance des arrays d'évaluation pour les notebooks 06 et 07
    y_proba = xgb_model.predict_proba(X_hybrid_test)[:, 1]
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    os.makedirs(processed_dir, exist_ok=True)
    joblib.dump({'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba},
                os.path.join(processed_dir, 'eval_data.pkl'))

    # --- SAUVEGARDE DES GRAPHIQUES ---
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'report', 'assets'))
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Matrice de confusion
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion (Test Set)')
    plt.savefig(os.path.join(assets_dir, 'matrice.png'), dpi=300)
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 6))
    feat_names = tab_cols + [f'Motif_Temporel_{i}' for i in range(8)]
    imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': xgb_model.feature_importances_})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='magma')
    plt.title("Top 10 Facteurs d'Achat")
    plt.savefig(os.path.join(assets_dir, 'importance.png'), dpi=300)
    
    logging.info(f"Graphiques sauvegardés dans {assets_dir}")
    return xgb_model, cnn

if __name__ == "__main__":
    # Remplace par le chemin réel de ton fichier parquet nettoyé
    train_hybrid_pipeline("../data/processed/marketing_clean.parquet")