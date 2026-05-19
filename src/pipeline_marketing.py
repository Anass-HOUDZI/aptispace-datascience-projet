import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from phik.report import plot_correlation_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_pipeline(filepath: str, output_path: str):
    start_time = time.perf_counter()
    
    # --- PHASE 1 : INGESTION ---
    logging.info("PHASE 1 : Ingestion Optimisée & Sanity Check...")
    dtypes_schema = {
        'ID': 'int32', 'Year_Birth': 'int16', 
        'Education': 'category', 'Marital_Status': 'category',
        'Income': 'Float32', 'Kidhome': 'int8', 'Teenhome': 'int8', 'Recency': 'int16',
        'MntWines': 'int32', 'MntFruits': 'int32', 'MntMeatProducts': 'int32',
        'MntFishProducts': 'int32', 'MntSweetProducts': 'int32', 'MntGoldProds': 'int32',
        'NumDealsPurchases': 'int8', 'NumWebPurchases': 'int8', 'NumCatalogPurchases': 'int8',
        'NumStorePurchases': 'int8', 'NumWebVisitsMonth': 'int8',
        'AcceptedCmp3': 'int8', 'AcceptedCmp4': 'int8', 'AcceptedCmp5': 'int8',
        'AcceptedCmp1': 'int8', 'AcceptedCmp2': 'int8', 
        'Complain': 'int8', 'Response': 'int8'
    }
    
    colonnes_utiles = list(dtypes_schema.keys()) + ['Dt_Customer']
    df = pd.read_csv(filepath, sep='\t', usecols=colonnes_utiles, dtype=dtypes_schema, engine='c')
    print("\n--- AUDIT MÉMOIRE (Sanity Check) ---")
    df.info(memory_usage='deep')

    # --- PHASE 2 : DATA WRANGLING ---
    logging.info("PHASE 2 : Data Wrangling & Encodage...")
    df['Income_Missing_Flag'] = df['Income'].isna().astype('int8')
    df['Income'] = df['Income'].fillna(df['Income'].median())
    
    for col in ['Education', 'Marital_Status']:
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    df['Marital_Status'] = df['Marital_Status'].replace(['absurd', 'yolo', 'alone'], 'single')
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', utc=True)
    df['Age'] = 2014 - df['Year_Birth']
    df = df[df['Age'] < 100].copy()
    
    edu_map = {'basic': 0, '2n cycle': 1, 'graduation': 2, 'master': 3, 'phd': 4}
    df['Education_Encoded'] = df['Education'].map(edu_map).astype('int8')
    df = pd.get_dummies(df, columns=['Marital_Status'], drop_first=True, dtype='int8')
    
    mnt_cols = [col for col in df.columns if col.startswith('Mnt')]
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[mnt_cols] = scaler.fit_transform(df_scaled[mnt_cols].astype(np.float32))

    # --- PHASE 3 & 4 : EDA & GRAPHIQUES ---
    logging.info("PHASE 3 & 4 : Analyse Exploratoire & Visualisation...")
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled[mnt_cols])
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df_scaled['Response'], cmap='coolwarm', alpha=0.6, s=15)
    plt.title('PCA: Comportements financiers (Rouge = Achat)')
    plt.colorbar(label='Response')
    plt.show()
    
    # 2. Matrice Phik
    cols_of_interest = ['Education_Encoded', 'Income', 'MntWines', 'NumWebPurchases', 'NumStorePurchases', 'Age', 'Response']
    phik_matrix = df[cols_of_interest].astype(float).phik_matrix(interval_cols=['MntWines', 'NumWebPurchases', 'NumStorePurchases', 'Age', 'Income'])
    plot_correlation_matrix(phik_matrix.values, x_labels=phik_matrix.columns, y_labels=phik_matrix.index, vmin=0, vmax=1, title="Matrice Phik")
    plt.show()

    # 3. Segmentation Cythonisée
    df['Income_Strata'] = pd.qcut(df['Income'], q=4, labels=['Pauvre', 'Moyen-Bas', 'Moyen-Haut', 'Riche'])
    kpi = df.groupby('Income_Strata', observed=False).agg(
        Total_Wines_Median=('MntWines', 'median'),
        Total_Meat_Median=('MntMeatProducts', 'median'),
        Conversion_Rate=('Response', 'mean')
    )
    print("\n--- SYNTHÈSE DE SEGMENTATION ---")
    print(kpi)

    # --- PHASE 5 : EXPORT PARQUET ---
    df.to_parquet(output_path, engine="pyarrow")
    logging.info(f"Export terminé : {output_path} généré.")
    logging.info(f"PIPELINE TERMINÉ EN {time.perf_counter() - start_time:.3f} SECONDES.")

if __name__ == "__main__":
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    in_path = base_dir / "data" / "raw" / "marketing_campaign.csv"
    out_path = base_dir / "data" / "processed" / "marketing_clean.parquet"
    process_pipeline(str(in_path), str(out_path))
