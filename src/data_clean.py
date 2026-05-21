import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import logging

def phase_1_ingestion(filepath: str) -> pd.DataFrame:
    """
    Phase 1: Ingestion optimisée des données avec downcasting.
    """
    logging.info("Ingestion des données...")
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
    # Lecture via le moteur C
    df = pd.read_csv(filepath, sep='\t', usecols=colonnes_utiles, dtype=dtypes_schema, engine='c')
    return df

def phase_2_wrangling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Data Wrangling (gestion MNAR, dates ISO 8601, Scaling).
    """
    logging.info("Nettoyage et encodage des données...")
    df = df.copy()
    
    # Gestion du flag MNAR pour Income
    df['Income_Missing_Flag'] = df['Income'].isna().astype('int8')
    df['Income'] = df['Income'].fillna(df['Income'].median())
    
    # Nettoyage des chaînes de caractères
    for col in ['Education', 'Marital_Status']:
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    df['Marital_Status'] = df['Marital_Status'].replace(['absurd', 'yolo', 'alone'], 'single')
    
    # Date ISO 8601
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', utc=True)
    
    # Ingénierie
    df['Age'] = 2014 - df['Year_Birth']
    df = df[df['Age'] < 100].copy()
    
    edu_map = {'basic': 0, '2n cycle': 1, 'graduation': 2, 'master': 3, 'phd': 4}
    df['Education_Encoded'] = df['Education'].map(edu_map).astype('int8')
    df = pd.get_dummies(df, columns=['Marital_Status'], drop_first=True, dtype='int8')
    
    # RobustScaler
    mnt_cols = [col for col in df.columns if col.startswith('Mnt')]
    scaler = RobustScaler()
    df[mnt_cols] = scaler.fit_transform(df[mnt_cols].astype(np.float32))
    
    return df
