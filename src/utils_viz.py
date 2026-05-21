import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from phik.report import plot_correlation_matrix
import pandas as pd

def set_custom_style():
    """Applique le style par défaut."""
    sns.set_theme(style="whitegrid", context="paper")

def plot_pca_projection(df: pd.DataFrame):
    """
    Projette les caractéristiques financières via PCA et affiche le scatter plot.
    """
    mnt_cols = [col for col in df.columns if col.startswith('Mnt')]
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[mnt_cols])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Response'], 
                          cmap='coolwarm', alpha=0.8, s=50, edgecolors='w', linewidth=0.5)
    plt.title('PCA: Comportements Financiers', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Composante Principale 1', fontsize=12)
    plt.ylabel('Composante Principale 2', fontsize=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Achat (Response)', fontsize=12)
    sns.despine()
    plt.tight_layout()
    return plt

def plot_phik_matrix(df: pd.DataFrame):
    """
    Affiche la matrice de corrélation Phik pour les variables clés.
    """
    cols_of_interest = ['Education_Encoded', 'Income', 'MntWines', 'NumWebPurchases', 'NumStorePurchases', 'Age', 'Response']
    # Vérification que les colonnes existent
    cols_present = [c for c in cols_of_interest if c in df.columns]
    
    phik_matrix = df[cols_present].astype(float).phik_matrix(
        interval_cols=[c for c in ['MntWines', 'NumWebPurchases', 'NumStorePurchases', 'Age', 'Income'] if c in cols_present]
    )
    
    plot_correlation_matrix(phik_matrix.values, x_labels=phik_matrix.columns, y_labels=phik_matrix.index, vmin=0, vmax=1, title="Matrice Phik", fontsize_factor=1.2)
    return plt
