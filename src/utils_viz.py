import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from phik.report import plot_correlation_matrix
import pandas as pd
import plotly.express as px

def set_custom_style():
    sns.set_theme(style="whitegrid", context="paper")

def plot_pca_projection(df: pd.DataFrame):
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
    cols_of_interest = ['Education_Encoded', 'Income', 'MntWines', 'NumWebPurchases', 'NumStorePurchases', 'Age', 'Response']
    cols_present = [c for c in cols_of_interest if c in df.columns]
    phik_matrix = df[cols_present].astype(float).phik_matrix(
        interval_cols=[c for c in ['MntWines', 'NumWebPurchases', 'NumStorePurchases', 'Age', 'Income'] if c in cols_present]
    )
    plot_correlation_matrix(phik_matrix.values, x_labels=phik_matrix.columns, y_labels=phik_matrix.index,
                            vmin=0, vmax=1, title="Matrice Phik", fontsize_factor=1.2)
    return plt

def plot_bivariate_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str):
    """Scatter Matplotlib (fallback PDF/Typst)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in df.groupby(color_col):
        ax.scatter(group[x_col], group[y_col], label=label, alpha=0.7, s=40, edgecolors='w', linewidth=0.4)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(title=color_col)
    sns.despine()
    plt.tight_layout()
    return fig

def plot_interactive_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str):
    """Scatter Plotly interactif (HTML)."""
    return px.scatter(df, x=x_col, y=y_col, color=color_col,
                      template='plotly_white', opacity=0.75,
                      title=f"{y_col} en fonction de {x_col}")

def phase_3_eda_and_viz(df: pd.DataFrame):
    """EDA visuelle complète : distributions, Phik, PCA."""
    set_custom_style()
    mnt_cols = [col for col in df.columns if col.startswith('Mnt')]

    # Distributions des dépenses
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), mnt_cols[:6]):
        sns.histplot(df[col], ax=ax, kde=True, color='steelblue')
        ax.set_title(col)
    plt.suptitle('Distributions des dépenses (post-scaling)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Boxplots Response vs Income
    if 'Response' in df.columns and 'Income' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Response', y='Income', palette='coolwarm', ax=ax2)
        ax2.set_title('Revenu médian selon la conversion (Response)')
        sns.despine()
        plt.tight_layout()
        plt.show()

    plot_phik_matrix(df)
    plt.show()
    plot_pca_projection(df)
    plt.show()
