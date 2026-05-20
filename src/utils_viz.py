import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

def set_custom_style(theme: str = 'light'):
    """
    Applique une charte graphique premium pour toutes les visualisations du projet.
    Met en place une typographie soignée, des grilles discrètes et des palettes harmonieuses.
    
    Parameters:
    -----------
    theme : str, default 'light'
        Thème général des graphiques : 'light' (sleek minimaliste) ou 'dark' (cyberpunk).
    """
    colors_light = ["#1A73E8", "#D93025", "#188038", "#F29900", "#70757A"]
    colors_dark = ["#8AB4F8", "#F28B82", "#81C995", "#FDD663", "#9AA0A6"]
    
    selected_colors = colors_light if theme == 'light' else colors_dark
    
    sns.set_theme(style="whitegrid" if theme == 'light' else "darkgrid")
    
    plt.rcParams.update({
        'figure.dpi': 120,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'grid.color': '#E5E7EB' if theme == 'light' else '#374151',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'legend.frameon': True,
        'legend.facecolor': '#FFFFFF' if theme == 'light' else '#1F2937',
        'legend.edgecolor': 'none',
        'legend.fontsize': 9,
        'axes.prop_cycle': plt.cycler(color=selected_colors)
    })
    
    print(f"🎨 Charte graphique '{theme}' initialisée avec succès.")

def plot_generic_trends(df: pd.DataFrame, x_col: str, y_col: str, group_col: Optional[str] = None) -> plt.Figure:
    """
    Génère un graphique linéaire d'évolution temporelle ou tendancielle de base.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée contenant les données.
    x_col : str
        Le nom de la colonne pour l'axe des abscisses (ex: date, index).
    y_col : str
        Le nom de la colonne pour l'axe des ordonnées.
    group_col : Optional[str], default None
        Optionnel : Le nom de la colonne catégorielle pour grouper et dessiner plusieurs courbes.
        
    Returns:
    --------
    plt.Figure
        La figure Matplotlib créée.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    if group_col and group_col in df.columns:
        for val, grp in df.groupby(group_col):
            ax.plot(grp[x_col], grp[y_col], label=f"{group_col}: {val}", alpha=0.8)
        ax.legend()
    else:
        ax.plot(df[x_col], df[y_col], alpha=0.8)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Évolution de {y_col} par rapport à {x_col}")
    fig.tight_layout()
    return fig

def plot_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> plt.Figure:
    """
    Génère une carte de chaleur (heatmap) montrant les corrélations de Pearson
    entre plusieurs colonnes spécifiées.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée.
    columns : List[str]
        Les colonnes numériques à inclure.
        
    Returns:
    --------
    plt.Figure
        La figure Matplotlib créée.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[columns].corr()
    
    # Masque pour masquer la partie supérieure redondante
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True, square=True)
    ax.set_title("Matrice de Corrélation")
    fig.tight_layout()
    return fig

def plot_bivariate_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None) -> plt.Figure:
    """
    Génère un nuage de points (scatter plot) bivarié de base, éventuellement coloré par une troisième variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée.
    x_col : str
        Nom de la colonne sur l'axe X.
    y_col : str
        Nom de la colonne sur l'axe Y.
    color_col : Optional[str], default None
        Optionnel : Nom de la colonne numérique ou catégorielle pour la coloration des points.
        
    Returns:
    --------
    plt.Figure
        La figure Matplotlib créée.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if color_col and color_col in df.columns:
        sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap='viridis', alpha=0.7, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.7, edgecolors='none')
        
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Nuage de points : {y_col} en fonction de {x_col}")
    fig.tight_layout()
    return fig

def plot_interactive_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, hover_data: Optional[List[str]] = None) -> go.Figure:
    """
    Génère un nuage de points interactif avec Plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée.
    x_col : str
        Nom de la colonne sur l'axe X.
    y_col : str
        Nom de la colonne sur l'axe Y.
    color_col : Optional[str], default None
        Optionnel : Nom de la colonne pour la coloration des points.
    hover_data : Optional[List[str]], default None
        Optionnel : Liste de colonnes à afficher au survol.
        
    Returns:
    --------
    go.Figure
        La figure Plotly interactive.
    """
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        hover_data=hover_data,
        title=f"Nuage de points interactif : {y_col} vs {x_col}",
        template="plotly_white"
    )
    
    fig.update_layout(
        title_font_size=16,
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
