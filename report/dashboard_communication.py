"""
Dashboard de communication interactif — Projet Marketing.
Usage  : python report/dashboard_communication.py
Ouvre  : http://127.0.0.1:8050/
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc

BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC   = os.path.join(BASE, "data", "processed")

eval_data = joblib.load(os.path.join(PROC, "eval_data.pkl"))
xgb_model = joblib.load(os.path.join(PROC, "xgb_model.pkl"))
df_raw    = pd.read_parquet(os.path.join(PROC, "marketing_clean.parquet"))

y_test  = eval_data["y_test"]
y_pred  = eval_data["y_pred"]
y_proba = eval_data["y_proba"]

report  = classification_report(y_test, y_pred, output_dict=True)
auc_val = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)
recall_1    = report["1"]["recall"]
precision_1 = report["1"]["precision"]
f1_1        = report["1"]["f1-score"]
accuracy    = report["accuracy"]
cm = confusion_matrix(y_test, y_pred)

campaign_cols = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"]
exclude = set(campaign_cols + ["Response","ID","Dt_Customer","Education","Income_Strata"])
tab_cols = [c for c in df_raw.select_dtypes(include="number").columns if c not in exclude]
feat_names = tab_cols + [f"Motif_Temporel_{i+1}" for i in range(8)]
imp_df = (pd.DataFrame({"Feature": feat_names, "Importance": xgb_model.feature_importances_})
          .sort_values("Importance", ascending=False).head(10))

order    = np.argsort(-y_proba)
y_sorted = y_test[order]
gains    = np.cumsum(y_sorted) / y_sorted.sum()
pct_ctc  = np.linspace(0, 100, len(y_test))
baseline = pct_ctc / 100
idx30    = int(0.30 * len(y_test))
gain30   = gains[idx30] * 100

C = {"bd":"#1e40af","bm":"#2563eb","bl":"#0ea5e9","gr":"#16a34a",
     "or":"#f97316","pu":"#a855f7","sl":"#64748b","re":"#dc2626"}

def fig_kpi_tiles():
    kpis = [
        ("Recall",    f"{recall_1:.0%}",    "Acheteurs detectes",       C["bm"]),
        ("Precision", f"{precision_1:.0%}", "Alertes fiables",           C["gr"]),
        ("F1-Score",  f"{f1_1:.0%}",        "Equilibre global",          C["pu"]),
        ("AUC-ROC",   f"{auc_val:.2f}",     "Robustesse au seuil",       C["or"]),
        ("Accuracy",  f"{accuracy:.0%}",    "Globale (test stratifie)",  C["sl"]),
    ]
    n = len(kpis)
    fig = go.Figure()
    for i, (name, val, label, color) in enumerate(kpis):
        x0, x1 = i / n, (i + 1) / n
        raw = float(val.strip("%").replace(",", "."))
        suffix = "%" if "%" in val else ""
        fig.add_trace(go.Indicator(
            mode="number", value=raw,
            number={"suffix": suffix, "font": {"size": 40, "color": "white"}},
            title={"text": f"<b>{name}</b><br><span style='font-size:11px'>{label}</span>",
                   "font": {"size": 13, "color": "white"}},
            domain={"x": [x0 + 0.01, x1 - 0.01], "y": [0, 1]},
        ))
    fig.update_layout(height=160, margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor=C["bd"], paper_bgcolor=C["bd"])
    return fig

def fig_cumgain():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct_ctc, y=gains*100, mode="lines", name="Modele hybride",
        line=dict(color=C["bm"], width=3), fill="tonexty", fillcolor="rgba(37,99,235,0.10)"))
    fig.add_trace(go.Scatter(x=pct_ctc, y=baseline*100, mode="lines", name="Ciblage aleatoire",
        line=dict(color=C["sl"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[30], y=[gain30], mode="markers+text",
        text=[f"  30% clients => {gain30:.0f}% acheteurs captures"],
        textposition="middle right", marker=dict(color=C["or"], size=12), showlegend=False))
    fig.update_layout(title="Courbe de Gains Cumules - Efficacite du Ciblage",
        xaxis_title="% de clients contactes (tries par score decroissant)",
        yaxis_title="% d'acheteurs reels captures",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380, margin=dict(t=70, b=50), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(range=[0,100], gridcolor="#f1f5f9")
    fig.update_yaxes(range=[0,105], gridcolor="#f1f5f9")
    return fig

def fig_importance():
    colors = [C["bd"] if "Motif" in f else C["bl"] for f in imp_df["Feature"]]
    fig = go.Figure(go.Bar(
        x=imp_df["Importance"][::-1].values, y=imp_df["Feature"][::-1].values,
        orientation="h", marker_color=colors[::-1],
        text=[f"{v:.3f}" for v in imp_df["Importance"][::-1]], textposition="outside"))
    fig.update_layout(title="Top 10 des Facteurs Predictifs d'Achat (Gain XGBoost)",
        xaxis_title="Importance (gain)", height=380,
        margin=dict(t=60, b=40, l=180, r=60), plot_bgcolor="white", paper_bgcolor="white")
    return fig

def fig_confusion():
    z = cm[::-1]
    labels = ["Non-acheteur (0)", "Acheteur (1)"]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels[::-1],
        text=[[str(v) for v in row] for row in z], texttemplate="%{text}",
        textfont={"size": 24, "color": "white"},
        colorscale=[[0, "#eff6ff"], [1, C["bd"]]], showscale=False))
    fig.update_layout(title="Matrice de Confusion (Test - n=448)",
        xaxis_title="Prediction", yaxis_title="Realite", height=340, margin=dict(t=60, b=60))
    return fig

def fig_roc():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
        name=f"Modele hybride (AUC={auc_val:.2f})",
        line=dict(color=C["bm"], width=3), fill="tonexty", fillcolor="rgba(37,99,235,0.10)"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatoire (AUC=0.50)",
        line=dict(color=C["sl"], dash="dash", width=2)))
    fig.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="Recall (TPR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=340, margin=dict(t=70, b=50), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(range=[0,1], gridcolor="#f1f5f9")
    fig.update_yaxes(range=[0,1.02], gridcolor="#f1f5f9")
    return fig

def fig_score_distribution():
    df_s = pd.DataFrame({"score": y_proba,
        "classe": ["Acheteur" if y==1 else "Non-acheteur" for y in y_test]})
    fig = px.histogram(df_s, x="score", color="classe", nbins=40, barmode="overlay",
        color_discrete_map={"Acheteur": C["bd"], "Non-acheteur": C["bl"]}, opacity=0.7,
        labels={"score": "Score de Propension"}, title="Distribution des Scores par Classe")
    fig.add_vline(x=0.5, line_dash="dash", line_color=C["or"],
        annotation_text="Seuil 0.5", annotation_position="top right")
    fig.update_layout(height=320, margin=dict(t=60, b=40), plot_bgcolor="white", paper_bgcolor="white")
    return fig

def fig_revenue():
    if "Income" not in df_raw.columns:
        return go.Figure()
    df_p = df_raw[["Income","Response"]].copy()
    df_p["Classe"] = df_p["Response"].map({0: "Non-acheteur", 1: "Acheteur"})
    fig = px.violin(df_p, y="Income", x="Classe", color="Classe", box=True, points=False,
        color_discrete_map={"Acheteur": C["bd"], "Non-acheteur": C["bl"]},
        title="Distribution du Revenu par Classe", labels={"Income": "Revenu annuel"})
    fig.update_layout(height=320, margin=dict(t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                title="Dashboard Marketing Campaign")

app.layout = dbc.Container([
    dbc.Navbar(dbc.Container([html.Div([
        html.Span("Optimisation des Campagnes Marketing",
                  className="navbar-brand mb-0 h1 fw-bold text-white"),
        html.Br(),
        html.Small("Pipeline CNN 1D + XGBoost  |  Anass HOUDZI & Baptiste MAES",
                   className="text-white-50"),
    ])], fluid=True), color="dark", dark=True, className="mb-3"),

    dbc.Card([
        dbc.CardHeader(html.H5("Indicateurs Cles de Performance", className="mb-0")),
        dbc.CardBody(dcc.Graph(id="kpi-tiles", figure=fig_kpi_tiles(),
                               config={"displayModeBar": False})),
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader(html.H5("Visualisations Analytiques", className="mb-0")),
        dbc.CardBody([
            dcc.Tabs(id="viz-tabs", value="gains", children=[
                dcc.Tab(label="Courbe de Gains",   value="gains"),
                dcc.Tab(label="Facteurs d'Achat",  value="importance"),
                dcc.Tab(label="Matrice Confusion", value="confusion"),
                dcc.Tab(label="Courbe ROC",        value="roc"),
                dcc.Tab(label="Scores par Classe", value="scores"),
                dcc.Tab(label="Revenu vs Reponse", value="revenue"),
            ]),
            html.Div(id="viz-content", className="mt-3"),
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("Actions CRM", className="mb-0 text-white"),
                           style={"background": C["bd"]}),
            dbc.CardBody(html.Ol([
                html.Li("Scorer la base -> segment prioritaire (score > 0.5)"),
                html.Li("Activer les leviers : MntWines · Income · Motifs CNN"),
                html.Li("A/B test 50/50 pour valider l'impact reel"),
            ], className="small")),
        ]), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("Limites", className="mb-0 text-white"),
                           style={"background": C["or"]}),
            dbc.CardBody(html.Ul([
                html.Li("Recall 55% -> 45% des acheteurs manques"),
                html.Li("Donnees 2012-2014 -> risque de derive"),
                html.Li("Correlation != causalite (valider par A/B test)"),
            ], className="small")),
        ]), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("Axes d'Amelioration", className="mb-0 text-white"),
                           style={"background": C["gr"]}),
            dbc.CardBody(html.Ul([
                html.Li("Court terme : tuning seuil + SHAP"),
                html.Li("Moyen terme : MLOps + features digitales"),
                html.Li("Long terme : LightGBM/TabNet + API CRM"),
            ], className="small")),
        ]), md=4),
    ], className="mb-3"),
], fluid=True)

@callback(Output("viz-content", "children"), Input("viz-tabs", "value"))
def render_tab(tab):
    fig_map = {"gains": fig_cumgain, "importance": fig_importance,
               "confusion": fig_confusion, "roc": fig_roc,
               "scores": fig_score_distribution, "revenue": fig_revenue}
    return dcc.Graph(figure=fig_map[tab](), config={"displayModeBar": True})

if __name__ == "__main__":
    print("Dashboard disponible sur : http://127.0.0.1:8050/")
    app.run(debug=False, host="127.0.0.1", port=8050)
