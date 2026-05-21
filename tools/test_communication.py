import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, 'src')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

processed_dir = 'data/processed'
xgb_model = joblib.load(os.path.join(processed_dir, 'xgb_model.pkl'))
eval_data  = joblib.load(os.path.join(processed_dir, 'eval_data.pkl'))
y_test = eval_data['y_test']
y_pred = eval_data['y_pred']

# KPIs
report = classification_report(y_test, y_pred, output_dict=True)
auc    = roc_auc_score(y_test, y_pred)
print("=== KPIs ===")
print(f"  Recall acheteurs  : {report['1']['recall']:.0%}")
print(f"  Précision         : {report['1']['precision']:.0%}")
print(f"  F1-Score          : {report['1']['f1-score']:.0%}")
print(f"  AUC-ROC           : {auc:.2f}")
print(f"  Accuracy          : {report['accuracy']:.0%}")

# Feature importance
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
exclude = set(campaign_cols + ['Response', 'ID', 'Dt_Customer', 'Education', 'Income_Strata'])
df_raw  = pd.read_parquet(os.path.join(processed_dir, 'marketing_clean.parquet'))
tab_cols = [c for c in df_raw.select_dtypes(include='number').columns if c not in exclude]
feat_names = tab_cols + [f'Motif_Temporel_{i+1}' for i in range(8)]
imp_df = (pd.DataFrame({'Feature': feat_names, 'Importance': xgb_model.feature_importances_})
          .sort_values('Importance', ascending=False).head(10))
print("\n=== TOP 10 FEATURES ===")
print(imp_df.to_string(index=False))

# Gains cumulés — 30% target
proba = eval_data['y_proba']
order  = np.argsort(-proba)
gains  = np.cumsum(y_test[order]) / y_test.sum()
idx30  = int(0.30 * len(y_test))
print(f"\n=== IMPACT METIER ===")
print(f"  30% clients contactes -> {gains[idx30]:.0%} acheteurs captures")
print("\n=== TEST COMMUNICATION OK ===")
