import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

sys.path.insert(0, 'src')
import matplotlib
matplotlib.use('Agg')

from model_marketing import train_hybrid_pipeline
import joblib

print("=== DEBUT ENTRAINEMENT ===")
xgb, cnn = train_hybrid_pipeline('data/processed/marketing_clean.parquet')
joblib.dump(xgb, 'data/processed/xgb_model.pkl')
cnn.save('data/processed/cnn_extractor.keras')
print("=== ARTEFACTS SAUVEGARDES ===")

artefacts = [
    'data/processed/xgb_model.pkl',
    'data/processed/cnn_extractor.keras',
    'data/processed/eval_data.pkl',
    'report/assets/matrice.png',
    'report/assets/importance.png',
]
for f in artefacts:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = "OK" if exists else "MANQUANT"
    print(f"  {status}: {f} ({size} bytes)")
