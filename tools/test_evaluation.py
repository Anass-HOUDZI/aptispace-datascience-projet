import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report

processed_dir = 'data/processed'
assets_dir = 'report/assets'

xgb_model = joblib.load(os.path.join(processed_dir, 'xgb_model.pkl'))
eval_data = joblib.load(os.path.join(processed_dir, 'eval_data.pkl'))
y_test, y_pred = eval_data['y_test'], eval_data['y_pred']

print("--- RAPPORT DE PERFORMANCE (Test Set) ---")
print(classification_report(y_test, y_pred, target_names=['Non-Acheteur (0)', 'Acheteur (1)']))

# Test affichage images
for img_name in ['matrice.png', 'importance.png']:
    path = os.path.join(assets_dir, img_name)
    img = plt.imread(path)
    print(f"Image {img_name}: shape={img.shape} OK")

print("=== TEST EVALUATION OK ===")
