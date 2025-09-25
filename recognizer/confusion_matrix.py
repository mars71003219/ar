import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

with open("/workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-l_falldown_aihub_stable/results.pkl", "rb") as f:
    results = pickle.load(f)

# y_true, y_pred, y_score 생성
y_true = [int(item["gt_label"].item()) for item in results]
y_pred = [int(item["pred_label"].item()) for item in results]
y_score = [float(item["pred_score"][1].item()) for item in results]  # positive class 확률

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report (Accuracy, Precision, Recall, F1)
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

# AUROC
try:
    auc = roc_auc_score(y_true, y_score)
    print(f"\nAUROC: {auc:.4f}")
except ValueError:
    print("\nAUROC 계산 불가 (positive/negative 클래스 불균형)")
