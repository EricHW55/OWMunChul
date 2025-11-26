# train_xgb_win.py
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import joblib


# -----------------------------
# 1. 데이터 로드
# -----------------------------
DATA_PATH = "ow_stats_features.csv"

df = pd.read_csv(DATA_PATH)

# 타깃 / 피처 분리
target_col = "win"
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

print("전체 데이터:", X.shape, "샘플,", X.shape[1], "피쳐")

# -----------------------------
# 2. train / test 분할
# -----------------------------
# test set은 완전히 홀드아웃
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # win 비율 유지
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# -----------------------------
# 3. XGBoost 분류기 + 하이퍼파라미터 탐색
# -----------------------------
base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",  # 분류에서는 logloss/auc 주로 씀
    tree_method="hist",     # GPU 있으면 'gpu_hist' 로 변경
    n_jobs=-1,
    random_state=42,
)

param_dist = {
    "n_estimators":     [600, 700, 800],
    "max_depth":        [8, 9, 10, 13, 15, 17],
    "learning_rate":    [0.04, 0.05, 0.06],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda":       [1.0, 3.0, 5.0, 10.0],
    "min_child_weight": [1, 3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=40,                        # 탐색 횟수 (데이터 양/시간 보고 조절)
    cv=cv,
    scoring="roc_auc",                # win 확률이니까 AUC 기준
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

print("\n===== 하이퍼파라미터 튜닝 시작 =====")
search.fit(X_train, y_train)

print("\n===== 튜닝 완료 =====")
print("Best params:", search.best_params_)
print("Best CV ROC-AUC:", search.best_score_)

best_model = search.best_estimator_

# -----------------------------
# 4. 테스트 성능 평가
# -----------------------------
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n===== Test 성능 =====")
print("Accuracy:", acc)
print("F1-score:", f1)
print("ROC-AUC:", auc)
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

# -----------------------------
# 5. 모델 아티팩트 저장
# -----------------------------
artifact = {
    "model": best_model,
    "feature_cols": X.columns.tolist(),
    "target_col": target_col,
    "threshold": 0.5,   # win 판정에 사용한 기준
}

joblib.dump(artifact, "xgb_win_model.joblib")
print("\nSaved model to xgb_win_model.joblib")
