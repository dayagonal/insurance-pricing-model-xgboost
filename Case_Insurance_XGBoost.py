#!/usr/bin/env python
# coding: utf-8

# In[5]:


# =============================================================================
# Freedom Insurance Case (XGBoost)


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import brentq
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, r2_score, roc_curve, mean_squared_error, mean_absolute_error
from sklearn.metrics import auc as sk_auc  # переименовали, чтобы избежать конфликта
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

# Для VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# =============================================================================
# 1. Параметры и настройки
# =============================================================================
TARGET_LR = 0.70
GROUP1_THRESHOLD = 1.05          # допустимо повышение до 5% (увеличит долю группы 1)
RANDOM_STATE = 42
N_FOLDS = 5
OUTPUT_DIR = Path("output_final")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 2. Загрузка данных
# =============================================================================
print("=" * 60)
print("1. Загрузка данных")
train = pd.read_csv("train.csv", low_memory=False)
test  = pd.read_csv("test_final.csv", low_memory=False)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# =============================================================================
# 3. Очистка "грязных" признаков
# =============================================================================
print("\n2. Очистка признаков")

def clean_experience_year(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return np.nan
    if v < 0:
        return np.nan
    if 1900 <= v <= 2026:
        return max(0, 2026 - int(v))
    if v > 70:
        return np.nan
    return v

def clean_car_year(val):
    try:
        s = str(val).strip().replace(" ", "")
        v = int(float(s))
    except (ValueError, TypeError):
        return np.nan
    if 1970 <= v <= 2026:
        return v
    if v < 100:
        c2 = 2000 + v
        c1 = 1900 + v
        if 1990 <= c2 <= 2026:
            return c2
        if 1970 <= c1 <= 2026:
            return c1
    return np.nan

def clean_bonus_malus(val):
    s = str(val).strip()
    if s in ("", "nan", "None", "0"):
        return 3
    if s.upper() == "M":
        return 0
    try:
        v = int(s)
        return v if 1 <= v <= 13 else 3
    except ValueError:
        return 3

for df in [train, test]:
    df["experience_year"]   = df["experience_year"].apply(clean_experience_year)
    df["car_year"]          = df["car_year"].apply(clean_car_year)
    df["bonus_malus_clean"] = df["bonus_malus"].apply(clean_bonus_malus)

print("   Очистка выполнена.")

# =============================================================================
# 4. Целевая переменная (для train)
# =============================================================================
train["claim_amount"] = pd.to_numeric(train["claim_amount"], errors="coerce").fillna(0)
train["claim_cnt"]    = pd.to_numeric(train["claim_cnt"], errors="coerce").fillna(0)
if "is_claim" not in train.columns or train["is_claim"].isna().all():
    train["is_claim"] = (train["claim_amount"] > 0).astype(int)
else:
    train["is_claim"] = train["is_claim"].fillna(0).astype(int)

CLAIM_RATE = train["is_claim"].mean()
print(f"\n3. Частота выплат: {CLAIM_RATE:.3%}")

if "claim_amount" not in test.columns:
    test["claim_amount"] = np.nan
if "is_claim" not in test.columns:
    test["is_claim"] = np.nan

# =============================================================================
# 5. Feature Engineering (базовые признаки)
# =============================================================================
print("\n4. Базовый feature engineering")

def base_features(df):
    df = df.copy()
    df["operation_date"] = pd.to_datetime(df["operation_date"], errors="coerce")
    df["policy_year"]    = df["operation_date"].dt.year.fillna(2026).astype(int)
    df["policy_month"]   = df["operation_date"].dt.month.fillna(1).astype(int)
    df["policy_quarter"] = ((df["policy_month"] - 1) // 3 + 1).astype(int)

    df["car_age_calc"]   = (df["policy_year"] - df["car_year"]).clip(0, 50)

    df["bm_risk"]        = (13 - df["bonus_malus_clean"]) / 13.0
    df["bm_is_base"]     = (df["bonus_malus_clean"] == 3).astype(int)
    df["bm_is_good"]     = (df["bonus_malus_clean"] >= 8).astype(int)
    df["bm_is_bad"]      = (df["bonus_malus_clean"] <= 2).astype(int)

    df["exp_group"] = pd.cut(
        df["experience_year"],
        bins=[-1, 0, 2, 5, 10, 20, 100],
        labels=[0,1,2,3,4,5]
    ).astype("float")
    df["is_new_driver"]  = (df["experience_year"] <= 2).astype(float)

    df["engine_volume"]  = pd.to_numeric(df["engine_volume"], errors="coerce")
    df["engine_power"]   = pd.to_numeric(df["engine_power"],  errors="coerce")
    df["power_per_vol"]  = df["engine_power"] / (df["engine_volume"] + 1)
    df["high_power"]     = (df["engine_power"] > 150).astype(float)

    df["is_individual_person"] = pd.to_numeric(df["is_individual_person"], errors="coerce").fillna(1)
    df["is_residence"]         = pd.to_numeric(df["is_residence"], errors="coerce").fillna(1)
    df["car_age"]              = pd.to_numeric(df["car_age"], errors="coerce")

    df["bm_x_exp"]    = df["bm_risk"] * df["experience_year"].fillna(0)
    df["power_x_new"] = df["engine_power"].fillna(0) * df["is_new_driver"]

    return df

train = base_features(train)
test  = base_features(test)
print("   Базовые признаки созданы.")

# =============================================================================
# 6. Агрегаты по водителю (с LOO) и по полису
# =============================================================================
print("\n5. Агрегаты по водителю и полису")

# ---- Агрегаты по водителю (LOO для train) ----
driver_stats_train = (
    train.groupby("driver_iin")
         .agg(
             driver_claim_rate_global = ("is_claim", "mean"),
             driver_avg_bm_global     = ("bonus_malus_clean", "mean"),
             driver_policy_cnt_global = ("contract_number", "nunique")
         )
         .reset_index()
)

driver_sum   = train.groupby("driver_iin")["is_claim"].transform("sum")
driver_count = train.groupby("driver_iin")["is_claim"].transform("count")
train["driver_claim_rate"]  = (driver_sum - train["is_claim"]) / (driver_count - 1).clip(lower=1)
train["driver_claim_count"] = (driver_sum - train["is_claim"])

solo_mask = (driver_count == 1)
train.loc[solo_mask, "driver_claim_rate"] = CLAIM_RATE

driver_bm_sum   = train.groupby("driver_iin")["bonus_malus_clean"].transform("sum")
driver_bm_count = train.groupby("driver_iin")["bonus_malus_clean"].transform("count")
train["driver_avg_bm"] = (driver_bm_sum - train["bonus_malus_clean"]) / (driver_bm_count - 1).clip(lower=1)
train.loc[solo_mask, "driver_avg_bm"] = train["bonus_malus_clean"].mean()

# ---- Применение к тесту ----
test = test.merge(driver_stats_train, on="driver_iin", how="left")
test.rename(columns={
    "driver_claim_rate_global": "driver_claim_rate",
    "driver_avg_bm_global": "driver_avg_bm",
    "driver_policy_cnt_global": "driver_policy_cnt"
}, inplace=True)

for col in ["driver_claim_rate", "driver_avg_bm", "driver_policy_cnt"]:
    if col in test.columns:
        fill_val = train[col.replace("_global", "")].mean() if col in train.columns else CLAIM_RATE
        test[col] = test[col].fillna(fill_val)
    else:
        test[col] = CLAIM_RATE if "claim" in col else train["bonus_malus_clean"].mean()

# ---- Агрегаты по полису ----
policy_feat_agg = (
    train.groupby("contract_number", as_index=False)
         .agg(
             pol_driver_cnt  = ("driver_iin",         "nunique"),
             pol_avg_bm      = ("bonus_malus_clean",  "mean"),
             pol_max_bm_risk = ("bm_risk",            "max"),
             pol_min_exp     = ("experience_year",    "min"),
             pol_any_new     = ("is_new_driver",      "max"),
         )
)

train = train.merge(policy_feat_agg, on="contract_number", how="left")
test  = test.merge(policy_feat_agg,  on="contract_number", how="left")

for col in ["pol_driver_cnt", "pol_avg_bm", "pol_max_bm_risk", "pol_min_exp", "pol_any_new"]:
    fill_val = policy_feat_agg[col].mean()
    train[col] = train[col].fillna(fill_val)
    test[col]  = test[col].fillna(fill_val)

print("   Агрегаты добавлены.")

# =============================================================================
# 7. Создание дополнительных взаимодействий
# =============================================================================
train['region_x_bm'] = train['region_id'] * train['bm_risk']
test['region_x_bm'] = test['region_id'] * test['bm_risk']

train['vehicle_x_power'] = train['vehicle_type_id'] * train['power_per_vol']
test['vehicle_x_power'] = test['vehicle_type_id'] * test['power_per_vol']

train['drivers_x_new'] = train['pol_driver_cnt'] * train['pol_any_new']
test['drivers_x_new'] = test['pol_driver_cnt'] * test['pol_any_new']

print("   Взаимодействия добавлены.")

# =============================================================================
# 8. Агрегация SCORE-признаков (средние по группам) и удаление исходных
# =============================================================================
print("\n6. Агрегация SCORE-признаков")
score_cols = [c for c in train.columns if c.startswith("SCORE_")]

# Создаём средние по группам (например, SCORE_1_avg, SCORE_2_avg и т.д.)
score_groups = {}
for col in score_cols:
    parts = col.split('_')
    prefix = '_'.join(parts[:2])  # SCORE_1, SCORE_2 и т.д.
    if prefix not in score_groups:
        score_groups[prefix] = []
    score_groups[prefix].append(col)

for prefix, cols in score_groups.items():
    train[f'{prefix}_avg'] = train[cols].mean(axis=1)
    test[f'{prefix}_avg'] = test[cols].mean(axis=1)
    print(f"   Создан {prefix}_avg из {len(cols)} признаков")

# Удаляем исходные SCORE-колонки
train = train.drop(columns=score_cols)
test = test.drop(columns=[c for c in score_cols if c in test.columns])

print(f"   Исходные SCORE-признаки удалены. Осталось {len(train.columns)} колонок в train.")

# =============================================================================
# 9. WoE / IV анализ (для отчёта)
# =============================================================================
print("\n7. WoE / IV анализ")

def calc_iv(df, feature, target, bins=10):
    tmp = df[[feature, target]].copy().dropna()
    if len(tmp) < 100 or tmp[feature].nunique() == 1:
        return 0.0
    if tmp[feature].dtype in ["object", "category"]:
        tmp["bucket"] = tmp[feature].astype(str)
    else:
        try:
            tmp["bucket"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
        except Exception:
            tmp["bucket"] = tmp[feature].astype(str)
    te  = tmp[target].sum()
    tne = len(tmp) - te
    g   = tmp.groupby("bucket")[target].agg(["sum", "count"])
    g.columns = ["events", "count"]
    g["ne"]   = g["count"] - g["events"]
    g["pe"]   = g["events"] / (te + 1e-9)
    g["pne"]  = g["ne"]    / (tne + 1e-9)
    g["woe"]  = np.log(g["pe"] / (g["pne"] + 1e-9) + 1e-9)
    g["iv"]   = (g["pe"] - g["pne"]) * g["woe"]
    return g["iv"].sum()

iv_features = [
    "bonus_malus_clean", "bm_risk", "bm_is_base", "bm_is_good", "bm_is_bad",
    "experience_year", "exp_group", "is_new_driver",
    "car_age", "car_age_calc", "car_year",
    "engine_power", "engine_volume", "power_per_vol", "high_power",
    "is_individual_person", "is_residence",
    "region_id", "vehicle_type_id",
    "policy_month", "policy_quarter",
    "bm_x_exp", "power_x_new",
    "driver_claim_rate", "driver_avg_bm",
    "pol_driver_cnt", "pol_avg_bm", "pol_max_bm_risk", "pol_min_exp", "pol_any_new",
    "region_x_bm", "vehicle_x_power", "drivers_x_new"
] + [f"{p}_avg" for p in score_groups.keys()]  # добавляем все агрегированные SCORE

iv_results = []
for feat in iv_features:
    if feat in train.columns:
        iv = calc_iv(train, feat, "is_claim")
        iv_results.append({"feature": feat, "IV": round(iv, 4)})

iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False)
print(iv_df.head(15).to_string(index=False))
iv_df.to_csv(OUTPUT_DIR / "iv_results.csv", index=False)
print("   IV результаты сохранены.")

# =============================================================================
# 10. Подготовка финального набора признаков
# =============================================================================
feature_cols = [f for f in iv_features if f in train.columns and f not in ["region_id", "vehicle_type_id"]]
feature_cols += ["region_id", "vehicle_type_id"]

# Кодирование категориальных переменных
le_dict = {}
for col in ["region_id", "vehicle_type_id"]:
    if col in train.columns and col in test.columns:
        le = LabelEncoder()
        combined = train[col].astype(str).tolist() + test[col].astype(str).tolist()
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        le_dict[col] = le

X = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y_freq = train["is_claim"]

X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"\n8. Число признаков: {len(feature_cols)}")
print(f"   Размер X: {X.shape}, X_test: {X_test.shape}")

# =============================================================================
# 11. VIF анализ (мультиколлинеарность) - до отбора
# =============================================================================
print("\n9. VIF анализ (на выборке 50k строк для скорости)")
vif_sample = X.sample(min(50000, len(X)), random_state=RANDOM_STATE)
vif_sample_const = add_constant(vif_sample)
vif_data = pd.DataFrame()
vif_data["feature"] = vif_sample_const.columns
vif_data["VIF"] = [variance_inflation_factor(vif_sample_const.values, i) for i in range(vif_sample_const.shape[1])]
vif_data = vif_data[vif_data["feature"] != "const"].sort_values("VIF", ascending=False)
print(vif_data.head(20).to_string(index=False))
vif_data.to_csv(OUTPUT_DIR / "vif_results.csv", index=False)
print("   VIF результаты сохранены.")

# =============================================================================
# 12. Параметры XGBoost (до отбора признаков)
# =============================================================================
scale_pos_weight = (y_freq == 0).sum() / (y_freq == 1).sum()

params_xgb = {
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "learning_rate":     0.03,
    "max_depth":         5,
    "min_child_weight":  7,
    "subsample":         0.7,
    "colsample_bytree":  0.7,
    "reg_lambda":        2.0,
    "reg_alpha":         1.0,
    "scale_pos_weight":  scale_pos_weight,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbosity":         0,
}

# =============================================================================
# 13. Отбор признаков по важности (95% суммарной важности)
# =============================================================================
print("\n10. Отбор признаков по важности")

# Обучаем быструю модель на всех данных (без CV) для оценки важности
temp_model = xgb.XGBClassifier(**params_xgb, n_estimators=500)  # random_state уже в params_xgb
temp_model.fit(X, y_freq)

# Получаем важность
importances = temp_model.feature_importances_
feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
feat_imp = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)
feat_imp["cumsum"] = feat_imp["importance"].cumsum() / feat_imp["importance"].sum()

# Определяем порог: оставляем признаки, составляющие 95% суммарной важности
threshold = 0.95
selected_features = feat_imp[feat_imp["cumsum"] <= threshold]["feature"].tolist()
# Добавляем следующий признак, чтобы перешагнуть порог
if len(selected_features) < len(feat_imp):
    selected_features.append(feat_imp.iloc[len(selected_features)]["feature"])

print(f"   Отобрано {len(selected_features)} из {len(feature_cols)} признаков (порог {threshold:.0%} суммарной важности)")
print("   Топ-10 признаков по важности:")
print(feat_imp.head(10).to_string(index=False))

# Сохраняем список отобранных признаков
joblib.dump(selected_features, OUTPUT_DIR / "selected_features.pkl")

# Обновляем feature_cols, X, X_test
feature_cols = selected_features
X = X[feature_cols]
X_test = X_test[feature_cols]

print(f"   Новый размер X: {X.shape}, X_test: {X_test.shape}")

# Повторяем VIF анализ для нового набора (опционально)
print("\n11. VIF анализ после отбора признаков")
vif_sample = X.sample(min(50000, len(X)), random_state=RANDOM_STATE)
vif_sample_const = add_constant(vif_sample)
vif_data_new = pd.DataFrame()
vif_data_new["feature"] = vif_sample_const.columns
vif_data_new["VIF"] = [variance_inflation_factor(vif_sample_const.values, i) for i in range(vif_sample_const.shape[1])]
vif_data_new = vif_data_new[vif_data_new["feature"] != "const"].sort_values("VIF", ascending=False)
print(vif_data_new.head(20).to_string(index=False))
vif_data_new.to_csv(OUTPUT_DIR / "vif_results_after_selection.csv", index=False)
print("   VIF после отбора сохранён.")

# =============================================================================
# 14. Frequency Model (XGBoost) с 5-кратной CV
# =============================================================================
print("\n12. Frequency Model (5-fold CV, XGBoost)")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_prob = np.zeros(len(X))
test_prob = np.zeros(len(X_test))
auc_scores = []
best_iters = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_freq)):
    print(f"   Fold {fold+1}/{N_FOLDS}")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_freq.iloc[tr_idx], y_freq.iloc[val_idx]

    model = xgb.XGBClassifier(**params_xgb, n_estimators=2000, early_stopping_rounds=100)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    pred_val = model.predict_proba(X_val)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1] / N_FOLDS
    oof_prob[val_idx] = pred_val
    test_prob += pred_test
    auc = roc_auc_score(y_val, pred_val)
    auc_scores.append(auc)
    best_iters.append(model.best_iteration)
    print(f"      AUC = {auc:.4f}, best_iter = {model.best_iteration}")

# Усредняем тестовые предсказания
test_prob /= N_FOLDS

mean_auc = np.mean(auc_scores)
print(f"\n   CV AUC: {mean_auc:.4f} ± {np.std(auc_scores):.4f}")
print(f"   GINI:   {2*mean_auc - 1:.4f}")

# Финальная модель на всех данных
final_model = xgb.XGBClassifier(**params_xgb, n_estimators=int(np.mean(best_iters)))
final_model.fit(X, y_freq)

# Сохраняем модель
final_model.save_model(str(OUTPUT_DIR / "model_frequency.json"))

# Калибровка вероятностей
def calibrate_probs(probs, target_rate):
    logits = np.log(np.clip(probs, 1e-9, 1-1e-9) / (1 - np.clip(probs, 1e-9, 1-1e-9)))
    def mean_diff(b):
        return (1 / (1 + np.exp(-(logits + b)))).mean() - target_rate
    b_low, b_high = -20, 20
    if mean_diff(b_low) * mean_diff(b_high) > 0:
        while mean_diff(b_low) * mean_diff(b_high) > 0:
            b_low *= 2
            b_high *= 2
    bias = brentq(mean_diff, b_low, b_high)
    return 1 / (1 + np.exp(-(logits + bias)))

train["prob_claim"] = calibrate_probs(oof_prob, CLAIM_RATE)
test["prob_claim"]  = calibrate_probs(test_prob, CLAIM_RATE)

print(f"   Калибровка: средняя prob на train = {train['prob_claim'].mean():.4f} (цель {CLAIM_RATE:.4f})")

# =============================================================================
# 15. Severity Model (XGBoost)
# =============================================================================
print("\n13. Severity Model")

mask_claim = train["is_claim"] == 1
X_sev = X[mask_claim]
y_sev = np.log1p(train.loc[mask_claim, "claim_amount"])

X_sev_tr, X_sev_val, y_sev_tr, y_sev_val = train_test_split(
    X_sev, y_sev, test_size=0.2, random_state=RANDOM_STATE
)

# Параметры для регрессии (аналогичные, но objective другой)
params_sev = params_xgb.copy()
params_sev["objective"] = "reg:squarederror"
params_sev["eval_metric"] = "rmse"
del params_sev["scale_pos_weight"]  # для регрессии не нужен

model_sev = xgb.XGBRegressor(**params_sev, n_estimators=2000, early_stopping_rounds=100)
model_sev.fit(
    X_sev_tr, y_sev_tr,
    eval_set=[(X_sev_val, y_sev_val)],
    verbose=False
)

train["pred_log_sev"] = model_sev.predict(X)
test["pred_log_sev"]  = model_sev.predict(X_test)

actual_log_mean = y_sev.mean()
pred_log_mean   = train.loc[mask_claim, "pred_log_sev"].mean()
log_bias = actual_log_mean - pred_log_mean
print(f"   Log-bias коррекция: {log_bias:.4f}")

train["expected_severity"] = np.expm1(train["pred_log_sev"] + log_bias)
test["expected_severity"]  = np.expm1(test["pred_log_sev"] + log_bias)

# Нижняя граница для severity (10% от медианы)
min_sev = np.expm1(y_sev.median()) * 0.1
train["expected_severity"] = train["expected_severity"].clip(lower=min_sev)
test["expected_severity"]  = test["expected_severity"].clip(lower=min_sev)

print(f"   Минимальная ожидаемая выплата: {min_sev:,.0f} тг")
print(f"   Средняя expected_severity (train): {train['expected_severity'].mean():,.0f} тг")

# Сохраняем модель severity
model_sev.save_model(str(OUTPUT_DIR / "model_severity.json"))

# =============================================================================
# 16. Расчёт expected loss и агрегация до полиса
# =============================================================================
train["expected_loss"] = train["prob_claim"] * train["expected_severity"]
test["expected_loss"]  = test["prob_claim"]  * test["expected_severity"]

def policy_agg(df, extra_cols=None):
    agg_dict = {
        "premium": ("premium", "first"),
        "premium_wo_term": ("premium_wo_term", "first"),
        "claim_amount": ("claim_amount", "first"),
        "is_claim": ("is_claim", "max"),
        "expected_loss": ("expected_loss", "first"),
        "prob_claim": ("prob_claim", "first"),
    }
    if extra_cols:
        for col in extra_cols:
            agg_dict[col] = (col, "first")
    return df.groupby("contract_number", as_index=False).agg(**agg_dict)

train_pol = policy_agg(train)
test_pol  = policy_agg(test, extra_cols=["prob_claim", "expected_loss"])

train_pol["prem_base"] = np.where(
    train_pol["premium_wo_term"] > 0,
    train_pol["premium_wo_term"],
    train_pol["premium"]
)
test_pol["prem_base"] = np.where(
    test_pol["premium_wo_term"] > 0,
    test_pol["premium_wo_term"],
    test_pol["premium"]
)

LR_BEFORE = train_pol["claim_amount"].sum() / train_pol["prem_base"].sum()
print(f"\n14. Текущий loss ratio: {LR_BEFORE:.2%}")

# =============================================================================
# 17. Калибровка цены для достижения целевого loss ratio 70%
# =============================================================================
print("\n15. Калибровка цены")

total_claims = train_pol["claim_amount"].sum()
target_premium_sum = total_claims / TARGET_LR
print(f"   Сумма выплат: {total_claims:,.0f} тг")
print(f"   Необходимая сумма премий (при LR={TARGET_LR:.0%}): {target_premium_sum:,.0f} тг")

def apply_pricing(df, scale):
    raw = df["expected_loss"] * scale / TARGET_LR
    upper = df["premium"] * 3
    lower = 1  # снижение не более 100% -> до 1 тенге
    new_prem = raw.clip(lower=lower, upper=upper)
    ratio = new_prem / (df["premium"] + 1e-9)
    new_prem_base = (df["prem_base"] * ratio).clip(lower=1)
    lr = df["claim_amount"].sum() / (new_prem_base.sum() + 1e-9)
    return lr, new_prem, new_prem_base

scale = total_claims / (TARGET_LR * train_pol["expected_loss"].sum() + 1e-9)

for i in range(20):
    lr_cur, _, _ = apply_pricing(train_pol, scale)
    print(f"   iter {i+1:2d}: scale = {scale:.4f}, LR = {lr_cur:.2%}")
    if abs(lr_cur - TARGET_LR) < 0.001:
        break
    scale *= (lr_cur / TARGET_LR)

lr_final, train_pol["premium_new"], train_pol["prem_base_new"] = apply_pricing(train_pol, scale)
print(f"\n   Финальный scale = {scale:.4f}, LR = {lr_final:.2%}")

train_pol["group"] = np.where(train_pol["premium_new"] <= train_pol["premium"] * GROUP1_THRESHOLD, 1, 2)

lr_group1 = train_pol.loc[train_pol["group"] == 1, "claim_amount"].sum() / \
            train_pol.loc[train_pol["group"] == 1, "prem_base_new"].sum()
lr_group2 = train_pol.loc[train_pol["group"] == 2, "claim_amount"].sum() / \
            train_pol.loc[train_pol["group"] == 2, "prem_base_new"].sum()

print(f"\n   Группа 1 (≤ {GROUP1_THRESHOLD:.0%} old): доля = {len(train_pol[train_pol['group']==1])/len(train_pol):.2%}, LR = {lr_group1:.2%}")
print(f"   Группа 2 (> {GROUP1_THRESHOLD:.0%} old):    доля = {len(train_pol[train_pol['group']==2])/len(train_pol):.2%}, LR = {lr_group2:.2%}")

# =============================================================================
# 18. Применение к тесту
# =============================================================================
print("\n16. Применение к тестовым данным")

print("   Проверка premium в тесте:")
print(f"   Минимальное значение premium: {test_pol['premium'].min()}")
print(f"   Количество нулевых premium: {(test_pol['premium'] == 0).sum()}")
if (test_pol['premium'] <= 0).any():
    test_pol.loc[test_pol['premium'] <= 0, 'premium'] = 1
    print("   Нулевые/отрицательные premium заменены на 1.")

_, test_pol["premium_new"], test_pol["prem_base_new"] = apply_pricing(test_pol, scale)

test_pol["pred_loss_ratio"] = (test_pol["expected_loss"] / test_pol["prem_base_new"]).clip(0, 5)

result_test = test_pol[["contract_number", "prob_claim", "pred_loss_ratio", "premium_new"]].rename(
    columns={"prob_claim": "probability_dtp"}
)
result_test.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
print("   Результаты сохранены.")

# =============================================================================
# 19. Сохранение параметров
# =============================================================================
print("\n17. Сохранение параметров")
joblib.dump(le_dict, OUTPUT_DIR / "label_encoders.pkl")
joblib.dump(feature_cols, OUTPUT_DIR / "feature_cols.pkl")
joblib.dump({"scale": scale, "log_bias": log_bias, "min_sev": min_sev}, OUTPUT_DIR / "pricing_params.pkl")
print("   Параметры сохранены.")

# =============================================================================
# 20. Визуализация и диагностика
# =============================================================================
print("\n18. Визуализация и диагностика")
plt.style.use("seaborn-v0_8-whitegrid")
fig_dir = OUTPUT_DIR / "figures"
fig_dir.mkdir(exist_ok=True)

# 20.1 ROC-кривая
fpr, tpr, _ = roc_curve(y_freq, oof_prob)
roc_auc_val = sk_auc(fpr, tpr)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.4f})")
plt.plot([0,1],[0,1], color="navy", lw=1, linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost (OOF)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(fig_dir / "roc_curve.png", dpi=150)
plt.close()

# 20.2 Распределение вероятностей
fig, axes = plt.subplots(1,2, figsize=(14,5))
axes[0].hist(train["prob_claim"][y_freq==0], bins=50, alpha=0.5, label="No claim", density=True)
axes[0].hist(train["prob_claim"][y_freq==1], bins=50, alpha=0.5, label="Claim", density=True)
axes[0].set_xlabel("Predicted probability")
axes[0].set_ylabel("Density")
axes[0].set_title("Distribution of predicted probabilities")
axes[0].legend()

axes[1].hist(train["prob_claim"], bins=80, edgecolor="white")
axes[1].set_xlabel("Predicted probability")
axes[1].set_ylabel("Count")
axes[1].set_title("Overall distribution")
axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig(fig_dir / "prob_distribution.png", dpi=150)
plt.close()

# 20.3 Важность признаков (после отбора)
importances = final_model.feature_importances_
fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False).head(20)
plt.figure(figsize=(9,6))
colors = ["orange" if "SCORE" in f else "steelblue" for f in fi["feature"]]
plt.barh(fi["feature"], fi["importance"], color=colors[::-1], edgecolor="white")
plt.xlabel("Feature importance (gain)")
plt.title("Top 20 features - XGBoost (after selection)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(fig_dir / "feature_importance_after_selection.png", dpi=150)
plt.close()

# 20.4 IV chart
iv_plot = iv_df[iv_df["IV"] > 0].head(15).sort_values("IV")
plt.figure(figsize=(9,6))
colors_iv = ["green" if v>=0.1 else "blue" if v>=0.02 else "gray" for v in iv_plot["IV"]]
plt.barh(iv_plot["feature"], iv_plot["IV"], color=colors_iv[::-1], edgecolor="white")
plt.xlabel("Information Value (IV)")
plt.title("Top 15 features by IV")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(fig_dir / "iv_chart.png", dpi=150)
plt.close()

# 20.5 Loss ratio до/после и группы
fig, axes = plt.subplots(1,2, figsize=(14,6))
groups = ["Before", "After", "Group 1", "Group 2"]
lr_values = [LR_BEFORE*100, lr_final*100, lr_group1*100, lr_group2*100]
colors_lr = ["red", "green", "green", "blue"]
bars = axes[0].bar(groups, lr_values, color=colors_lr, edgecolor="white")
axes[0].axhline(y=70, color="black", linestyle="--", label="Target 70%")
axes[0].axhline(y=100, color="red", linestyle=":", alpha=0.5, label="Break-even 100%")
for bar, val in zip(bars, lr_values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{val:.1f}%", ha="center")
axes[0].set_ylabel("Loss ratio (%)")
axes[0].set_title("Loss ratio before and after")
axes[0].legend()

grp_counts = train_pol["group"].value_counts().sort_index()
labels_pie = [f"Group 1 (≤ {GROUP1_THRESHOLD:.0%} old)\n{grp_counts[1]:,} policies", 
              f"Group 2 (> {GROUP1_THRESHOLD:.0%} old)\n{grp_counts[2]:,} policies"]
axes[1].pie(grp_counts, labels=labels_pie, colors=["green","red"], autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor":"white"})
axes[1].set_title("Policy distribution by group")
plt.tight_layout()
plt.savefig(fig_dir / "loss_ratio.png", dpi=150)
plt.close()

# 20.6 Сравнение премий
sample = train_pol.sample(min(5000, len(train_pol)), random_state=RANDOM_STATE)
plt.figure(figsize=(8,8))
colors_scatter = ["green" if g==1 else "red" for g in sample["group"]]
plt.scatter(sample["premium"], sample["premium_new"], c=colors_scatter, alpha=0.3, s=10)
max_val = max(sample["premium"].max(), sample["premium_new"].max())
plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", label="No change")
plt.plot([0, max_val], [0, max_val*3], color="gray", linestyle=":", label="Max 3x")
plt.xlabel("Current premium")
plt.ylabel("New premium")
plt.title("Current vs New premium (sample)")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "premium_scatter.png", dpi=150)
plt.close()

# 20.7 Severity model diagnostics
fig, axes = plt.subplots(1,2, figsize=(14,5))
axes[0].scatter(y_sev, model_sev.predict(X_sev), alpha=0.3, s=5)
axes[0].plot([y_sev.min(), y_sev.max()], [y_sev.min(), y_sev.max()], "r--", lw=1)
axes[0].set_xlabel("Actual log(claim)")
axes[0].set_ylabel("Predicted log(claim)")
axes[0].set_title(f"Severity model (R² = {r2_score(y_sev, model_sev.predict(X_sev)):.4f})")

residuals = y_sev - model_sev.predict(X_sev)
axes[1].hist(residuals, bins=60, edgecolor="white")
axes[1].axvline(x=0, color="red", linestyle="--")
axes[1].set_xlabel("Residuals (log scale)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residuals distribution")
plt.tight_layout()
plt.savefig(fig_dir / "severity_diagnostics.png", dpi=150)
plt.close()

# =============================================================================
# 21. Дополнительные тесты
# =============================================================================
print("\n19. Дополнительные тесты")

# 21.1 Сравнение с логистической регрессией
print("\n19.1 Сравнение с логистической регрессией")
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
log_reg.fit(X, y_freq)
lr_proba = log_reg.predict_proba(X)[:, 1]
lr_auc = roc_auc_score(y_freq, lr_proba)
print(f"   Logistic Regression AUC: {lr_auc:.4f}")
print(f"   XGBoost AUC: {mean_auc:.4f}")
print(f"   Прирост: {mean_auc - lr_auc:.4f}")

# 21.2 Тест на стабильность при добавлении шума
print("\n19.2 Стабильность к шуму")
X_noise = X + np.random.normal(0, 0.01, X.shape)
pred_noise = final_model.predict_proba(X_noise)[:, 1]
corr_noise = np.corrcoef(oof_prob, pred_noise)[0,1]
print(f"   Корреляция с исходными после шума 1%: {corr_noise:.4f}")
if corr_noise < 0.95:
    print("   ⚠️ Модель чувствительна к малому шуму")
else:
    print("   ✅ Модель устойчива к шуму")

# 21.3 Тест на переобучение по learning curves
print("\n19.3 Learning curves (один фолд)")
X_tr_lc, X_val_lc, y_tr_lc, y_val_lc = train_test_split(X, y_freq, test_size=0.2, random_state=RANDOM_STATE)
model_lc = xgb.XGBClassifier(**params_xgb, n_estimators=500, early_stopping_rounds=100)
model_lc.fit(X_tr_lc, y_tr_lc, eval_set=[(X_val_lc, y_val_lc)], verbose=False)
results = model_lc.evals_result()
plt.figure(figsize=(8,5))
plt.plot(results['validation_0']['auc'], label='Validation AUC')
plt.xlabel('Boosting rounds')
plt.ylabel('AUC')
plt.title('Learning curve (XGBoost)')
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "learning_curve.png", dpi=150)
plt.close()
print("   График learning curve сохранён.")

# 21.4 Проверка нормальности остатков severity
print("\n19.4 Тест на нормальность остатков severity")
stat, p = stats.normaltest(residuals)
print(f"   Нормальность остатков: p-value = {p:.4f}")
if p > 0.05:
    print("   ✅ Остатки распределены нормально (p > 0.05)")
else:
    print("   ⚠️ Остатки не нормальны (p < 0.05) — возможно, нужна другая трансформация")

# =============================================================================
# 22. Итоговая сводка
# =============================================================================
print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)
print(f"Текущий loss ratio:                 {LR_BEFORE:.2%}")
print(f"Loss ratio после корректировки:      {lr_final:.2%}")
print(f"Доля полисов в группе 1 (≤ {GROUP1_THRESHOLD:.0%} old): {len(train_pol[train_pol['group']==1])/len(train_pol):.2%}")
print(f"Loss ratio в группе 1:                {lr_group1:.2%}")
print(f"Loss ratio в группе 2:                {lr_group2:.2%}")
print(f"ROC-AUC (XGBoost CV):                 {mean_auc:.4f}")
print(f"GINI:                                  {2*mean_auc-1:.4f}")
print(f"R² severity model (log scale):         {r2_score(y_sev, model_sev.predict(X_sev)):.4f}")
print("=" * 60)
print("Все файлы сохранены в папке output_final/")

# =============================================================================
# 23. Вывод итоговой таблицы
# =============================================================================
print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА (первые 20 строк)")
print("=" * 60)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 20)
print(result_test.head(20))

print("\n" + "=" * 60)
print("СТАТИСТИКА ПО ТАБЛИЦЕ")
print("=" * 60)
print(result_test.describe(include='all').round(2))

# =============================================================================
# 24. Топ-10 самых высоких значений
# =============================================================================
print("\nТоп-10 самых высоких премий:")
print(result_test.nlargest(10, 'premium_new')[['contract_number', 'premium_new']])

print("\nТоп-10 самых высоких вероятностей ДТП:")
print(result_test.nlargest(10, 'probability_dtp')[['contract_number', 'probability_dtp']])

print("\nТоп-10 самых высоких прогнозируемых коэффициентов выплат:")
print(result_test.nlargest(10, 'pred_loss_ratio')[['contract_number', 'pred_loss_ratio']])

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)


# In[7]:


# =============================================================================
# Вывод финальной таблицы
# =============================================================================
print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА (первые 20 строк)")
print("=" * 60)

# Настройка pandas для красивого вывода
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 20)

# Выводим первые 20 строк
print(result_test.head(20))

# Если хотите сохранить полную таблицу отдельно (уже сохранена, но на всякий случай)
# result_test.to_csv(OUTPUT_DIR / "result_test_full.csv", index=False)

# Краткая статистика
print("\n" + "=" * 60)
print("СТАТИСТИКА ПО ТАБЛИЦЕ")
print("=" * 60)
print(result_test.describe(include='all').round(2))


# In[1]:


train_pred = final_model.predict_proba(X)[:, 1]
train_auc = roc_auc_score(y_freq, train_pred)
print(f"Train AUC: {train_auc:.4f}")

