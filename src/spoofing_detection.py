import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import optuna


#######################################################################
#
#  This code is based on an XGBoost algorithm to determine if the
#  the gnss satellites' signal is spoofed or not. It is performing the
#  check on a raw obs data converted into rinex, then into csv format.
#
#  Below is the code of the detection modell.
#
#######################################################################


# ============================================================
# 1. Loading the data
# ============================================================
log_files = ["obs_Oct21log7_spoofed.csv", "obs_Oct27log3_spoofed.csv", "obs_Oct21log8_spoofed.csv", "obs_Oct27log1_spoofed.csv"]
dfs = [pd.read_csv(f) for f in log_files]
df = pd.concat(dfs, ignore_index=True)
 
# Time conversion
df["time"] = pd.to_datetime(df["time"])

# Clipping anomalies/too large values
df['doppler_vs_prrate'] = df['doppler_vs_prrate'].clip(-2000, 2000)

# Add a few extra parameters that helps classification
def add_extra_features(df):
    df['doppler_prrate_nonlinear'] = (df['doppler'] - df['pr_rate'])**2
    df['snr_z'] = (df['snr'] - df['snr_mean_5']) / (df['snr_std_5'] + 1e-6)
    return df

df = add_extra_features(df)

# ================================
# 2. Encoding Category Variables 
# ================================
for col in ["sys", "sv", "prn"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Missing values in numeric form
df_all = df.fillna(-9999)

# =========================================================
# 3. Chronological stratification (for spoofing simulation)
# =========================================================
df_attack = df_all[df_all["attack_label"] == 1]
df_clean = df_all[df_all["attack_label"] == 0]

# Test set: 20% spoofed and 20% clean
test_attack = df_attack.sample(frac=0.2, random_state=42)
test_clean = df_clean.sample(frac=0.2, random_state=42)
df_test = pd.concat([test_attack, test_clean]).sort_values("time").reset_index(drop=True)

# Train set: the rest
train_attack = df_attack.drop(test_attack.index)
train_clean = df_clean.drop(test_clean.index)
df_train = pd.concat([train_attack, train_clean]).sort_values("time").reset_index(drop=True)

print("Train class distribution:\n", df_train["attack_label"].value_counts())
print("Test class distribution:\n", df_test["attack_label"].value_counts())

# =================================
# 4. Separating function and goal
# =================================
drop_cols = ["time", "attack_label", "attack_type", "time_utc"]
X_train = df_train.drop(columns=drop_cols, errors="ignore")
y_train = df_train["attack_label"]
X_test = df_test.drop(columns=drop_cols, errors="ignore")
y_test = df_test["attack_label"]


# =====================================
# 5. SMOTE over-sampling on train-data
# =====================================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
#print(f"After SMOTE, train distribution:\n{pd.Series(y_train_res).value_counts()}")

# ============================================================
# 6. Optuna hyperparameter tuning
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,

        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),

        # important: keep them between 0.5–1.0!
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        # XGBoost bug prevention: don't let them overflow
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),

        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
    }

    # BUGFIX: base_score fixed on 0.5
    params["base_score"] = 0.5

    model = XGBClassifier(**params)

    scores = []
    for train_idx, valid_idx in tscv.split(X_train_res):
        X_tr, X_val = X_train_res.iloc[train_idx], X_train_res.iloc[valid_idx]
        y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[valid_idx]

        model.fit(X_tr, y_tr)

        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)

        scores.append(f1_score(y_val, y_pred))

    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50) 

print("\nBest params:")
print(study.best_params)
best_params = study.best_params


# ============================================================
# 7. Training the modell with the best parameters
# ============================================================
best_model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)

best_model.fit(X_train_res, y_train_res)

# ============================================================
# 8. Test Evaluation
# ============================================================
y_proba = best_model.predict_proba(X_test)[:, 1]
threshold = 0.4
y_pred = (y_proba > threshold).astype(int)

f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"\nTest F1-score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")


# ============================================================
# 8. Save predictions
# ============================================================
out = df_test.copy()
out["pred_attack"] = y_pred
out["attack_probability"] = y_proba
out.to_csv("gnss_test_predictions_optuna.csv", index=False)

print("\nSaved: gnss_test_predictions_optuna.csv")


# ============================================================
# 9. Exporting the modell
# ============================================================
import joblib

joblib.dump(best_model, "gnss_xgboost_model.joblib")
print("Modell mentve mint gnss_xgboost_model.joblib")


