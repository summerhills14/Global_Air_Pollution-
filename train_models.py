"""
Recreates the trained artifacts from the Kaggle notebook in PyCharm.

Run once from the project root:
    python train_models.py

It writes:
models/best_aqi_regressor_rf.pkl
models/best_aqi_classifier_rf.pkl
models/aqi_regressor_dl.keras
models/aqi_classifier_dl.keras
models/scaler_reg.pkl
models/scaler_clf.pkl
models/label_encoders.pkl
models/target_encoder.pkl
models/model_metrics.json
"""

from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_global_air_pollution.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

NUMERIC_FEATURES = [
    "Temperature_C", "Humidity_%", "WindSpeed_mps", "Pressure_hPa",
    "Rainfall_mm", "Visibility_km", "UV_Index", "PM2_5", "PM10",
    "NO2", "SO2", "CO", "Ozone", "NH3", "Aerosol_Optical_Depth",
    "Traffic_Density", "Industrial_Activity", "Green_Cover_%",
    "Population_Density"
]
CATEGORICAL_FEATURES = ["Season", "Country", "City"]
ENGINEERED_FEATURES = ["Month_sin", "Month_cos", "DayOfYear"]

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

data = df.copy()
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)
data["DayOfYear"] = data["Date"].dt.dayofyear

imputer = SimpleImputer(strategy="median")
data[NUMERIC_FEATURES] = imputer.fit_transform(data[NUMERIC_FEATURES])

encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    data[col + "_enc"] = le.fit_transform(data[col])
    encoders[col] = le

FEATURE_COLS = (
    NUMERIC_FEATURES
    + [c + "_enc" for c in CATEGORICAL_FEATURES]
    + ENGINEERED_FEATURES
)

X = data[FEATURE_COLS]
y_reg = data["AQI"]

target_encoder = LabelEncoder()
y_clf = target_encoder.fit_transform(data["AQI_Category"])

X_train, X_test, yreg_train, yreg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=RANDOM_STATE
)
Xc_train, Xc_test, yclf_train, yclf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

scaler_reg = StandardScaler().fit(X_train)
X_train_s = scaler_reg.transform(X_train)
X_test_s = scaler_reg.transform(X_test)

scaler_clf = StandardScaler().fit(Xc_train)
Xc_train_s = scaler_clf.transform(Xc_train)
Xc_test_s = scaler_clf.transform(Xc_test)

reg_results = {}

def eval_reg(name, model):
    model.fit(X_train_s, yreg_train)
    pred = model.predict(X_test_s)
    reg_results[name] = {
        "R2": float(r2_score(yreg_test, pred)),
        "MAE": float(mean_absolute_error(yreg_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(yreg_test, pred))),
    }
    print(name, reg_results[name])
    return model

lin_reg = eval_reg("Linear Regression", LinearRegression())
rf_reg = eval_reg(
    "Random Forest",
    RandomForestRegressor(
        n_estimators=200, max_depth=None,
        random_state=RANDOM_STATE, n_jobs=-1
    )
)
gb_reg = eval_reg(
    "Gradient Boosting",
    GradientBoostingRegressor(random_state=RANDOM_STATE)
)

if HAS_XGB:
    xgb_reg = eval_reg(
        "XGBoost",
        XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    )

clf_results = {}

def eval_clf(name, model):
    model.fit(Xc_train_s, yclf_train)
    pred = model.predict(Xc_test_s)
    clf_results[name] = {
        "Accuracy": float(accuracy_score(yclf_test, pred)),
        "F1_weighted": float(f1_score(yclf_test, pred, average="weighted")),
    }
    print(name, clf_results[name])
    return model

log_clf = eval_clf(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced")
)
rf_clf = eval_clf(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
)

if HAS_XGB:
    xgb_clf = eval_clf(
        "XGBoost",
        XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_STATE, eval_metric="mlogloss"
        )
    )

def build_regressor(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse", metrics=["mae"]
    )
    return model

dl_reg = build_regressor(X_train_s.shape[1])
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
dl_reg.fit(
    X_train_s, yreg_train,
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

dl_preds = dl_reg.predict(X_test_s, verbose=0).flatten()
reg_results["Deep Learning (ANN)"] = {
    "R2": float(r2_score(yreg_test, dl_preds)),
    "MAE": float(mean_absolute_error(yreg_test, dl_preds)),
    "RMSE": float(np.sqrt(mean_squared_error(yreg_test, dl_preds))),
}

def build_classifier(input_dim, n_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

dl_clf = build_classifier(Xc_train_s.shape[1], len(target_encoder.classes_))
class_weights = compute_class_weight(
    "balanced", classes=np.unique(yclf_train), y=yclf_train
)
class_weight_dict = dict(enumerate(class_weights))

early_stop_clf = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

dl_clf.fit(
    Xc_train_s, yclf_train,
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stop_clf],
    verbose=1
)

dl_clf_probs = dl_clf.predict(Xc_test_s, verbose=0)
dl_clf_preds = np.argmax(dl_clf_probs, axis=1)
clf_results["Deep Learning (ANN)"] = {
    "Accuracy": float(accuracy_score(yclf_test, dl_clf_preds)),
    "F1_weighted": float(f1_score(yclf_test, dl_clf_preds, average="weighted")),
}

# Same artifacts named by the original notebook.
joblib.dump(rf_reg, MODEL_DIR / "best_aqi_regressor_rf.pkl")
joblib.dump(rf_clf, MODEL_DIR / "best_aqi_classifier_rf.pkl")
dl_reg.save(MODEL_DIR / "aqi_regressor_dl.keras")
dl_clf.save(MODEL_DIR / "aqi_classifier_dl.keras")
joblib.dump(scaler_reg, MODEL_DIR / "scaler_reg.pkl")
joblib.dump(scaler_clf, MODEL_DIR / "scaler_clf.pkl")
joblib.dump(encoders, MODEL_DIR / "label_encoders.pkl")
joblib.dump(target_encoder, MODEL_DIR / "target_encoder.pkl")

(MODEL_DIR / "model_metrics.json").write_text(
    json.dumps({
        "regression": reg_results,
        "classification": clf_results,
        "feature_columns": FEATURE_COLS,
        "target_classes": list(target_encoder.classes_),
    }, indent=2),
    encoding="utf-8"
)

print("\nTraining complete.")
print("Artifacts saved to:", MODEL_DIR)
