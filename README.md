# Global Air Pollution — PyCharm + Streamlit

This is a UI project built from the supplied Kaggle notebook:

`global-air-pollution-minhal(1).ipynb`

The app keeps the notebook's feature engineering and saved-model naming convention.

## 1. Project structure

```text
Global_Air_Pollution_PyCharm_Streamlit/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_global_air_pollution.csv
├── models/
│   ├── best_aqi_regressor_rf.pkl
│   ├── best_aqi_classifier_rf.pkl
│   ├── aqi_regressor_dl.keras
│   ├── aqi_classifier_dl.keras
│   ├── scaler_reg.pkl
│   ├── scaler_clf.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   └── model_metrics.json
├── pages/
│   ├── 1_📊_EDA.py
│   ├── 2_🤖_AQI_Prediction.py
│   ├── 3_🏷️_AQI_Category.py
│   ├── 4_📈_Model_Performance.py
│   └── 5_📋_Dataset.py
└── utils/
    ├── __init__.py
    └── common.py
```

## 2. Put the Kaggle dataset in PyCharm

Create:

```text
data/synthetic_global_air_pollution.csv
```

The supplied notebook reports 36,733 rows and 28 columns.

## 3. Create the PyCharm interpreter

Recommended:

```text
Python 3.10 or 3.11
```

Create a virtual environment in PyCharm.

Then install:

```bash
pip install -r requirements.txt
```

TensorFlow installation can be platform-dependent. If TensorFlow is difficult to install on your machine, you can temporarily remove the DL package and use the saved Random Forest models for the UI.

## 4. Recreate the trained artifacts

The notebook saves:

```text
best_aqi_regressor_rf.pkl
best_aqi_classifier_rf.pkl
aqi_regressor_dl.keras
aqi_classifier_dl.keras
scaler_reg.pkl
scaler_clf.pkl
label_encoders.pkl
target_encoder.pkl
```

Run:

```bash
python train_models.py
```

This reproduces the notebook's preprocessing, train/test split, classical ML models, ANN models and saved artifacts.

## 5. Run the UI

From the PyCharm terminal:

```bash
streamlit run app.py
```

The browser should open the dashboard automatically.

## 6. Pages

### Dashboard
Dataset KPIs, monthly AQI trend and category overview.

### EDA
The notebook's main analysis:
- AQI distribution
- AQI category distribution
- average AQI by country
- average AQI by season
- city rankings
- pollutant vs AQI
- correlation heatmap
- missing-value summary

### AQI Prediction
Uses:
- `scaler_reg.pkl`
- `best_aqi_regressor_rf.pkl`

### AQI Category
Uses:
- `scaler_clf.pkl`
- `best_aqi_classifier_rf.pkl`
- `target_encoder.pkl`

### Model Performance
Displays the classical and DL model families used in the notebook and, after training, the exact saved metric values.

## 7. Important notebook compatibility detail

The notebook creates these features:

```text
Month_sin
Month_cos
DayOfYear
```

It label-encodes:

```text
Season
Country
City
```

and uses this exact feature order:

```text
Temperature_C
Humidity_%
WindSpeed_mps
Pressure_hPa
Rainfall_mm
Visibility_km
UV_Index
PM2_5
PM10
NO2
SO2
CO
Ozone
NH3
Aerosol_Optical_Depth
Traffic_Density
Industrial_Activity
Green_Cover_%
Population_Density
Season_enc
Country_enc
City_enc
Month_sin
Month_cos
DayOfYear
```

The Streamlit prediction code follows that order.

## 8. Do not retrain on every Streamlit refresh

The app loads the saved model and preprocessing artifacts. Training belongs in:

```bash
python train_models.py
```

This keeps the UI fast.

## 9. Kaggle-to-PyCharm mapping

Kaggle:

```python
pd.read_csv("/kaggle/input/...")
```

PyCharm:

```python
data/synthetic_global_air_pollution.csv
```

Kaggle model files:

```text
best_aqi_regressor_rf.pkl
...
```

PyCharm:

```text
models/
```

## 10. Synthetic-data disclaimer

The notebook's dataset is synthetic. The dashboard should be presented as an analytical/ML demonstration, not as verified real-world atmospheric monitoring.
