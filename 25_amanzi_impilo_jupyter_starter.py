# Amanzi Impilo — Jupyter ML Starter
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# MODEL 1 — Predict potable water saved
# ------------------------------------------------------------
X_train = pd.read_csv("04_water_savings_train_X.csv")
y_train = pd.read_csv("05_water_savings_train_y.csv")["potable_saved_l"]

X_test = pd.read_csv("06_water_savings_test_X.csv")
y_test = pd.read_csv("07_water_savings_test_y.csv")["potable_saved_l"]

model_savings = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_savings.fit(X_train, y_train)
pred_savings = model_savings.predict(X_test)

print("WATER SAVINGS MODEL")
print("MAE:", mean_absolute_error(y_test, pred_savings))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_savings)))
print("R2:", r2_score(y_test, pred_savings))

# ------------------------------------------------------------
# MODEL 2 — Predict potable water use WITHOUT the system
# ------------------------------------------------------------
X_train = pd.read_csv("10_counterfactual_train_X.csv")
y_train = pd.read_csv("11_counterfactual_train_y.csv")["potable_before_l"]

X_test = pd.read_csv("12_counterfactual_test_X.csv")
y_test = pd.read_csv("13_counterfactual_test_y.csv")["potable_before_l"]

model_counterfactual = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_counterfactual.fit(X_train, y_train)
pred_counterfactual = model_counterfactual.predict(X_test)

print("\nCOUNTERFACTUAL MODEL")
print("MAE:", mean_absolute_error(y_test, pred_counterfactual))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_counterfactual)))
print("R2:", r2_score(y_test, pred_counterfactual))

# ------------------------------------------------------------
# MODEL 3 — Predict sensor replacement
# ------------------------------------------------------------
X_train = pd.read_csv("17_sensor_replacement_train_X.csv")
y_train = pd.read_csv("18_sensor_replacement_train_y.csv")["replacement_required"]

X_test = pd.read_csv("19_sensor_replacement_test_X.csv")
y_test = pd.read_csv("20_sensor_replacement_test_y.csv")["replacement_required"]

model_sensor = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model_sensor.fit(X_train, y_train)
pred_sensor = model_sensor.predict(X_test)

print("\nSENSOR REPLACEMENT MODEL")
print("Accuracy:", accuracy_score(y_test, pred_sensor))
print(classification_report(y_test, pred_sensor))

# ------------------------------------------------------------
# MODEL 4 — Predict sensor remaining useful life
# ------------------------------------------------------------
X_train = pd.read_csv("17_sensor_replacement_train_X.csv")
y_train = pd.read_csv("22_sensor_remaining_life_train_y.csv")["remaining_useful_life_days"]

X_test = pd.read_csv("19_sensor_replacement_test_X.csv")
y_test = pd.read_csv("23_sensor_remaining_life_test_y.csv")["remaining_useful_life_days"]

model_life = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_life.fit(X_train, y_train)
pred_life = model_life.predict(X_test)

print("\nSENSOR REMAINING LIFE MODEL")
print("MAE:", mean_absolute_error(y_test, pred_life))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_life)))
print("R2:", r2_score(y_test, pred_life))
