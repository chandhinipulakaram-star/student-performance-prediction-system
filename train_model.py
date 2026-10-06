import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ==========================================
# 1. Load Dataset
# ==========================================

DATA_FILE = "student_performance.csv"

if not os.path.exists(DATA_FILE):
    print("ERROR: student_performance.csv not found.")
    print("Make sure the CSV file is in the same folder as train_model.py.")
    exit()

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print()
print(df.head())

print()
print("Dataset columns:")
print(df.columns.tolist())


# ==========================================
# 2. Define Features
# ==========================================

features = [
    "Hours_Studied",
    "Attendance",
    "Previous_Score",
    "Assignments",
    "Participation",
    "Sleep_Hours"
]

target = "Final_Score"


# ==========================================
# 3. Check Columns
# ==========================================

missing_columns = [
    column
    for column in features + [target]
    if column not in df.columns
]

if missing_columns:

    print()
    print("ERROR: Missing columns:")
    print(missing_columns)

    print()
    print("Your CSV contains:")
    print(df.columns.tolist())

    exit()


# ==========================================
# 4. Prepare Data
# ==========================================

X = df[features]
y = df[target]

X = np.asarray(X)
y = np.asarray(y)


# ==========================================
# 5. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 6. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 7. Create Random Forest Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 8. Train Model
# ==========================================

print()
print("Training Random Forest model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")


# ==========================================
# 9. Make Predictions
# ==========================================

predictions = model.predict(X_test_scaled)


# ==========================================
# 10. Evaluate Model
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


print()
print("======================================")
print("       MODEL PERFORMANCE")
print("======================================")

print(
    "MAE  :",
    round(mae, 2)
)

print(
    "MSE  :",
    round(mse, 2)
)

print(
    "RMSE :",
    round(rmse, 2)
)

print(
    "R2   :",
    round(r2, 2)
)


# ==========================================
# 11. Create Model Folder
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)


# ==========================================
# 12. Save Model
# ==========================================

joblib.dump(
    model,
    "model/student_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)


# ==========================================
# 13. Final Message
# ==========================================

print()
print("======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")

print()
print("Created files:")

print("model/student_model.pkl")

print("model/scaler.pkl")
