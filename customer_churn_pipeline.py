import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# 1. CREATE A SIMULATED CUSTOMER DATASET
# ============================================================

np.random.seed(42)
n_samples = 1500

data = {
    'CustomerID': range(1001, 1001 + n_samples),
    'Tenure_Months': np.random.randint(1, 72, n_samples),
    'Monthly_Charges': np.random.uniform(20, 120, n_samples),
    'Total_Charges': np.zeros(n_samples),
    'Internet_Service': np.random.choice(
        ['DSL', 'Fiber Optic', 'No'],
        n_samples,
        p=[0.3, 0.5, 0.2]
    ),
    'Contract_Type': np.random.choice(
        ['Month-to-Month', 'One Year', 'Two Year'],
        n_samples,
        p=[0.5, 0.2, 0.3]
    ),
    'Tech_Support_Calls': np.random.poisson(1.5, n_samples)
}

df = pd.DataFrame(data)

# Calculate total charges
df['Total_Charges'] = (
    df['Tenure_Months']
    * df['Monthly_Charges']
    * np.random.uniform(0.9, 1.1, n_samples)
)

# ============================================================
# 2. CREATE CHURN TARGET
# ============================================================

churn_prob = (
    (df['Contract_Type'] == 'Month-to-Month').astype(int) * 0.4
    + (df['Tech_Support_Calls'] > 2).astype(int) * 0.3
)

df['Churn'] = (
    np.random.rand(n_samples) < churn_prob
).astype(int)

# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

missing_indices = df.sample(
    frac=0.02,
    random_state=42
).index

df.loc[missing_indices, 'Total_Charges'] = np.nan

df['Total_Charges'] = df['Total_Charges'].fillna(
    df['Total_Charges'].median()
)

# ============================================================
# 4. FEATURE ENGINEERING & ENCODING
# ============================================================

df = pd.get_dummies(
    df,
    columns=['Internet_Service', 'Contract_Type'],
    drop_first=True
)

X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 7. RANDOM FOREST + HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# ============================================================
# 8. PREDICTION
# ============================================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)

y_prob = best_model.predict_proba(
    X_test_scaled
)[:, 1]

# ============================================================
# 9. MODEL EVALUATION
# ============================================================

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nROC-AUC Score:")
print(f"{roc_auc_score(y_test, y_prob):.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("=" * 60)