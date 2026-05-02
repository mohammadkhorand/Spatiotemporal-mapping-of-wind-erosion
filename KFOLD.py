import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Load data from Excel
# ----------------------------
file_path = r""
df = pd.read_excel(file_path)

# ----------------------------
# 2. Define X and y
# ----------------------------
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ----------------------------
# 3. 10-Fold Cross Validation
# ----------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)

rf_r2_list = []
rf_mse_list = []

svr_r2_list = []
svr_mse_list = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # -----------------
    # Random Forest
    # -----------------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_r2_list.append(r2_score(y_test, rf_pred))
    rf_mse_list.append(mean_squared_error(y_test, rf_pred))

    # -----------------
    # SVR (Scaled)
    # -----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(
        C=10,
        epsilon=0.1,
        gamma='scale',
        kernel='rbf'
    )

    svr.fit(X_train_scaled, y_train)
    svr_pred = svr.predict(X_test_scaled)

    svr_r2_list.append(r2_score(y_test, svr_pred))
    svr_mse_list.append(mean_squared_error(y_test, svr_pred))

# ----------------------------
# 4. Print Results
# ----------------------------
print("===== Random Forest 10-Fold Results =====")
print("Mean R2  :", round(np.mean(rf_r2_list), 4))
print("SD   R2  :", round(np.std(rf_r2_list), 4))
print("Mean MSE :", round(np.mean(rf_mse_list), 4))
print("SD   MSE :", round(np.std(rf_mse_list), 4))

print("\n===== SVR 10-Fold Results =====")
print("Mean R2  :", round(np.mean(svr_r2_list), 4))
print("SD   R2  :", round(np.std(svr_r2_list), 4))
print("Mean MSE :", round(np.mean(svr_mse_list), 4))
print("SD   MSE :", round(np.std(svr_mse_list), 4))
