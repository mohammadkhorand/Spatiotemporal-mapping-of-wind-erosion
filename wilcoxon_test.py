import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import wilcoxon
import os
os.system('cls' if os.name == 'nt' else 'clear')

# 1. Read dataset from Excel file
file_path = "F:/Work/python/erosion rate ML difference/test-wind-erosion .xlsx"
df = pd.read_excel(file_path)

# 2. Define input features (X) and target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train SVR model with hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

grid_svr = GridSearchCV(SVR(), param_grid, cv=5)
grid_svr.fit(X_train_scaled, y_train)

# Select the best SVR model
svr_model = grid_svr.best_estimator_

# Predict using SVR
y_pred_svr = svr_model.predict(X_test_scaled)

# 6. Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict using Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)

# 7. Calculate absolute prediction errors for both models
svr_abs_error = np.abs(y_test - y_pred_svr)
rf_abs_error = np.abs(y_test - y_pred_rf)

# 8. Perform Wilcoxon signed-rank test to compare model errors
stat, p_value = wilcoxon(svr_abs_error, rf_abs_error)

# 9. Save predictions and errors into an Excel file
df_results = pd.DataFrame({
    "Actual": y_test,
    "SVR_Pred": y_pred_svr,
    "RF_Pred": y_pred_rf,
    "SVR_AbsError": svr_abs_error,
    "RF_AbsError": rf_abs_error
})

output_path = "F:/Work/python/erosion rate ML difference/comparison_results.xlsx"
df_results.to_excel(output_path, index=False)

# 10. Print statistical test results
print("Wilcoxon statistic:", stat)
print("P-value:", p_value)

# 11. Interpret statistical significance
if p_value < 0.05:
    print("There is a statistically significant difference between the two models.")
    if svr_abs_error.mean() < rf_abs_error.mean():
        print("SVR performs significantly better than Random Forest.")
    else:
        print("Random Forest performs significantly better than SVR.")
else:
    print("No statistically significant difference was found between the two models.")
