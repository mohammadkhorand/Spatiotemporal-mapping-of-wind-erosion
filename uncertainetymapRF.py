import pandas as pd, numpy as np, os, json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ---------------- Paths ----------------
base = 'e:/Study/final/'
file_xlsx = os.path.join(base,'categorized_wind_erosion.xlsx')
out_csv = os.path.join(base,'uncertainty_map_points.csv')

# ---------------- Load data ----------------
df = pd.read_excel(file_xlsx)

features = ['sand','clay','silt','wind_speed','moisture','test_duration']
X = df[features]
y = df['Erosion_Rate']

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
coords = df.loc[mask, ['Latitude','Longitude']]

# ---------------- Random Forest ----------------
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# ---------------- Tree-level predictions ----------------
tree_preds = np.vstack([tree.predict(X) for tree in rf.estimators_])

# Equation (14): mean prediction
y_mean = tree_preds.mean(axis=0)

# Equation (15): standard deviation (uncertainty)
y_std = tree_preds.std(axis=0, ddof=1)

# Equation (17): empirical prediction interval
y_p05 = np.percentile(tree_preds, 5, axis=0)
y_p95 = np.percentile(tree_preds, 95, axis=0)
pi_width = y_p95 - y_p05

# ---------------- Output table ----------------
result = coords.copy()
result['mean_erosion_rate'] = y_mean
result['uncertainty_std'] = y_std
result['PI_width_P95_P05'] = pi_width

result.to_csv(out_csv, index=False)

# ---------------- Figures ----------------
def plot_map(lon, lat, values, title, label, fname, cmap):
    plt.figure(figsize=(7,6))
    sc = plt.scatter(lon, lat, c=values, cmap=cmap, s=20)
    plt.colorbar(sc, label=label)
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(base, fname), dpi=300)
    plt.close()

# Figure A
plot_map(
    result['Longitude'], result['Latitude'],
    result['mean_erosion_rate'],
    'Mean Predicted Wind Erosion Rate',
    'Erosion rate',
    'Figure_A_mean_erosion.png',
    'viridis'
)

# Figure B
plot_map(
    result['Longitude'], result['Latitude'],
    result['uncertainty_std'],
    'Standard Deviation of RF Predictions',
    'σ_RF',
    'Figure_B_RF_uncertainty_STD.png',
    'Reds'
)

# Figure C
plot_map(
    result['Longitude'], result['Latitude'],
    result['PI_width_P95_P05'],
    'Prediction Interval Width (P95 − P5)',
    'PI width',
    'Figure_C_prediction_interval.png',
    'magma'
)

print("✅ Figures A–C generated and saved.")
print("✅ Output table:", out_csv)
