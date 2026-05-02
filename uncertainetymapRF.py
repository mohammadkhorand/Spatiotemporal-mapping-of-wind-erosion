import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
os.system('cls' if os.name == 'nt' else 'clear')
# ---------------- Path ----------------
base = 'E:/Study/final/data'

files = sorted(glob.glob(os.path.join(base,'categorized_wind_erosion_*.xlsx')))

summary = []
all_points = []

for f in files:

    year = os.path.basename(f).split("_")[-1].split(".")[0]
    print("Processing:", year)

    df = pd.read_excel(f)

    # -------- features ----------
    features = ['sand','clay','silt','wind_speed','moisture','test_duration']
    X = df[features]
    y = df['Erosion_Rate']

    mask = X.notna().all(axis=1) & y.notna()

    X = X.loc[mask]
    y = y.loc[mask]

    coords = df.loc[mask,['Latitude','Longitude']]

    # -------- Random Forest --------
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X,y)

    # -------- tree predictions --------
    tree_preds = np.vstack([tree.predict(X) for tree in rf.estimators_])

    y_mean = tree_preds.mean(axis=0)
    y_std = tree_preds.std(axis=0,ddof=1)

    y_p05 = np.percentile(tree_preds,5,axis=0)
    y_p95 = np.percentile(tree_preds,95,axis=0)
    pi_width = y_p95 - y_p05

    # -------- save table ----------
    result = coords.copy()
    result['mean_erosion_rate'] = y_mean
    result['uncertainty_std'] = y_std
    result['PI_width_P95_P05'] = pi_width

    out_file = os.path.join(base,f'categorized_wind_erosion_{year}_uncertainty.xlsx')
    result.to_excel(out_file,index=False)

    print("saved:",out_file)

    # -------- yearly statistics --------
    summary.append({
        "Year":year,
        "Mean_Uncertainty":y_std.mean(),
        "Min_Uncertainty":y_std.min(),
        "Max_Uncertainty":y_std.max(),
        "Std_Uncertainty":y_std.std()
    })

    result['Year'] = year
    all_points.append(result)

# ---------------- Save statistics table ----------------
summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values("Year")

summary_file = os.path.join(base,"RF_uncertainty_yearly_statistics.xlsx")
summary_df.to_excel(summary_file,index=False)

print("✅ yearly statistics saved")

# ---------------- Combine all points for map ----------------
map_df = pd.concat(all_points)

# ---------------- Final map ----------------
plt.figure(figsize=(8,7))

sc = plt.scatter(
    map_df['Longitude'],
    map_df['Latitude'],
    c=map_df['uncertainty_std'],
    cmap='Reds',
    s=8
)

plt.colorbar(sc,label='RF Prediction Uncertainty (STD)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Random Forest Uncertainty (2015–2024)')

plt.tight_layout()

map_file = os.path.join(base,"RF_uncertainty_map.png")
plt.savefig(map_file,dpi=600)
plt.close()

print("✅ uncertainty map saved:",map_file)
