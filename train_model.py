import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Example dataset (replace with real one later)
data = {
    "slope": [20, 35, 10, 45, 50],
    "ndvi": [0.2, 0.4, -0.1, 0.3, 0.5],
    "rain_3d": [100, 200, 50, 300, 400],
    "label": [1, 1, 0, 1, 1]  # 1 = high risk, 0 = low risk
}
df = pd.DataFrame(data)

X = df[['slope','ndvi','rain_3d']]
y = df['label']

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

# Save trained model
joblib.dump(clf, "models/landslide_model.pkl")
print("✅ Model saved at models/landslide_model.pkl")
    