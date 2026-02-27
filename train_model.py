import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# ---- BREAK PERFECT RULE ----
np.random.seed(42)

base_pump = ((df["moisture"] < 22) & (df["rainfall"] == 0)).astype(int)
noise = np.random.rand(len(df)) < 0.10  # 10% noise
df["pump"] = np.where(noise, 1 - base_pump, base_pump)

# Remove leakage feature
X = df.drop(["pump", "rainfall"], axis=1)
y = df["pump"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model trained successfully ✅")
print("Accuracy:", acc)

# Save model
joblib.dump(model, "smartirrigation.pkl")
print("Model saved as smartirrigation.pkl ✅")