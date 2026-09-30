"""Train SVM, Random Forest and KNN on a bearing fault CSV and save them.

Usage:
    python train_model.py                                  (uses the first CSV in data/, last column = label)
    python train_model.py --data data/myfile.csv --label fault_type
"""
import argparse
import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------- arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="path to the CSV file (default: first CSV in data/)")
parser.add_argument("--label", help="name of the fault label column (default: last column)")
parser.add_argument("--test_size", type=float, default=0.2)
args = parser.parse_args()

data_path = args.data
if data_path is None:
    files = glob.glob(os.path.join("data", "*.csv"))
    if not files:
        raise SystemExit("No CSV found. Put your Kaggle CSV in the data/ folder.")
    data_path = files[0]

# ---------------------------------------------------------------- load data
df = pd.read_csv(data_path).dropna()
label_col = args.label or df.columns[-1]
print(f"Data: {data_path}  |  rows: {len(df)}  |  label column: {label_col}")

y = df[label_col]
X = df.drop(columns=[label_col]).select_dtypes(include="number")   # numeric features only
print(f"Features ({X.shape[1]}): {list(X.columns)}")
print(f"Classes: {sorted(y.unique())}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42, stratify=y
)

# ---------------------------------------------------------------- models
# SVM and KNN need scaled data, so they use a pipeline with StandardScaler.
models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, pred)
    print(f"===== {name} : accuracy = {accuracies[name]:.2%} =====")
    print(classification_report(y_test, pred, zero_division=0))

# ---------------------------------------------------------------- save
os.makedirs("models", exist_ok=True)
joblib.dump(
    {
        "models": models,
        "features": list(X.columns),
        "defaults": X.median().to_dict(),   # starting values for manual input in the app
        "accuracies": accuracies,
        "label_col": label_col,
    },
    "models/bearing_models.joblib",
)
print("Saved -> models/bearing_models.joblib")
