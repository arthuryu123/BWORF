import time, sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, "models")
from bworf_with_mi import BWORFClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[" + ts + "] " + str(msg), flush=True)

log("Loading subsample CSV")
df = pd.read_csv("data/cleaned/heart_attack_2022_brfss.csv.gz")
log("Loaded " + str(len(df)) + " rows")

y_full = df["HadHeartAttack"].values
X_full = df.drop(columns=["HadHeartAttack"])
cat_cols = ["RaceEthnicityCategory", "TetanusLast10Tdap", "CovidPos"]
for c in cat_cols:
    X_full[c] = X_full[c].astype(str)

ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)], remainder="passthrough")
X_enc = ct.fit_transform(X_full)
log("After one-hot: X.shape=" + str(X_enc.shape))

for n in [1000, 2000, 5000, 10000, 18000]:
    if n > len(X_enc):
        log("SKIP n=" + str(n))
        continue
    X_sub, _, y_sub, _ = train_test_split(X_enc, y_full, train_size=n, stratify=y_full, random_state=42)
    log("--- n=" + str(n) + " ---")
    log("  positives in subsample: " + str(int(y_sub.sum())) + "/" + str(n))
    m = BWORFClassifier(n_estimators=100, max_depth=5, l1_strength=1.0, weighted_bootstrap=True, bootstrap_temperature=1.0, n_tries=2, random_state=13)
    log("  fit start")
    t0 = time.time()
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        m.fit(X_sub, y_sub)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()
    elapsed = time.time() - t0
    log("  FIT n=" + str(n) + " elapsed=" + str(round(elapsed,1)) + "s (" + str(round(elapsed/60,2)) + "min)")
log("PROFILE DONE")
