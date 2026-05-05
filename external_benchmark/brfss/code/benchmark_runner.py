"""
benchmark_runner.py – thesis-grade external benchmark runner.

Reads cleaned datasets produced by prepare_data.py, runs deterministic
Stratified 10-Fold CV across 5 fixed seeds for LR / RF / XGB / LGBM / BWORF,
and writes fold-level metrics, OOF predictions, summaries, and run configs.
No subsampling—every cleaned row is used.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    label_binarize,
)

# Optional deps
try:
    import xgboost as xgb
    _HAS_XGB = True
except (ImportError, OSError):
    _HAS_XGB = False

try:
    import lightgbm as lgbm
    _HAS_LGBM = True
except (ImportError, OSError):
    _HAS_LGBM = False

# BWORF import
_models_dir = str(Path(__file__).resolve().parent / "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
from bworf_with_mi import BWORFClassifier  # noqa: E402

# ORF import (optional – scikit-tree may not be installed)
from orf_treeple import check_orf_available, make_orf, predict_proba_aligned  # noqa: E402
_HAS_ORF = check_orf_available()

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {
    "breast_cancer": dict(
        file="breast_cancer.csv.gz", target="class", task="binary",
        categorical_cols=[
            "age", "menopause", "tumor_size", "inv_nodes", "node_caps",
            "deg_malig", "breast", "breast_quad", "irradiat",
        ],
    ),
    "diabetes": dict(
        file="diabetes.csv.gz", target="Outcome", task="binary",
        categorical_cols=[],
    ),
    "diabetes_cdc": dict(
        file="diabetes_cdc.csv.gz", target="class", task="binary",
        categorical_cols=[],
    ),
    "diabetes_hospital": dict(
        file="diabetes_hospital.csv.gz", target="class", task="multiclass",
        categorical_cols=[
            "race", "gender", "age", "admission_type_id",
            "discharge_disposition_id", "admission_source_id", "payer_code",
            "metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone",
            "change", "diabetesMed", "admission_source_id_mapped",
        ],
    ),
    "heart_failure": dict(
        file="heart_failure.csv.gz", target="DEATH_EVENT", task="binary",
        categorical_cols=[],
    ),
    "thyroid": dict(
        file="thyroid.csv.gz", target="class", task="multiclass",
        categorical_cols=[],
    ),
    "heart_attack_2022_brfss": dict(
        file="heart_attack_2022_brfss.csv.gz", target="HadHeartAttack", task="binary",
        categorical_cols=["RaceEthnicityCategory", "TetanusLast10Tdap", "CovidPos"],
    ),
    "heart_attack_2022_brfss_full": dict(
        file="heart_attack_2022_brfss_full.csv.gz", target="HadHeartAttack", task="binary",
        categorical_cols=["RaceEthnicityCategory", "TetanusLast10Tdap", "CovidPos"],
    ),
}

BINARY_METRICS = [
    "accuracy", "balanced_accuracy", "f1_pos", "precision_pos", "recall_pos",
    "mcc", "auroc", "auprc", "log_loss", "brier",
]
MULTI_METRICS = [
    "accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
    "macro_recall", "mcc", "auroc_macro_ovr", "auprc_macro_ovr",
    "log_loss", "brier_multiclass",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External benchmark runner")
    p.add_argument("--data_dir", default="data/cleaned")
    p.add_argument("--out_dir", default="outputs_external_thesis")
    p.add_argument("--datasets", default="all")
    p.add_argument("--models", default="lr,rf,xgb,lgbm,bworf")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--seeds", default="13,42,77,123,2025")
    p.add_argument("--n_jobs", type=int, default=None)
    p.add_argument("--rf_n_estimators", type=int, default=500)
    p.add_argument("--xgb_n_estimators", type=int, default=500)
    p.add_argument("--lgbm_n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--orf_n_estimators", type=int, default=100)
    p.add_argument("--bworf_n_estimators", type=int, default=50)
    p.add_argument("--bworf_max_depth", type=int, default=5)
    p.add_argument("--bworf_l1_strength", type=float, default=1.0)
    p.add_argument("--bworf_weighted_bootstrap", type=_str2bool, default=True)
    p.add_argument("--bworf_bootstrap_temperature", type=float, default=1.0)
    p.add_argument("--bworf_parallel_n_jobs", type=int, default=-1,
                   help="n_jobs for BWORFParallelClassifier (default -1 = all CPUs)")
    p.add_argument("--bworf_n_tries", type=int, default=2)
    p.add_argument("--fold_ids", type=str, default="", help="Comma-separated fold indices to run")
    p.add_argument("--drop_incomplete_oof", type=_str2bool, default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(spec: dict, X_columns: list[str], for_lr: bool) -> Pipeline:
    cat_cols = [c for c in spec["categorical_cols"] if c in X_columns]
    num_cols = [c for c in X_columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    ct = ColumnTransformer(transformers, remainder="drop")

    steps: list[tuple] = [("prep", ct)]
    if for_lr:
        steps.append(("scale", StandardScaler(with_mean=False)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Model builders (return fresh instance per fold)
# ---------------------------------------------------------------------------

def _build_lr(**_kw) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs",
    )


def _build_rf(task: str, seed: int, n_jobs: int, args: argparse.Namespace, **_kw):
    return RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )


def _build_xgb(task: str, seed: int, n_jobs: int, args: argparse.Namespace,
               n_classes: int = 2, y_train: np.ndarray | None = None, **_kw):
    if not _HAS_XGB:
        return None
    params: dict = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.max_depth,
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
    )
    if task == "binary":
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        if y_train is not None:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            if n_pos > 0:
                params["scale_pos_weight"] = n_neg / n_pos
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"
    return xgb.XGBClassifier(**params)


def _build_lgbm(task: str, seed: int, n_jobs: int, args: argparse.Namespace,
                n_classes: int = 2, **_kw):
    if not _HAS_LGBM:
        return None
    params: dict = dict(
        n_estimators=args.lgbm_n_estimators,
        max_depth=args.max_depth,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
        verbose=-1,
    )
    if task == "multiclass":
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
    return lgbm.LGBMClassifier(**params)


def _build_bworf_parallel(seed: int, args: argparse.Namespace, **_kw):
    """Build BWORFParallelClassifier using the same hyperparameters
    as _build_bworf, plus the n_jobs argument.

    The parallel implementation is byte-equivalent to the serial one for
    the same random_state; see validate_parallel_bworf.py for the proof.
    Use this model name (--models bworf_parallel) only when fitting on
    large datasets where the serial version is the bottleneck.
    """
    try:
        # models/ is on sys.path; importing bworf_parallel pulls in
        # bworf_with_mi as a side effect via its own absolute import.
        from bworf_parallel import BWORFParallelClassifier
    except ImportError as e:
        raise ImportError(
            "BWORFParallelClassifier not available: "
            f"{e}. Make sure models/bworf_parallel.py is present "
            "and models/ is on sys.path."
        )

    return BWORFParallelClassifier(
        n_estimators=args.bworf_n_estimators,
        max_depth=args.bworf_max_depth,
        l1_strength=args.bworf_l1_strength,
        weighted_bootstrap=args.bworf_weighted_bootstrap,
        bootstrap_temperature=args.bworf_bootstrap_temperature,
        n_tries=args.bworf_n_tries,
        random_state=seed,
        n_jobs=args.bworf_parallel_n_jobs,
        verbose=0,
    )



def _build_bworf(seed: int, args: argparse.Namespace, **_kw):
    return BWORFClassifier(
        n_estimators=args.bworf_n_estimators,
        max_depth=args.bworf_max_depth,
        l1_strength=args.bworf_l1_strength,
        weighted_bootstrap=args.bworf_weighted_bootstrap,
        bootstrap_temperature=args.bworf_bootstrap_temperature,
        n_tries=args.bworf_n_tries,
        random_state=seed,
    )


def _build_orf(seed: int, n_jobs: int, args: argparse.Namespace, **_kw):
    if not _HAS_ORF:
        return None
    return make_orf(
        random_state=seed,
        n_estimators=args.orf_n_estimators,
        max_depth=args.max_depth,
        n_jobs=n_jobs,
        class_weight="balanced",
    )


MODEL_BUILDERS: dict[str, callable] = {
    "lr": _build_lr,
    "rf": _build_rf,
    "xgb": _build_xgb,
    "lgbm": _build_lgbm,
    "bworf": _build_bworf,
    "bworf_parallel": _build_bworf_parallel,
    "orf": _build_orf,
}

# ---------------------------------------------------------------------------
# Probability sanitisation
# ---------------------------------------------------------------------------

def sanitize_proba(P: np.ndarray, n_classes: int,
                   context: str = "") -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        if n_classes == 2:
            P = np.column_stack([1.0 - P, P])
        else:
            raise ValueError(f"1-D proba with n_classes={n_classes}. {context}")
    if P.shape[1] != n_classes:
        raise ValueError(
            f"proba shape {P.shape} but expected n_classes={n_classes}. {context}"
        )
    P = np.clip(P, 1e-15, 1.0 - 1e-15)
    if n_classes > 2:
        P = P / P.sum(axis=1, keepdims=True)
    return P


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    proba: np.ndarray, task: str,
                    n_classes: int) -> dict[str, float]:
    m: dict[str, float] = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["log_loss"] = log_loss(y_true, proba, labels=list(range(n_classes)))

    if task == "binary":
        m["f1_pos"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["precision_pos"] = precision_score(y_true, y_pred, pos_label=1,
                                             zero_division=0)
        m["recall_pos"] = recall_score(y_true, y_pred, pos_label=1,
                                       zero_division=0)
        p1 = proba[:, 1]
        m["auroc"] = roc_auc_score(y_true, p1)
        m["auprc"] = average_precision_score(y_true, p1)
        m["brier"] = float(np.mean((p1 - y_true) ** 2))
    else:
        m["macro_f1"] = f1_score(y_true, y_pred, average="macro",
                                 zero_division=0)
        m["macro_precision"] = precision_score(y_true, y_pred, average="macro",
                                               zero_division=0)
        m["macro_recall"] = recall_score(y_true, y_pred, average="macro",
                                         zero_division=0)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        m["auroc_macro_ovr"] = roc_auc_score(
            y_bin, proba, multi_class="ovr", average="macro",
        )
        m["auprc_macro_ovr"] = float(np.mean([
            average_precision_score(y_bin[:, k], proba[:, k])
            for k in range(n_classes)
        ]))
        y_oh = label_binarize(y_true, classes=list(range(n_classes)))
        m["brier_multiclass"] = float(np.mean(np.sum((proba - y_oh) ** 2, axis=1)))

    return m


# ---------------------------------------------------------------------------
# BWORF stdout suppression
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_stdout_stderr():
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


# ---------------------------------------------------------------------------
# Package versions
# ---------------------------------------------------------------------------

def _pkg_versions() -> dict[str, str]:
    import sklearn
    v: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    if _HAS_XGB:
        v["xgboost"] = xgb.__version__
    if _HAS_LGBM:
        v["lightgbm"] = lgbm.__version__
    if _HAS_ORF:
        try:
            import sktree
            v["scikit-tree"] = sktree.__version__
        except Exception:
            v["scikit-tree"] = "installed (version unknown)"
    return v


# ---------------------------------------------------------------------------
# Model‐params JSON helper
# ---------------------------------------------------------------------------

def _model_params_json(model_name: str, args: argparse.Namespace,
                       seed: int) -> str:
    if model_name == "lr":
        d = dict(max_iter=2000, class_weight="balanced", solver="lbfgs")
    elif model_name == "rf":
        d = dict(n_estimators=args.rf_n_estimators, max_depth=args.max_depth,
                 class_weight="balanced", random_state=seed)
    elif model_name == "xgb":
        d = dict(n_estimators=args.xgb_n_estimators, max_depth=args.max_depth,
                 random_state=seed)
    elif model_name == "lgbm":
        d = dict(n_estimators=args.lgbm_n_estimators, max_depth=args.max_depth,
                 class_weight="balanced", random_state=seed)
    elif model_name == "bworf":
        d = dict(
            n_estimators=args.bworf_n_estimators,
            max_depth=args.bworf_max_depth,
            l1_strength=args.bworf_l1_strength,
            weighted_bootstrap=args.bworf_weighted_bootstrap,
            bootstrap_temperature=args.bworf_bootstrap_temperature,
            n_tries=args.bworf_n_tries,
            random_state=seed,
        )
    elif model_name == "orf":
        d = dict(n_estimators=args.orf_n_estimators, max_depth=args.max_depth,
                 class_weight="balanced", random_state=seed)
    else:
        d = {}
    return json.dumps(d, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_dataset(ds_name: str, spec: dict, seeds: list[int],
                model_names: list[str], args: argparse.Namespace,
                n_jobs: int) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    task = spec["task"]
    target = spec["target"]

    # Load data
    df = pd.read_csv(data_dir / spec["file"])
    n_samples = len(df)
    y_raw = df[target].values
    X_df = df.drop(columns=[target])
    feature_cols = list(X_df.columns)

    # Cast categorical cols to string BEFORE anything else
    for c in spec["categorical_cols"]:
        if c in X_df.columns:
            X_df[c] = X_df[c].astype(str)

    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    label_map = {str(orig): int(enc) for orig, enc in
                 zip(le.classes_, le.transform(le.classes_))}

    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}  |  n={n_samples}  |  features={len(feature_cols)}"
          f"  |  classes={n_classes}  |  task={task}")

    # Determine which models actually run
    models_run: list[str] = []
    models_skipped: dict[str, str] = {}
    for mn in model_names:
        if mn == "xgb" and not _HAS_XGB:
            models_skipped[mn] = "skipped_missing_dependency"
        elif mn == "lgbm" and not _HAS_LGBM:
            models_skipped[mn] = "skipped_missing_dependency"
        elif mn == "orf" and not _HAS_ORF:
            models_skipped[mn] = "skipped_missing_dependency"
        else:
            models_run.append(mn)

    if models_skipped:
        print(f"  Skipped: {models_skipped}")
    print(f"  Models:  {models_run}")

    metric_names = BINARY_METRICS if task == "binary" else MULTI_METRICS
    fold_rows: list[dict] = []

    # OOF bookkeeping: write incrementally per seed+model
    oof_path = out_dir / "oof_predictions.csv.gz"
    oof_header_written = False
    oof_total_rows = 0

    for mi, model_name in enumerate(models_run):
        is_lr = (model_name == "lr")
        is_bworf = (model_name == "bworf")
        is_orf = (model_name == "orf")

        for si, seed in enumerate(seeds):
            # Pre-allocate OOF arrays for this seed+model
            oof_pred = np.full(n_samples, -1, dtype=int)
            oof_proba = np.full((n_samples, n_classes), np.nan)

            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                                  random_state=seed)

            _fold_ids_filter = set()
            if getattr(args, "fold_ids", ""):
                _fold_ids_filter = set(int(x) for x in args.fold_ids.split(",") if x.strip())
            for fold_id, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
                if _fold_ids_filter and fold_id not in _fold_ids_filter:
                    continue
                X_train_df = X_df.iloc[train_idx]
                X_test_df = X_df.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Build & fit preprocessor on train fold only
                preprocessor = build_preprocessor(spec, feature_cols, for_lr=is_lr)
                X_train = preprocessor.fit_transform(X_train_df)
                X_test = preprocessor.transform(X_test_df)

                # Build model
                ctx = f"ds={ds_name} model={model_name} seed={seed} fold={fold_id}"
                note = ""
                builder_kw = dict(
                    task=task, seed=seed, n_jobs=n_jobs, args=args,
                    n_classes=n_classes, y_train=y_train,
                )
                model = MODEL_BUILDERS[model_name](**builder_kw)
                if model is None:
                    note = "skipped_missing_dependency"
                    row = _empty_fold_row(
                        ds_name, seed, fold_id, model_name, task,
                        len(train_idx), len(test_idx), n_classes,
                        metric_names, note, args,
                    )
                    fold_rows.append(row)
                    continue

                # --- ORF: wrap fit/predict in try/except (non-fatal) ---
                if is_orf:
                    try:
                        t0 = time.perf_counter()
                        model.fit(X_train, y_train)
                        fit_sec = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        raw_proba = predict_proba_aligned(model, X_test, n_classes)
                        y_pred_model = model.predict(X_test)
                        pred_sec = time.perf_counter() - t0
                    except Exception as exc:
                        short_msg = str(exc)[:120].replace("\n", " ")
                        note = f"orf_error:{short_msg}"
                        fold_rows.append(_empty_fold_row(
                            ds_name, seed, fold_id, model_name, task,
                            len(train_idx), len(test_idx), n_classes,
                            metric_names, note, args,
                        ))
                        continue
                else:
                    # Fit
                    t0 = time.perf_counter()
                    if is_bworf:
                        with suppress_stdout_stderr():
                            model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train)
                    fit_sec = time.perf_counter() - t0

                    # Predict
                    t0 = time.perf_counter()
                    if is_bworf:
                        with suppress_stdout_stderr():
                            raw_proba = model.predict_proba(X_test)
                            y_pred_model = model.predict(X_test)
                    else:
                        raw_proba = model.predict_proba(X_test)
                        y_pred_model = model.predict(X_test)
                    pred_sec = time.perf_counter() - t0

                # Binary: ensure 2-column matrix if model returned 1D
                if raw_proba.ndim == 1 and n_classes == 2:
                    raw_proba = np.column_stack([1.0 - raw_proba, raw_proba])
                proba = sanitize_proba(raw_proba, n_classes, context=ctx)
                y_pred = np.argmax(proba, axis=1).astype(int)

                # Sanity check: compare argmax-derived y_pred to model's native predict (non-fatal)
                if y_pred_model is not None:
                    mismatch_rate = float(np.mean(y_pred_model != y_pred))
                    if mismatch_rate > 0.0:
                        note = (note + f" pred_mismatch_rate={mismatch_rate:.3f}").strip()

                # Metrics: discrete metrics use argmax-derived y_pred; proba-based use proba
                mvals = compute_metrics(y_test, y_pred, proba, task, n_classes)

                row: dict = {
                    "dataset_name": ds_name,
                    "seed": seed,
                    "fold_id": fold_id,
                    "model": model_name,
                    "task_type": task,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_classes": n_classes,
                }
                for mk in metric_names:
                    row[mk] = mvals.get(mk, np.nan)
                row["fit_seconds"] = fit_sec
                row["pred_seconds"] = pred_sec
                row["model_params_json"] = _model_params_json(model_name, args, seed)
                row["notes"] = note
                fold_rows.append(row)

                # OOF fill: y_pred is argmax-derived; probabilities are sanitized
                oof_pred[test_idx] = y_pred.astype(int)
                oof_proba[test_idx] = proba

            # After all folds for this seed+model: validate OOF completeness
            has_nan = np.isnan(oof_proba).any()
            if has_nan and not args.drop_incomplete_oof:
                raise ValueError(
                    f"OOF incomplete for {ds_name}/{model_name}/seed={seed}. "
                    "NaN found in OOF proba."
                )

            # Build OOF dataframe for this seed+model and append
            proba_cols = {f"p{k}": oof_proba[:, k] for k in range(n_classes)}
            oof_df = pd.DataFrame({
                "sample_id": np.arange(n_samples),
                "seed": seed,
                "model": model_name,
                "y_true": y,
                "y_pred": oof_pred,
                **proba_cols,
            })
            oof_total_rows += len(oof_df)

            with gzip.open(oof_path, "at", encoding="utf-8") as fh:
                oof_df.to_csv(fh, index=False, header=(not oof_header_written),
                              lineterminator="\n")
            oof_header_written = True

            tag = f"  {model_name:<6} seed={seed}  done"
            print(tag, end="  ", flush=True)
        print()  # newline after all seeds for this model

    # ---- Write fold_results.csv ----
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_results.csv", index=False)

    # ---- Write summary.csv ----
    summary_rows = []
    for mn in models_run:
        sub = fold_df[fold_df["model"] == mn]
        row_s: dict = {"model": mn}
        for mk in metric_names:
            if mk in sub.columns:
                row_s[f"{mk}_mean"] = sub[mk].mean()
                row_s[f"{mk}_std"] = sub[mk].std()
        row_s["fit_seconds_mean"] = sub["fit_seconds"].mean()
        row_s["fit_seconds_std"] = sub["fit_seconds"].std()
        summary_rows.append(row_s)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    # ---- Write run_config.json ----
    config = {
        "dataset_name": ds_name,
        "file": spec["file"],
        "target": target,
        "categorical_cols": spec["categorical_cols"],
        "task": task,
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "n_classes": n_classes,
        "label_mapping": label_map,
        "seeds": seeds,
        "n_splits": args.n_splits,
        "models_requested": model_names,
        "models_run": models_run,
        "models_skipped": models_skipped,
        "cli_args": {k: v for k, v in vars(args).items()},
        "package_versions": _pkg_versions(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    # ---- Sanity checks ----
    expected_fold_rows = len(seeds) * args.n_splits * len(models_run)
    actual_fold_rows = len(fold_df)
    expected_oof_rows = n_samples * len(seeds) * len(models_run)

    print(f"\n  Fold rows:  expected={expected_fold_rows}  actual={actual_fold_rows}")
    print(f"  OOF rows:   expected={expected_oof_rows}  actual={oof_total_rows}")

    if actual_fold_rows != expected_fold_rows and not args.drop_incomplete_oof:
        raise ValueError(
            f"[{ds_name}] fold_results row mismatch: "
            f"expected {expected_fold_rows}, got {actual_fold_rows}"
        )
    if oof_total_rows != expected_oof_rows and not args.drop_incomplete_oof:
        raise ValueError(
            f"[{ds_name}] OOF row mismatch: "
            f"expected {expected_oof_rows}, got {oof_total_rows}"
        )

    print(f"  {ds_name} OK")


def _empty_fold_row(ds_name, seed, fold_id, model_name, task,
                    n_train, n_test, n_classes, metric_names,
                    note, args) -> dict:
    row: dict = {
        "dataset_name": ds_name, "seed": seed, "fold_id": fold_id,
        "model": model_name, "task_type": task,
        "n_train": n_train, "n_test": n_test, "n_classes": n_classes,
    }
    for mk in metric_names:
        row[mk] = np.nan
    row["fit_seconds"] = np.nan
    row["pred_seconds"] = np.nan
    row["model_params_json"] = _model_params_json(model_name, args, seed)
    row["notes"] = note
    return row


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    if args.n_jobs is None:
        env_val = os.environ.get("SLURM_CPUS_PER_TASK")
        n_jobs = int(env_val) if env_val else -1
    else:
        n_jobs = args.n_jobs

    if args.datasets == "all":
        ds_names = list(REGISTRY.keys())
    else:
        ds_names = [s.strip() for s in args.datasets.split(",")]

    model_names = [s.strip() for s in args.models.split(",")]

    # Validate requested datasets
    for dn in ds_names:
        if dn not in REGISTRY:
            raise ValueError(f"Unknown dataset: {dn}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Seeds: {seeds}")
    print(f"Folds: {args.n_splits}")
    print(f"Models requested: {model_names}")
    print(f"Datasets: {ds_names}")
    print(f"n_jobs: {n_jobs}")

    for dn in ds_names:
        run_dataset(dn, REGISTRY[dn], seeds, model_names, args, n_jobs)

    print(f"\nBENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
