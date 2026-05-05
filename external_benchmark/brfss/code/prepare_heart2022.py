"""
prepare_heart2022.py — standalone preparation for the BRFSS 2022 Heart Attack
external benchmark addition.

Reads the raw `heart_2022_no_nans.csv` from Kaggle (Kamil Pytlak,
"Personal Key Indicators of Heart Disease", BRFSS 2022 survey) and produces
two cleaned, gzip-compressed CSVs in `data/cleaned/`:

    heart_attack_2022_brfss.csv.gz          # modeling set (10x5 CV here)
    heart_attack_2022_brfss_holdout.csv.gz  # one-shot confirmatory holdout

Plus an audit JSON in `reports/heart_attack_2022_brfss_audit.json`.

ENCODING POLICY (rationale documented for reviewers):

    Same encoding is applied uniformly to features that go into every model.
    Encoding choices respect the structure of the underlying variable; they
    are not tuned per model.

    DROP:
        * State (54 levels). Geographic state is not a clinical risk
          factor for heart attack and acts as a high-cardinality confounder
          for healthcare access / demographics already captured by other
          variables. Including it would inflate the feature space by 53
          dummy columns.

    BINARY (Yes/No -> 1/0), 19 columns:
        PhysicalActivities, HadAngina, HadStroke, HadAsthma, HadSkinCancer,
        HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis,
        DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating,
        DifficultyWalking, DifficultyDressingBathing, DifficultyErrands,
        ChestScan, AlcoholDrinkers, HIVTesting, FluVaxLast12, PneumoVaxEver,
        HighRiskLastYear

    BINARY (Female/Male -> 0/1):
        Sex

    ORDINAL (real ordering preserved as integers):
        GeneralHealth   (5):  Poor < Fair < Good < Very good < Excellent     -> 0..4
        AgeCategory     (13): "Age 18 to 24" .. "Age 80 or older"            -> 0..12
        LastCheckupTime (4):  "Within past year" < "past 2 years"
                              < "past 5 years" < "5 or more years ago"        -> 0..3
                              (0 = most recent checkup)
        RemovedTeeth    (4):  "None of them" < "1 to 5"
                              < "6 or more, but not all" < "All"              -> 0..3
        SmokerStatus    (4):  Never < Former < Current some days
                              < Current every day                             -> 0..3
        ECigaretteUsage (4):  Never used < Not at all (right now)
                              < Use some days < Use every day                 -> 0..3
        HadDiabetes     (4):  No < No, pre-diabetes
                              < Yes, only during pregnancy < Yes              -> 0..3

    NOMINAL (one-hot in the runner via categorical_cols):
        RaceEthnicityCategory (5 levels)
        TetanusLast10Tdap     (4 levels) - shot types, no real order
        CovidPos              (3 levels) - "No" / "Tested positive home" / "Yes"

    NUMERIC (kept as-is):
        BMI, HeightInMeters, WeightInKilograms,
        PhysicalHealthDays, MentalHealthDays, SleepHours

    TARGET:
        HadHeartAttack: Yes/No -> 1/0  (positive rate ~5.46%)

DEDUP / VALIDATION:
    * Drops exact duplicate rows (consistent with prepare_data.py policy).
    * Validates required columns and exact category levels; fails loudly
      if Kaggle ever changes the schema.

SUBSAMPLE / HOLDOUT:
    1. Stratified subsample to --subsample_n rows (default 25_000), preserving
       the ~5.46% positive prevalence. Set --subsample_n 0 for full N.
    2. Stratified 80/20 split of the (sub)sample into modeling + holdout.
       Modeling set is what 10-fold x 5-seed CV runs on.
       Holdout is touched once, post-CV, for confirmatory scoring.

    Both splits use --prep_seed (default 20260425), distinct from the CV
    seeds {13, 42, 77, 123, 2025} so the holdout is independent of anything
    the CV does.

USAGE on Sol:

    cd ~/external_benchmark
    # Subsample run (deadline-safe):
    python prepare_heart2022.py \\
        --raw_csv data/heart_2022_no_nans.csv \\
        --out_dir . \\
        --subsample_n 25000 \\
        --prep_seed 20260425

    # Full-N run (regenerate with same prep_seed for the larger experiment):
    python prepare_heart2022.py \\
        --raw_csv data/heart_2022_no_nans.csv \\
        --out_dir . \\
        --subsample_n 0 \\
        --prep_seed 20260425 \\
        --output_suffix _full
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Encoding maps (ordered; index = encoded value)
# ---------------------------------------------------------------------------
GEN_HEALTH_ORDER = ["Poor", "Fair", "Good", "Very good", "Excellent"]

AGE_ORDER = [
    "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
    "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
    "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
    "Age 80 or older",
]

# 0 = most recent checkup, 3 = oldest
LAST_CHECKUP_ORDER = [
    "Within past year (anytime less than 12 months ago)",
    "Within past 2 years (1 year but less than 2 years ago)",
    "Within past 5 years (2 years but less than 5 years ago)",
    "5 or more years ago",
]

REMOVED_TEETH_ORDER = [
    "None of them",
    "1 to 5",
    "6 or more, but not all",
    "All",
]

SMOKER_ORDER = [
    "Never smoked",
    "Former smoker",
    "Current smoker - now smokes some days",
    "Current smoker - now smokes every day",
]

ECIGARETTE_ORDER = [
    "Never used e-cigarettes in my entire life",
    "Not at all (right now)",
    "Use them some days",
    "Use them every day",
]

HAD_DIABETES_ORDER = [
    "No",
    "No, pre-diabetes or borderline diabetes",
    "Yes, but only during pregnancy (female)",
    "Yes",
]

YESNO_COLS = [
    "PhysicalActivities", "HadAngina", "HadStroke", "HadAsthma",
    "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease",
    "HadArthritis", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
    "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
    "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting",
    "FluVaxLast12", "PneumoVaxEver", "HighRiskLastYear",
]

NOMINAL_KEEP_AS_STR = [
    "RaceEthnicityCategory",  # 5 levels
    "TetanusLast10Tdap",      # 4 levels (shot types, not ordered)
    "CovidPos",               # 3 levels
]

NUMERIC_COLS = [
    "BMI", "HeightInMeters", "WeightInKilograms",
    "PhysicalHealthDays", "MentalHealthDays", "SleepHours",
]

TARGET_COL = "HadHeartAttack"

DROP_COLS = ["State"]  # high-cardinality nuisance, see encoding policy

DATASET_NAME = "heart_attack_2022_brfss"
SOURCE = (
    "Kaggle: kamilpytlak/personal-key-indicators-of-heart-disease "
    "(BRFSS 2022 Annual Survey, CDC; heart_2022_no_nans.csv)"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _ordinal_map(order: list[str]) -> dict[str, int]:
    return {label: i for i, label in enumerate(order)}


def _validate_levels(s: pd.Series, expected: list[str], col: str) -> None:
    actual = set(s.dropna().unique())
    expected_set = set(expected)
    if actual != expected_set:
        missing = expected_set - actual
        unexpected = actual - expected_set
        msg = f"{col}: levels do not match expected schema."
        if missing:
            msg += f" Missing in data: {sorted(missing)}."
        if unexpected:
            msg += f" Unexpected in data: {sorted(unexpected)}."
        raise ValueError(msg)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic encoding. Returns (encoded_df, encoding_info)."""
    info: dict = {}

    # Target
    _validate_levels(df[TARGET_COL], ["No", "Yes"], TARGET_COL)
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(np.int8)

    # Yes/No predictors
    for col in YESNO_COLS:
        _validate_levels(df[col], ["No", "Yes"], col)
        df[col] = (df[col] == "Yes").astype(np.int8)
    info["yes_no_to_01"] = list(YESNO_COLS)

    # Sex
    _validate_levels(df["Sex"], ["Female", "Male"], "Sex")
    df["Sex"] = (df["Sex"] == "Male").astype(np.int8)
    info["sex_map"] = {"Female": 0, "Male": 1}

    # Ordinal columns
    ordinal_specs = [
        ("GeneralHealth",   GEN_HEALTH_ORDER,       "general_health_ordinal_map"),
        ("AgeCategory",     AGE_ORDER,              "age_category_ordinal_map"),
        ("LastCheckupTime", LAST_CHECKUP_ORDER,     "last_checkup_ordinal_map"),
        ("RemovedTeeth",    REMOVED_TEETH_ORDER,    "removed_teeth_ordinal_map"),
        ("SmokerStatus",    SMOKER_ORDER,           "smoker_status_ordinal_map"),
        ("ECigaretteUsage", ECIGARETTE_ORDER,       "ecigarette_usage_ordinal_map"),
        ("HadDiabetes",     HAD_DIABETES_ORDER,     "had_diabetes_ordinal_map"),
    ]
    for col, order, info_key in ordinal_specs:
        _validate_levels(df[col], order, col)
        m = _ordinal_map(order)
        df[col] = df[col].map(m).astype(np.int8)
        info[info_key] = m

    info["nominal_kept_as_str"] = list(NOMINAL_KEEP_AS_STR)

    return df, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare BRFSS 2022 heart attack dataset")
    p.add_argument("--raw_csv", required=True,
                   help="Path to raw heart_2022_no_nans.csv")
    p.add_argument("--out_dir", default=".",
                   help="Project root; cleaned CSVs go to <out_dir>/data/cleaned/, "
                        "audit JSON to <out_dir>/reports/")
    p.add_argument("--subsample_n", type=int, default=25000,
                   help="Stratified subsample size before holdout split. "
                        "Set 0 to disable subsampling (use full N).")
    p.add_argument("--holdout_frac", type=float, default=0.20,
                   help="Stratified holdout fraction (default 0.20)")
    p.add_argument("--prep_seed", type=int, default=20260425,
                   help="Seed for subsample + holdout split. MUST be distinct "
                        "from the CV seeds {13,42,77,123,2025}.")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Optional suffix appended to output basenames "
                        "(e.g. '_full' to coexist with a subsample run).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.prep_seed in {13, 42, 77, 123, 2025}:
        raise ValueError(
            "prep_seed collides with CV seeds; use a distinct value "
            "to keep the holdout split independent."
        )

    raw_path = Path(args.raw_csv)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    out_dir = Path(args.out_dir)
    cleaned_dir = out_dir / "data" / "cleaned"
    reports_dir = out_dir / "reports"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    raw_sha = sha256_file(raw_path)
    raw_size = raw_path.stat().st_size

    # ---- Load raw ----
    df_raw = pd.read_csv(raw_path)
    n_raw = len(df_raw)

    # ---- Drop columns we deliberately exclude ----
    for col in DROP_COLS:
        if col not in df_raw.columns:
            raise ValueError(f"Expected drop-column '{col}' not found in raw CSV")
    df = df_raw.drop(columns=DROP_COLS).copy()

    # ---- Drop exact duplicate rows ----
    dup_mask = df.duplicated(keep="first")
    n_duplicates = int(dup_mask.sum())
    df = df.loc[~dup_mask].reset_index(drop=True)

    # ---- Validate required columns ----
    expected = (
        {TARGET_COL, "Sex"}
        | set(YESNO_COLS)
        | set(NOMINAL_KEEP_AS_STR)
        | set(NUMERIC_COLS)
        | {"GeneralHealth", "AgeCategory", "LastCheckupTime",
           "RemovedTeeth", "SmokerStatus", "ECigaretteUsage", "HadDiabetes"}
    )
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    extra = set(df.columns) - expected
    if extra:
        raise ValueError(
            f"Unexpected extra columns after dropping {DROP_COLS}: {sorted(extra)}. "
            "Schema may have changed; refusing to proceed."
        )

    # ---- Validate no NaNs ----
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"Unexpected NaNs in raw 'no_nans' file, columns: {nan_cols}"
        )

    # ---- Encode ----
    df, encoding_info = encode(df)
    n_after_encode = len(df)

    # ---- Subsample (stratified by target) ----
    if args.subsample_n and args.subsample_n < len(df):
        n_sub = args.subsample_n
        df_sub, _ = train_test_split(
            df, train_size=n_sub,
            stratify=df[TARGET_COL].values,
            random_state=args.prep_seed,
        )
        df_sub = df_sub.reset_index(drop=True)
    else:
        n_sub = len(df)
        df_sub = df

    # ---- Stratified holdout split ----
    df_model, df_hold = train_test_split(
        df_sub, test_size=args.holdout_frac,
        stratify=df_sub[TARGET_COL].values,
        random_state=args.prep_seed,
    )
    df_model = df_model.reset_index(drop=True)
    df_hold = df_hold.reset_index(drop=True)

    # ---- Write cleaned CSVs ----
    suffix = args.output_suffix
    model_path = cleaned_dir / f"{DATASET_NAME}{suffix}.csv.gz"
    hold_path = cleaned_dir / f"{DATASET_NAME}{suffix}_holdout.csv.gz"
    df_model.to_csv(model_path, index=False, compression="gzip")
    df_hold.to_csv(hold_path, index=False, compression="gzip")

    # ---- Audit JSON ----
    def class_counts(s: pd.Series) -> dict:
        return {str(int(k)): int(v) for k, v in s.value_counts().items()}

    audit = {
        "dataset_name": DATASET_NAME + suffix,
        "source": SOURCE,
        "raw_file": {
            "path": str(raw_path),
            "sha256": raw_sha,
            "size_bytes": raw_size,
        },
        "raw_n_rows": int(n_raw),
        "dropped_columns": DROP_COLS,
        "duplicates_dropped": n_duplicates,
        "n_rows_after_encode": int(n_after_encode),
        "subsample": {
            "applied": bool(args.subsample_n and args.subsample_n < n_after_encode),
            "subsample_n": int(n_sub),
            "prep_seed": args.prep_seed,
        },
        "holdout_split": {
            "modeling_n": int(len(df_model)),
            "holdout_n": int(len(df_hold)),
            "holdout_frac": args.holdout_frac,
            "prep_seed": args.prep_seed,
        },
        "class_distribution_modeling": class_counts(df_model[TARGET_COL]),
        "class_distribution_holdout": class_counts(df_hold[TARGET_COL]),
        "modeling_positive_rate": float(df_model[TARGET_COL].mean()),
        "holdout_positive_rate": float(df_hold[TARGET_COL].mean()),
        "target_col": TARGET_COL,
        "task_type": "binary",
        "encoding": encoding_info,
        "categorical_cols_for_runner": NOMINAL_KEEP_AS_STR,
        "n_features": int(df_model.shape[1] - 1),
        "feature_columns": [c for c in df_model.columns if c != TARGET_COL],
        "modeling_csv": str(model_path),
        "holdout_csv": str(hold_path),
        "timestamp": now,
        "script_args": vars(args),
    }
    audit_path = reports_dir / f"{DATASET_NAME}{suffix}_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # ---- Console summary ----
    print(f"Raw rows:           {n_raw}")
    print(f"After drop({DROP_COLS}) + dedup: "
          f"{n_after_encode}  ({n_duplicates} duplicates dropped)")
    if args.subsample_n and args.subsample_n < n_after_encode:
        print(f"Subsample:          {n_sub}  (stratified, prep_seed={args.prep_seed})")
    else:
        print(f"Subsample:          (none, using full N = {n_sub})")
    print(f"Modeling set:       {len(df_model)} rows  -> {model_path}")
    print(f"Holdout set:        {len(df_hold)} rows  -> {hold_path}")
    print(f"Audit:              {audit_path}")
    print(f"Modeling positives: {int(df_model[TARGET_COL].sum())} "
          f"({df_model[TARGET_COL].mean()*100:.2f}%)")
    print(f"Holdout positives:  {int(df_hold[TARGET_COL].sum())} "
          f"({df_hold[TARGET_COL].mean()*100:.2f}%)")
    print(f"Final feature count (before one-hot): "
          f"{df_model.shape[1] - 1}")


if __name__ == "__main__":
    main()
