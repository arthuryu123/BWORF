# BRFSS 2022 Data Acquisition

The BRFSS 2022 raw CSV file is not committed to this repository because of its size (~250 MB) and its source licensing.

## How to obtain

The dataset is publicly available from Kaggle:

https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

Download the `heart_2022_no_nans.csv` variant. Verify the file integrity:

```bash
sha256sum heart_2022_no_nans.csv
# Expected: f5eddf47d85170f2f7bc3ba523c275d705ddf22518f95d50aa33ddcfd786a94f
```

Place the file in this directory:
data/heart_2022_no_nans.csv

## Original source

The Kaggle dataset is derived from the U.S. Centers for Disease Control and Prevention's Behavioral Risk Factor Surveillance System (BRFSS) 2022 survey. The original CDC data is in the public domain.

## Preparation

Run the preparation script (in `../code/`) to generate the cleaned subsample and full-N modeling/holdout splits:

```bash
cd ..
python code/prepare_heart2022.py
```

This writes the cleaned outputs to `data/cleaned/`.
