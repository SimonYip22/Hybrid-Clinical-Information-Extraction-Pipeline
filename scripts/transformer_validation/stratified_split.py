"""
stratified_split.py

Purpose:
    - Split the fully validated annotated dataset into train, validation,
      and test sets for transformer training.
    - Ensure balanced representation across both:
        - task (symptom_presence, intervention_performed, clinical_condition_active)
        - label (is_valid: True/False)
    - Prevent data leakage and preserve statistical integrity.

Workflow:
    1. Load validated annotated dataset (600 rows).
    2. Create stratification key combining:
        - task + is_valid
    3. Perform 2-stage stratified split:
        - Stage 1:
            Train (70%) vs Temp (30%)
        - Stage 2:
            Temp → Validation (50%) + Test (50%)
    4. Verify split sizes and distributions.
    5. Save splits to CSV files.

Outputs: data/extraction/splits/
    - train.csv (420 rows)
    - val.csv (90 rows)
    - test.csv (90 rows)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# -------------------------
# 1. Config
# -------------------------

INPUT_FILE = "data/extraction/sampling/annotation_sample_labeled.csv"

TRAIN_OUTPUT_FILE = "data/extraction/splits/train.csv"
VAL_OUTPUT_FILE = "data/extraction/splits/val.csv"
TEST_OUTPUT_FILE = "data/extraction/splits/test.csv"

RANDOM_STATE = 42

# Ensure output directory exists
Path("data/extraction/splits").mkdir(parents=True, exist_ok=True)

# -------------------------
# 2. Load Data
# -------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Loaded dataset with {len(df)} rows")

# -------------------------
# 3. Create Stratification Key
# -------------------------

# Combine task + is_valid to create a stratification key
# This ensures both task balance AND label balance are preserved
df["stratify_key"] = df["task"].astype(str) + "_" + df["is_valid"].astype(str)

# -------------------------
# 4. Stage 1 Split (Train vs Temp)
# -------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,  # 30% temp
    stratify=df["stratify_key"],
    random_state=RANDOM_STATE
)

# -------------------------
# 5. Stage 2 Split (Val vs Test)
# -------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,  # split 180 → 90 / 90
    stratify=temp_df["stratify_key"],
    random_state=RANDOM_STATE
)

# -------------------------
# 6. Drop helper column
# -------------------------

for split in [train_df, val_df, test_df]:
    split.drop(columns=["stratify_key"], inplace=True) # inplace=True modifies the DataFrame directly (no copy created)

# -------------------------
# 7. Reset indices (clean datasets)
# -------------------------

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# -------------------------
# 8. Verification
# -------------------------

# Verify sizes
print("\n=== SPLIT SIZES ===")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# Verify distributions
def check_distribution(name, data):
    print(f"\n=== {name.upper()} DISTRIBUTION ===")
    print("\nTask distribution:")
    print(data["task"].value_counts())

    print("\nis_valid distribution:")
    print(data["is_valid"].value_counts())

    print("\nTask vs is_valid:")
    print(pd.crosstab(data["task"], data["is_valid"])) # cross-tab to show distribution of is_valid within each task

check_distribution("Train", train_df)
check_distribution("Validation", val_df)
check_distribution("Test", test_df)

# -------------------------
# 9. Save Outputs
# -------------------------

train_df.to_csv(TRAIN_OUTPUT_FILE, index=False) 
val_df.to_csv(VAL_OUTPUT_FILE, index=False)
test_df.to_csv(TEST_OUTPUT_FILE, index=False)

print("\nSplits saved successfully.")

