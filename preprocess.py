import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "Data/equinox_merged_32feat_plus_bugs.csv"

df = pd.read_csv(DATA_PATH)

bins = [-1, 0, 2, 5, np.inf]
labels = ["0", "1-2", "3-5", "6+"]

df["bug_bin"] = pd.cut(
    df["bugs"],
    bins=bins,
    labels=labels
)

train_df, test_df = train_test_split(
    df,
    test_size=0.33,
    stratify=df["bug_bin"],
    random_state=42
)
train_df = train_df.drop(columns=["bug_bin"])
test_df  = test_df.drop(columns=["bug_bin"])

# 输出路径
TRAIN_PATH = "Data/train_data.csv"
TEST_PATH  = "Data/test_data.csv"

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("Train and test datasets saved.")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
