import pandas as pd

CHANGE_PATH = "data/change-metrics.csv"
CK_PATH     = "data/single-version-ck-oo.csv"
OUT_PATH    = "data/equinox_merged_32feat_plus_bugs.csv"

BUG_META_COLS = {"bugs", "nonTrivialBugs", "majorBugs", "criticalBugs", "highPriorityBugs"}

def read_semicolon_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c != ""]]
    if "classname" not in df.columns:
        raise ValueError(f"'classname' column not found in {path}. Columns: {df.columns.tolist()}")
    df["classname"] = df["classname"].astype(str).str.strip()
    return df

df_change = read_semicolon_csv(CHANGE_PATH)
df_ck     = read_semicolon_csv(CK_PATH)

change_feat_cols = [c for c in df_change.columns if c not in ({"classname"} | BUG_META_COLS)]
ck_feat_cols     = [c for c in df_ck.columns     if c not in ({"classname"} | BUG_META_COLS)]

merged = df_change[["classname", "bugs"] + change_feat_cols].merge(
    df_ck[["classname"] + ck_feat_cols],
    on="classname",
    how="inner",
    validate="one_to_one"
)

feature_cols = change_feat_cols + ck_feat_cols
for c in feature_cols + ["bugs"]:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")

n_rows = merged.shape[0]
n_features = len(feature_cols)

if n_rows != 324:
    raise ValueError(f"Expected 324 rows after merge, got {n_rows}. Check join key / missing classes.")

if n_features != 32:
    raise ValueError(
        f"Expected 32 features (15 change + 17 CK/OO), got {n_features}.\n"
        f"Change features: {len(change_feat_cols)}; CK/OO features: {len(ck_feat_cols)}.\n"
        f"Change feat cols: {change_feat_cols}\n"
        f"CK/OO feat cols: {ck_feat_cols}"
    )

final_cols = feature_cols + ["bugs"]
final = merged[final_cols].copy()

final.to_csv(OUT_PATH, index=False)
print(f"Saved merged dataset to: {OUT_PATH}")
print("Shape:", final.shape)
