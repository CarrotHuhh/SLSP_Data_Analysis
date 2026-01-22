from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

train_df = pd.read_csv("../Data/train_data.csv")
test_df  = pd.read_csv("../Data/test_data.csv")

train_df["bugbin"] = (train_df["bugs"] >= 1).astype(int)
test_df["bugbin"]  = (test_df["bugs"] >= 1).astype(int)

X_cols = [c for c in train_df.columns if c not in ["bugs", "bugbin"]]

X_train = train_df[X_cols]
y_train = train_df["bugbin"]

X_test  = test_df[X_cols]
y_test  = test_df["bugbin"]

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

Cs = np.logspace(-2, 2, 50)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_C = None
best_score = -np.inf

for C in Cs:
    scores = []
    for tr_idx, val_idx in kf.split(Xtr):
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            max_iter=5000
        )
        model.fit(Xtr[tr_idx], y_train.iloc[tr_idx])
        score = model.score(Xtr[val_idx], y_train.iloc[val_idx])
        scores.append(score)

    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_C = C

print("Selected C:", best_C)
lasso_logit = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=best_C,
    max_iter=5000
)
lasso_logit.fit(Xtr, y_train)
y_test_pred = lasso_logit.predict(Xte)
lasso_acc = accuracy_score(y_test, y_test_pred)

print("LASSO test accuracy:", lasso_acc)
coef = lasso_logit.coef_[0]
selected_features = [X_cols[i] for i in range(len(X_cols)) if abs(coef[i]) > 1e-6]

print("Selected features (logistic LASSO):")
print(selected_features)
print(lasso_logit.coef_)
