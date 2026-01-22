import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_poisson_deviance, accuracy_score
from sklearn.preprocessing import StandardScaler
from pyglmnet import GLM

train_df = pd.read_csv("../Data/train_data.csv")
test_df  = pd.read_csv("../Data/test_data.csv")

X_cols = [c for c in train_df.columns if c != "bugs"]

X_train = train_df[X_cols].values
y_train = train_df["bugs"].values

X_test  = test_df[X_cols].values
y_test  = test_df["bugs"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

lambdas = np.logspace(-4, 1, 60)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_lambda = None
best_score = np.inf

for lam in lambdas:
    fold_scores = []
    ok = True

    for tr_idx, val_idx in kf.split(X_train):
        model = GLM(
            distr="poisson",
            alpha=1.0,
            reg_lambda=lam,
            max_iter=20000
        )
        model.fit(X_train[tr_idx], y_train[tr_idx])
        y_val_pred = model.predict(X_train[val_idx])
        fold_scores.append(mean_poisson_deviance(y_train[val_idx], y_val_pred))

    if ok:
        avg_score = float(np.mean(fold_scores))
        if avg_score < best_score:
            best_score = avg_score
            best_lambda = lam

print("Selected lambda:", best_lambda)

lasso_model = GLM(
    distr="poisson",
    alpha=1.0,
    reg_lambda=best_lambda,
    max_iter=20000
)
lasso_model.fit(X_train, y_train)

y_test_pred = lasso_model.predict(X_test)
lasso_test_deviance = mean_poisson_deviance(y_test, y_test_pred)

print("LASSO test Poisson deviance:", lasso_test_deviance)

coef = lasso_model.beta_
selected_features = [X_cols[i] for i in range(len(X_cols)) if abs(coef[i]) > 1e-4]
print("Selected features (LASSO):")
print(selected_features)
print(lasso_model.beta_)
