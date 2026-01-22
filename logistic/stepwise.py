import statsmodels.api as sm
from sklearn.metrics import accuracy_score
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


def logit_aic(X, y):
    X_ = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_)
    res = model.fit(disp=False)
    return res.aic, res

def forward_stepwise_logit(X, y):
    remaining = list(X.columns)
    selected = []
    current_aic = np.inf
    best_model = None

    while remaining:
        aic_candidates = []
        for c in remaining:
            try:
                aic, _ = logit_aic(X[selected + [c]], y)
                aic_candidates.append((aic, c))
            except:
                pass

        if not aic_candidates:
            break

        aic_candidates.sort()
        best_new_aic, best_candidate = aic_candidates[0]

        if best_new_aic < current_aic:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_aic = best_new_aic
            _, best_model = logit_aic(X[selected], y)
        else:
            break

    return selected, best_model
stepwise_features, stepwise_model = forward_stepwise_logit(X_train, y_train)

print("Selected features (logistic stepwise):")
print(stepwise_features)
print(stepwise_model.summary())

X_test_step = sm.add_constant(X_test[stepwise_features], has_constant="add")
y_pred_prob = stepwise_model.predict(X_test_step)
y_pred = (y_pred_prob >= 0.5).astype(int)

stepwise_acc = accuracy_score(y_test, y_pred)
print("Stepwise test accuracy:", stepwise_acc)
