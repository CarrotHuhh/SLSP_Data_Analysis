import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_poisson_deviance

# read train/test data
train_df = pd.read_csv("../Data/train_data.csv")
test_df  = pd.read_csv("../Data/test_data.csv")

X_cols = [c for c in train_df.columns if c != "bugs"]

X_train = train_df[X_cols]
y_train = train_df["bugs"]

X_test  = test_df[X_cols]
y_test  = test_df["bugs"]


def poisson_aic(X, y):
    X_ = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X_, family=sm.families.Poisson())
    res = model.fit()
    return res.aic, res

def forward_stepwise_poisson(X, y):
    remaining = list(X.columns)
    selected = []
    current_aic = np.inf
    best_model = None

    while remaining:
        aic_candidates = []
        for c in remaining:
            aic, _ = poisson_aic(X[selected + [c]], y)
            aic_candidates.append((aic, c))

        aic_candidates.sort()
        best_new_aic, best_candidate = aic_candidates[0]

        if best_new_aic < current_aic:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_aic = best_new_aic
            _, best_model = poisson_aic(X[selected], y)
        else:
            break

    return selected, best_model

# stepwise on training set
stepwise_features, stepwise_model = forward_stepwise_poisson(X_train, y_train)

print("Selected features (stepwise):")
print(stepwise_features)
print(stepwise_model.summary())

# evaluation on test set

X_test_step = sm.add_constant(X_test[stepwise_features], has_constant="add")
y_pred_step = stepwise_model.predict(X_test_step)

stepwise_deviance = mean_poisson_deviance(y_test, y_pred_step)
print("Stepwise test Poisson deviance:", stepwise_deviance)