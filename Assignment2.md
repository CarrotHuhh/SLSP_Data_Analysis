# Part B: Data Analysis
## Data preprocessing and experimental setup
We combined two Equinox datasets containing change metrics and CK/OO metrics and used the class identifier `classname` as the join key. Defect-related metadata columns were removed from the feature set, while `bugs` was retained as the target variable.

We merged two datasets using a one-to-one inner join on `classname`. After merging, all feature columns and the target were converted to numeric values. The final dataset contains **324 classes**, **32 features**, and one target variable.

The dataset was split into a training set (67%) and a test set (33%). To preserve the distribution of bug counts, we used stratified random sampling based on the predefined bug-count bins (0, 1–2, 3–5, 6+). The stratification variable was used only for splitting and removed afterwards.

The resulting training and test sets were saved and used consistently in all analyses. Models were selected on the training set and the final selected model was evaluated on the test set.
## Poisson regression

### Stepwise model selection

We applied Poisson regression to model the number of bugs. Since the response variable is count data, a generalized linear model with Poisson family and log link was used. Model selection was performed on the training set using **forward stepwise selection**, starting from an intercept-only model and using **AIC** as the selection criterion.

The final model selected by stepwise regression includes the following five predictors:
$$
\text{bugs} \sim cbo + numberOfMethodsInherited + weightedAgeWithRespectTo + avgLinesAddedUntil + noc
$$


The table below shows the estimated coefficients of the selected Poisson regression model, together with standard errors, z-statistics, p-values, and 95% confidence intervals.

| Variable | Coefficient | Std. Error | z | p-value | 95% CI |
|--------|-------------|------------|----|---------|--------|
| Intercept | -1.9665 | 0.189 | -10.388 | < 0.001 | [-2.337, -1.595] |
| cbo | 0.0454 | 0.004 | 12.728 | < 0.001 | [0.038, 0.052] |
| numberOfMethodsInherited | 0.0167 | 0.003 | 5.249 | < 0.001 | [0.010, 0.023] |
| weightedAgeWithRespectTo | 0.0117 | 0.002 | 5.241 | < 0.001 | [0.007, 0.016] |
| avgLinesAddedUntil | 0.0084 | 0.003 | 3.097 | 0.002 | [0.003, 0.014] |
| noc | 0.1789 | 0.092 | 1.945 | 0.052 | [-0.001, 0.359] |

All selected predictors have positive coefficients, indicating that higher structural complexity or more extensive code evolution is associated with a higher expected number of bugs. Most variables are statistically significant at the 5% level, while `noc` shows a borderline effect.


The predictive performance of the selected model was evaluated on the test set using Poisson deviance:

- **Deviance:** 0.81

### LASSO regularization

To model the number of bugs, we applied Poisson regression with L1 (LASSO) regularization. All predictors were standardized prior to model fitting, which is necessary for LASSO to penalize coefficients in a comparable way. The regularization parameter $ \lambda $ was selected on the training set using 5-fold cross-validation, with **Poisson deviance** as the evaluation criterion.

Cross-validation selected the following value:

- **Selected $ \lambda $: 0.166**

This value corresponds to a relatively strong regularization effect, encouraging a sparse model.

Under the selected regularization strength, the LASSO Poisson model retained only a single non-zero coefficient:

- **Selected feature:** `cbo`

All other coefficients were shrunk exactly to zero by the L1 penalty. This indicates that, among all available predictors, `cbo` alone provides the strongest contribution to explaining variation in the number of bugs when model sparsity is enforced.

The final LASSO-selected model was evaluated on the test set:

- **Deviance:** 1.279

This deviance value is higher than that obtained by the stepwise Poisson model, suggesting that while LASSO achieves a very sparse model, this comes at the cost of reduced predictive performance on unseen data.

The Poisson LASSO result shows the trade-off between **model simplicity** and **predictive accuracy**. By regularization, LASSO identifies `cbo` as the most influential feature for bug counts. The model  with single feature does not capture as much information as the model selected by stepwise regression.

### Comparison of stepwise and LASSO

The two model selection methods lead to significantly different Poisson regression models.  
The model based on stepwise selection includes five predictors and achieves a **lower test Poisson deviance (0.81)**, indicating better predictive performance on unseen data. In contrast, Poisson regression with LASSO regularization selects a much sparser model, retaining only a single predictor (`cbo`), but yields a **higher test deviance (1.279)**.
While LASSO enforces strong sparsity and identifies the most influential predictor, the resulting model does not capture sufficient information to match the performance of the stepwise-selected model. For this dataset, stepwise selection appears more suitable for Poisson regression when the goal is accurate prediction of bug counts.

## Logistic regression
### Stepwise model selection
## Logistic regression with stepwise model selection

We constructed a binary target variable `bugbin` where `bugbin = 1` if `bugs ≥ 1` and `bugbin = 0` otherwise and fitted a logistic regression model and performed **forward stepwise selection** on the training set using **AIC** as the selection score, starting from an intercept-only model.

The final stepwise-selected logistic regression model includes the following predictors:

- `cbo`, `numberOfMethodsInherited`, `weightedAgeWithRespectTo`, `numberOfRefactoringsUntil`,  
  `avgLinesRemovedUntil`, `noc`, `dit`

| Variable | Coef. | Std. Err. | z | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| Intercept | -3.8173 | 0.629 | -6.069 | < 0.001 | [-5.050, -2.585] |
| cbo | 0.1191 | 0.025 | 4.845 | < 0.001 | [0.071, 0.167] |
| numberOfMethodsInherited | 0.0296 | 0.021 | 1.440 | 0.150 | [-0.011, 0.070] |
| weightedAgeWithRespectTo | 0.0168 | 0.006 | 2.894 | 0.004 | [0.005, 0.028] |
| numberOfRefactoringsUntil | -1.3044 | 0.525 | -2.486 | 0.013 | [-2.333, -0.276] |
| avgLinesRemovedUntil | 0.0643 | 0.024 | 2.675 | 0.007 | [0.017, 0.111] |
| noc | 0.7458 | 0.345 | 2.161 | 0.031 | [0.069, 1.422] |
| dit | 0.8081 | 0.416 | 1.944 | 0.052 | [-0.007, 1.623] |

Most selected predictors have statistically significant effects at the 5% level. In particular, `cbo`, `weightedAgeWithRespectTo`, `avgLinesRemovedUntil`, and `noc` are positively associated with the probability of having bugs, while `numberOfRefactoringsUntil` shows a negative association. The effect of `dit` is borderline significant, and `numberOfMethodsInherited` is not significant in this model. The final selected model was evaluated on the test set using classification accuracy:

- **Accuracy:** 0.720

### LASSO regularization
In the `sklearn` implementation, the regularization strength is controlled by the parameter $ C $, which is the inverse of the LASSO penalty parameter ($ \lambda = 1/C $). Cross-validation selected a relatively large value of $C $, corresponding to a weak regularization. As a result, the final model retained a large amount of predictors.

The performance of the selected logistic LASSO model was evaluated on the test set using classification accuracy:

- **Accuracy:** 0.729

Although the LASSO penalty was applied, the model did not yield a highly sparse solution. This indicates that retaining multiple correlated predictors leads to better classification performance on this dataset. 
### Comparison of stepwise and LASSO
For logistic regression, stepwise selection and LASSO regularization achieved similar classification performance on the test set. The stepwise model reached a **accuracy of 0.720**, while the LASSO model achieved a slightly higher **accuracy of 0.729**.

These two methods differ in their approach to model structure. Stepwise selection results in a more compact model, utilizing a limited number of predictors, which simplifies the interpretation of individual effects. On the other hand, logistic LASSO retains a larger set of predictors due to the weak regularization strength selected by cross-validation, suggesting that including multiple correlated features improves classification accuracy in this setting.

In summary, the results reveal that for logistic regression, LASSO provides marginally better performance at the cost of reduced sparsity, whereas stepwise selection yields a more interpretable model with comparable accuracy.