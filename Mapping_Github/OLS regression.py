import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt

# Read the dataset (replace with actual input file path)
input_path = 'path_to_your_data/Input_regression_1.xlsx'
output_path = 'path_to_output_folder/model_results.xlsx'
df = pd.read_excel(input_path)

# Define the predictor variables
# Note: The predictors depend on the specific predictor set you are using.
predictors = ['WB_Faces_2', 'WB_Faces_4', 'WB_Faces_6', 'WB_Faces_8', 'WB_Faces_10']
X = df[predictors]
y = df['OBSERVED UTILITY']

# Add a column to store predicted utilities
df['PREDICTED UTILITY'] = np.nan

# Prepare 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results per model
model_metrics = []
coefficient_data = []

# Function for bootstrapping metrics (MAE, MSE, RMSE)
def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000):
    mae_bootstrap = []
    mse_bootstrap = []
    rmse_bootstrap = []

    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        y_true_bs = np.array(y_true)[indices]
        y_pred_bs = np.array(y_pred)[indices]

        # Calculate bootstrapped metrics
        mae_bs = mean_absolute_error(y_true_bs, y_pred_bs)
        mse_bs = mean_squared_error(y_true_bs, y_pred_bs)
        rmse_bs = np.sqrt(mse_bs)

        mae_bootstrap.append(mae_bs)
        mse_bootstrap.append(mse_bs)
        rmse_bootstrap.append(rmse_bs)

    # Calculate mean and 95% confidence intervals
    mae_mean, mse_mean, rmse_mean = np.mean(mae_bootstrap), np.mean(mse_bootstrap), np.mean(rmse_bootstrap)
    mae_ci = np.percentile(mae_bootstrap, [2.5, 97.5])
    mse_ci = np.percentile(mse_bootstrap, [2.5, 97.5])
    rmse_ci = np.percentile(rmse_bootstrap, [2.5, 97.5])

    return mae_mean, mae_ci, mse_mean, mse_ci, rmse_mean, rmse_ci

# Iterate over the 5 folds
fold = 1
for train_index, val_index in kf.split(X):
    # Train and validation split
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Add a constant (intercept) for the OLS model
    X_train = sm.add_constant(X_train)
    X_val = sm.add_constant(X_val)

    # Train OLS model on the training data
    model = sm.OLS(y_train, X_train).fit()

    # Predict utilities for the validation set
    y_pred = model.predict(X_val)

    # Store the predictions back in the original DataFrame
    df.loc[val_index, 'PREDICTED UTILITY'] = y_pred

    # Calculate model fit metrics (AIC, BIC)
    aic = model.aic
    bic = model.bic

    # Calculate predictive accuracy metrics (MAE, MSE, RMSE)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    # Bootstrapping to compute confidence intervals
    mae_mean, mae_ci, mse_mean, mse_ci, rmse_mean, rmse_ci = bootstrap_metrics(y_val, y_pred)

    # Utility statistics (mean, min, max)
    mean_utility = np.mean(y_pred)
    min_utility = np.min(y_pred)
    max_utility = np.max(y_pred)

    # Store results for this fold
    model_metrics.append({
        'Fold': fold,
        'AIC': aic,
        'BIC': bic,
        'MAE Mean': mae_mean,
        'MAE 95% CI Lower': mae_ci[0],
        'MAE 95% CI Upper': mae_ci[1],
        'MSE Mean': mse_mean,
        'MSE 95% CI Lower': mse_ci[0],
        'MSE 95% CI Upper': mse_ci[1],
        'RMSE Mean': rmse_mean,
        'RMSE 95% CI Lower': rmse_ci[0],
        'RMSE 95% CI Upper': rmse_ci[1],
        'Mean Utility': mean_utility,
        'Min Utility': min_utility,
        'Max Utility': max_utility
    })

    # Collect coefficients and standard errors
    coefs = model.params
    std_errors = model.bse
    coef_df = pd.DataFrame({
        'Fold': fold,
        'Predictor': coefs.index,
        'Coefficient': coefs.values,
        'Standard Error': std_errors.values
    })
    coefficient_data.append(coef_df)

    # Confirmation message for each fold
    print(f"Fold {fold}: Metrics calculated and stored.")
    fold += 1

# Save the DataFrame with predictions to Excel
df.to_excel(input_path, index=False)

# Convert model metrics to DataFrame
results_df = pd.DataFrame(model_metrics)

# Calculate average results across the 5 folds
mean_results = pd.DataFrame([results_df.mean(numeric_only=True).to_dict()])
mean_results['Fold'] = 'Average'

# Combine individual fold results with average results
results_df = pd.concat([results_df, mean_results], ignore_index=True)

# Save the results to an Excel file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_excel(output_path, index=False)

# Save coefficients and standard errors
coef_df = pd.concat(coefficient_data, ignore_index=True)
coef_df_mean = coef_df.groupby('Predictor').agg(
    {'Coefficient': 'mean', 'Standard Error': 'mean'}).reset_index()
coef_df_mean['Fold'] = 'Average'

# Combine coefficients and standard error results
coef_df = pd.concat([coef_df, coef_df_mean], ignore_index=True)

# Save coefficients and standard errors to Excel (replace path with general)
coef_output_path = 'path_to_output_folder/coefficient_results.xlsx'
coef_df.to_excel(coef_output_path, index=False)

print(f"All results are saved in the Excel file: {output_path}.")
print(f"Coefficients and standard errors are saved in the Excel file: {coef_output_path}.")
