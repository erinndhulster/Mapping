import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
import matplotlib.pyplot as plt

# Read the dataset (replace with the actual file path)
input_path = 'path_to_your_data/Input_randomforest_1.xlsx'
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

# Lists to store results for each fold
model_metrics = []

# Function to perform bootstrapping for calculating metrics (MAE, MSE, RMSE)
def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000):
    mae_bootstrap = []
    mse_bootstrap = []
    rmse_bootstrap = []

    n = len(y_true)
    for _ in range(n_bootstraps):
        # Sample with replacement
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

    # Calculate the mean and 95% confidence intervals
    mae_mean = np.mean(mae_bootstrap)
    mse_mean = np.mean(mse_bootstrap)
    rmse_mean = np.mean(rmse_bootstrap)
    mae_ci = np.percentile(mae_bootstrap, [2.5, 97.5])
    mse_ci = np.percentile(mse_bootstrap, [2.5, 97.5])
    rmse_ci = np.percentile(rmse_bootstrap, [2.5, 97.5])

    return mae_mean, mae_ci, mse_mean, mse_ci, rmse_mean, rmse_ci

# Iterate over the 5 folds
fold = 1
for train_index, val_index in kf.split(X):
    # Split data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the Random Forest model on the training data
    rf_regressor.fit(X_train, y_train)

    # Predict utilities for the validation set
    y_pred = rf_regressor.predict(X_val)

    # Write the predictions back to the original DataFrame
    df.loc[val_index, 'PREDICTED UTILITY'] = y_pred

    # Calculate predictive accuracy metrics (MAE, MSE, RMSE)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    # Perform bootstrapping to calculate confidence intervals
    mae_mean, mae_ci, mse_mean, mse_ci, rmse_mean, rmse_ci = bootstrap_metrics(y_val, y_pred)

    # Utility statistics (mean, min, max)
    mean_utility = np.mean(y_pred)
    min_utility = np.min(y_pred)
    max_utility = np.max(y_pred)

    # Store results for this fold
    model_metrics.append({
        'Fold': fold,
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

    # Random Forest does not provide coefficients, so skip that part
    print(f"Fold {fold}: Metrics calculated and stored.")
    fold += 1

# Save the DataFrame with predictions to Excel
df.to_excel(input_path, index=False)

# Convert the model metrics to a DataFrame
results_df = pd.DataFrame(model_metrics)

# Calculate the average results across the 5 folds
mean_results = pd.DataFrame([results_df.mean(numeric_only=True).to_dict()])
mean_results['Fold'] = 'Average'

# Combine the individual fold results with the average results
results_df = pd.concat([results_df, mean_results], ignore_index=True)

# Save the results to Excel
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_excel(output_path, index=False)

print(f"All results are saved in the Excel file: {output_path}.")
