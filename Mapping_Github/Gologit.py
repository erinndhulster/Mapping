import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Load the dataset
# Replace with the correct file path
file_path = 'path_to_your_data/Input_gologit_1.xlsx'
df = pd.read_excel(file_path)

# Define the predictors and dependent variables
# Note: The specific predictors depend on the predictor set being used.
predictors = ["WB_Faces_2", "WB_Faces_4", "WB_Faces_6", "WB_Faces_8", "WB_Faces_10"]
dependent_vars = ['Mobility_EQ5D-Y', 'Taking_care_of_myself_EQ5D-Y', 'Daily_activities_EQ5D-Y', 'Pain_EQ5D-Y',
                  'Feeling_sad_EQ5D-Y']

# Split the data into 5 folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to fit a Generalized Ordered Logit model (GOLOGIT)
def fit_gologit_model(X, y):
    model = OrderedModel(y, X, distr='logit')  # 'logit' for logistic link function
    result = model.fit(method='bfgs', disp=False)  # Use BFGS optimization method
    return result

# Function to perform bootstrapping for metrics (e.g., MAE, MSE, RMSE)
def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000):
    mae_bootstrap = []
    mse_bootstrap = []
    rmse_bootstrap = []

    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        y_true_bs = np.array(y_true)[indices]
        y_pred_bs = np.array(y_pred)[indices]

        mae_bs = mean_absolute_error(y_true_bs, y_pred_bs)
        mse_bs = mean_squared_error(y_true_bs, y_pred_bs)
        rmse_bs = np.sqrt(mse_bs)

        mae_bootstrap.append(mae_bs)
        mse_bootstrap.append(mse_bs)
        rmse_bootstrap.append(rmse_bs)

    # Calculate means and confidence intervals
    mae_mean, mse_mean, rmse_mean = np.mean(mae_bootstrap), np.mean(mse_bootstrap), np.mean(rmse_bootstrap)
    mae_ci = np.percentile(mae_bootstrap, [2.5, 97.5])
    mse_ci = np.percentile(mse_bootstrap, [2.5, 97.5])
    rmse_ci = np.percentile(rmse_bootstrap, [2.5, 97.5])

    return mae_mean, mae_ci, mse_mean, mse_ci, rmse_mean, rmse_ci

# Initialize empty lists to store coefficients and model metrics
coefficients_list = []
model_metrics = []

# Begin the 5-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
    print(f"Start fold {fold}")

    # Select training and validation sets
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    # Fit GOLOGIT models for each dependent variable
    models = {}
    for dv in dependent_vars:
        X_train = train_data[predictors]
        y_train = train_data[dv]

        model = fit_gologit_model(X_train, y_train)
        models[dv] = model

        # Save coefficients and standard errors
        params = model.params
        bse = model.bse
        for predictor in predictors:
            if predictor in params.index:
                coef = params[predictor]
                se = bse[predictor]
                coefficients_list.append({
                    'Fold': fold,
                    'Dependent Variable': dv,
                    'Predictor': predictor,
                    'Coefficient': coef,
                    'SE': se
                })

    # Convert the list to a DataFrame for coefficients
    coefficients_df = pd.DataFrame(coefficients_list)

    # Save the coefficients DataFrame to an Excel file (replace with correct file path)
    output_coefficients_file = 'path_to_output_folder/coefficients.xlsx'
    coefficients_df.to_excel(output_coefficients_file, index=False)

    # Predict utility for the validation sample using the models
    y_true_list = []
    y_pred_list = []
    for index in test_index:
        row = df.iloc[index]
        P = {}
        for dv in dependent_vars:
            model = models[dv]
            exog = np.array(row[predictors]).reshape(1, -1)  # Ensure data is in the correct shape

            # Handle potential data issues (convert to numeric and handle NaNs)
            exog = pd.to_numeric(exog.flatten(), errors='coerce')  # Convert to numeric, handle errors as NaN
            exog = np.nan_to_num(exog, nan=0)  # Replace NaNs with zeros

            try:
                probs = model.predict(exog.reshape(1, -1))  # Ensure exog is a 2D array
                P[dv] = probs.flatten()
            except Exception as e:
                print(f"Error predicting for index {index} and dependent variable {dv}: {e}")
                P[dv] = np.nan  # Set to NaN if prediction fails

        # Calculate predicted utility if all required values are available
        if all(pd.notna(P[dv]).all() for dv in dependent_vars):
            P_WA = P['Mobility_EQ5D-Y']
            P_WD = P['Taking_care_of_myself_EQ5D-Y']
            P_UA = P['Daily_activities_EQ5D-Y']
            P_PD = P['Pain_EQ5D-Y']
            P_WS = P['Feeling_sad_EQ5D-Y']

            # Ensure probability distributions have the correct length
            if (len(P_WA) < 3 or len(P_WD) < 3 or len(P_UA) < 3 or len(P_PD) < 3 or len(P_WS) < 3):
                print(f"Invalid probability distribution for index {index}. Skipping.")
                predicted_utility = np.nan
            else:
                # Calculate predicted utility
                predicted_utility = 1 - 0.064 * P_WA[1] - 0.203 * P_WA[2] - 0.046 * P_WD[1] - 0.174 * P_WD[2] \
                                    - 0.104 * P_UA[1] - 0.281 * P_UA[2] - 0.157 * P_PD[1] - 0.487 * P_PD[2] \
                                    - 0.105 * P_WS[1] - 0.330 * P_WS[2]
        else:
            predicted_utility = np.nan

        # Update the original DataFrame with predicted utility values
        df.loc[index, 'PREDICTED UTILITY'] = predicted_utility

        # Append true and predicted values for metrics calculation
        y_true_list.append(df.loc[index, dv])
        y_pred_list.append(predicted_utility)

    # Calculate model metrics
    for dv in dependent_vars:
        y_true = np.array([df.loc[test_index, dv]])
        y_pred = np.array([df.loc[test_index, 'PREDICTED UTILITY']])

        # AIC and BIC
        aic = models[dv].aic
        bic = models[dv].bic

        # Bootstrapping metrics
        mae_mean, mae_ci, mse_mean, mse_ci, rmse_mean, rmse_ci = bootstrap_metrics(y_true, y_pred)

        # Utility statistics
        mean_utility = np.mean(y_pred)
        min_utility = np.min(y_pred)
        max_utility = np.max(y_pred)

        # Store the results per fold
        model_metrics.append({
            'Fold': fold,
            'Dependent Variable': dv,
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
            'Min Utility': min_utility,
            'Mean Utility': mean_utility,
            'Max Utility': max_utility
        })

# Convert model metrics to a DataFrame
model_metrics_df = pd.DataFrame(model_metrics)

# Save the model metrics to an Excel file (replace with correct file path)
output_model_metrics_file = 'path_to_output_folder/model_metrics.xlsx'
model_metrics_df.to_excel(output_model_metrics_file, index=False)

# Calculate averages across folds
mean_coefficients = coefficients_df.groupby(['Dependent Variable', 'Predictor']).agg({
    'Coefficient': 'mean',
    'SE': 'mean'
}).reset_index()
mean_coefficients['Fold'] = 'Average'

# Append mean coefficients to the coefficients DataFrame
coefficients_df = pd.concat([coefficients_df, mean_coefficients], ignore_index=True)

# Save the updated coefficients DataFrame (replace with correct file path)
output_coefficients_file = 'path_to_output_folder/coefficients.xlsx'
coefficients_df.to_excel(output_coefficients_file, index=False)

# Calculate average model metrics across folds
mean_model_metrics = model_metrics_df.groupby('Dependent Variable').agg({
    'AIC': 'mean',
    'BIC': 'mean',
    'MAE Mean': 'mean',
    'MAE 95% CI Lower': 'mean',
    'MAE 95% CI Upper': 'mean',
    'MSE Mean': 'mean',
    'MSE 95% CI Lower': 'mean',
    'MSE 95% CI Upper': 'mean',
    'RMSE Mean': 'mean',
    'RMSE 95% CI Lower': 'mean',
    'RMSE 95% CI Upper': 'mean',
    'Min Utility': 'mean',
    'Mean Utility': 'mean',
    'Max Utility': 'mean'
}).reset_index()
mean_model_metrics['Fold'] = 'Average'

# Append mean model metrics to the model metrics DataFrame
model_metrics_df = pd.concat([model_metrics_df, mean_model_metrics], ignore_index=True)

# Save the updated model metrics DataFrame (replace with correct file path)
output_model_metrics_file = 'path_to_output_folder/model_metrics.xlsx'
model_metrics_df.to_excel(output_model_metrics_file, index=False)

# Save the updated dataset with predicted utilities (replace with correct file path)
updated_file_path = 'path_to_output_folder/updated_data.xlsx'
df.to_excel(updated_file_path, index=False)
