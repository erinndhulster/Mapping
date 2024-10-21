import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from scipy.stats import chi2_contingency
from sklearn.utils import resample

# PART A: Load the dataset
# Provide the correct path to your input file (replace with actual file path)
file_path = 'path_to_your_data/Input_bn_1.xlsx'
data = pd.read_excel(file_path)

# View the first few rows of the dataset to verify the contents
print(data.head())

# Define the EQ5D-Y dimensions and the predictor variables
# Note: The predictors depend on the set of predictors you are using.
eq5d_y_dimensions = ['Mobility', 'Taking_care_of_myself', 'Daily_activities', 'Pain', 'Feeling_sad']
predictors = ['WB_2', 'WB_4', 'WB_6', 'WB_8', 'WB_10']

# Function to perform chi-squared test and find significant predictors for each dimension
def chi2_test(data, predictors, outcome):
    significant_predictors = []
    for predictor in predictors:
        contingency_table = pd.crosstab(data[predictor], data[outcome])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        # If p-value is significant at 5% level, append predictor to significant list
        if p < 0.05:
            significant_predictors.append(predictor)
    return significant_predictors

# Dictionary to store significant predictors per dimension
significant_predictors_per_dimension = {}

# PART B: Perform chi-squared test for each EQ5D-Y dimension
for dimension in eq5d_y_dimensions:
    significant_predictors = chi2_test(data, predictors, dimension)
    significant_predictors_per_dimension[dimension] = significant_predictors

# Display the significant predictors for each dimension
for dimension, significant_predictors in significant_predictors_per_dimension.items():
    print(f'Significant predictors for {dimension}: {significant_predictors}')

# PART C: Create Tree Augmented Naive (TAN) structures for each EQ5D-Y dimension

# Function to create a Directed Acyclic Graph (DAG) for each dimension and its significant predictors
def create_dag_structure(dimension, predictors):
    # Create a BayesianModel using the dimension as the outcome and predictors as parents
    model = BayesianModel([(dimension, predictor) for predictor in predictors])

    # Add interactions between predictors to the model
    model.add_edges_from(
        [(predictors[i], predictors[j]) for i in range(len(predictors)) for j in range(i + 1, len(predictors))])

    return model

# Function to estimate Conditional Probability Distributions (CPDs) and add them to the model
def add_cpd_to_model(model, data):
    # Estimate CPDs using Maximum Likelihood Estimation
    estimator = MaximumLikelihoodEstimator(model, data)
    cpds = estimator.get_parameters()

    # Add the estimated CPDs to the model
    model.add_cpds(*cpds)

    return model, cpds

# Dictionary to store the DAG structures and CPDs for each dimension
dag_structures = {}
cpds_per_dimension = {}

# Create a DAG structure and estimate CPDs for each EQ5D-Y dimension
for dimension, significant_predictors in significant_predictors_per_dimension.items():
    if significant_predictors:
        dag_model = create_dag_structure(dimension, significant_predictors)
        dag_model, cpds = add_cpd_to_model(dag_model, data)
        dag_structures[dimension] = dag_model
        cpds_per_dimension[dimension] = cpds
        print(f'DAG structure for {dimension} created with predictors: {significant_predictors}')
    else:
        print(f'No significant predictors found for {dimension}')

# PART D: Save the DAG models as PNG files
# Provide a path to save the output files
output_path = 'path_to_output_folder/'

for dimension, dag_model in dag_structures.items():
    print(f"\nStructure of DAG model for {dimension}:")
    print(dag_model.edges())

    # Convert the DAG model to a NetworkX graph
    G = nx.DiGraph(dag_model.edges())

    # Plot the structure of the DAG model
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=15, font_weight="bold",
            arrowsize=20)
    plt.title(f"DAG Model Structure for {dimension}")

    # Save as PNG file
    plt.savefig(f"{output_path}DAG_model_{dimension}.png")
    plt.close()

# PART E: Print CPDs to the console and save them to text files

# Function to write CPDs to text files
def write_cpds_to_file(cpds, dimension, output_path):
    # Define the path for the CPD file
    output_file = os.path.join(output_path, f'CPDs_{dimension}.txt')

    # Write CPDs to a .txt file for the given dimension
    with open(output_file, 'w') as f:
        f.write(f"CPDs for {dimension}:\n")
        for cpd in cpds:
            f.write(f"\nCPD for {cpd.variable}:\n")
            f.write(str(cpd))
            f.write("\n" + "-"*50 + "\n")  # Add separator for readability

    print(f"CPDs for {dimension} have been saved to {output_file}")

# PART F: Calculation of predictions and predicted utilities

# Load the dataset again for prediction (replace file path)
predictions_data = pd.read_excel(file_path)

# Function to calculate probability distribution based on evidence
def calculate_probability(model, cpds, evidence, outcome):
    model.add_cpds(*cpds)
    inference = VariableElimination(model)
    prob_dist = inference.query(variables=[outcome], evidence=evidence)
    return prob_dist

# Calculate predicted utilities for each data point
predicted_utilities = []

for _, row in predictions_data.iterrows():
    evidence = row[predictors].to_dict()  # Use predictors as evidence

    # Calculate the probabilities for each EQ5D-Y dimension
    mobility_probs = calculate_probability(dag_structures['Mobility'], cpds_per_dimension['Mobility'], evidence, 'Mobility')
    taking_care_probs = calculate_probability(dag_structures['Taking_care_of_myself'], cpds_per_dimension['Taking_care_of_myself'], evidence, 'Taking_care_of_myself')
    daily_activities_probs = calculate_probability(dag_structures['Daily_activities'], cpds_per_dimension['Daily_activities'], evidence, 'Daily_activities')
    pain_probs = calculate_probability(dag_structures['Pain'], cpds_per_dimension['Pain'], evidence, 'Pain')
    feeling_sad_probs = calculate_probability(dag_structures['Feeling_sad'], cpds_per_dimension['Feeling_sad'], evidence, 'Feeling_sad')

    # Calculate the utility based on the probabilities
    predicted_utility = (
        1 -
        0.064 * mobility_probs.values[1] - 0.203 * mobility_probs.values[2] -
        0.046 * taking_care_probs.values[1] - 0.174 * taking_care_probs.values[2] -
        0.104 * daily_activities_probs.values[1] - 0.281 * daily_activities_probs.values[2] -
        0.157 * pain_probs.values[1] - 0.487 * pain_probs.values[2] -
        0.105 * feeling_sad_probs.values[1] - 0.330 * feeling_sad_probs.values[2]
    )

    predicted_utilities.append(predicted_utility)

# Add the predicted utilities to the dataset
predictions_data['PREDICTED UTILITY'] = predicted_utilities

# Save the updated dataset (replace file path)
predictions_data.to_excel(file_path, index=False)

print("The predicted utilities have been saved in the 'PREDICTED UTILITY' column.")

# PART G: Fit and accuracy measures per dimension

# Function to calculate AIC and BIC
def calculate_aic_bic(observed, predicted, num_params):
    n = len(observed)
    residual_sum_of_squares = np.sum((observed - predicted) ** 2)
    aic = n * np.log(residual_sum_of_squares / n) + 2 * num_params
    bic = n * np.log(residual_sum_of_squares / n) + num_params * np.log(n)
    return aic, bic

# Function to calculate accuracy metrics using bootstrapping
def bootstrap_metrics(observed, predicted, n_bootstraps=1000):
    bootstrap_mae = []
    bootstrap_mse = []
    bootstrap_rmse = []

    # Perform bootstrapping to calculate confidence intervals for MAE, MSE, RMSE
    for _ in range(n_bootstraps):
        boot_obs, boot_pred = resample(observed, predicted)
        mae = np.mean(np.abs(boot_obs - boot_pred))
        mse = np.mean((boot_obs - boot_pred) ** 2)
        rmse = np.sqrt(mse)

        bootstrap_mae.append(mae)
        bootstrap_mse.append(mse)
        bootstrap_rmse.append(rmse)

    # Calculate 95% confidence intervals
    mae_ci = np.percentile(bootstrap_mae, [2.5, 97.5])
    mse_ci = np.percentile(bootstrap_mse, [2.5, 97.5])
    rmse_ci = np.percentile(bootstrap_rmse, [2.5, 97.5])

    return np.mean(bootstrap_mae), mae_ci, np.mean(bootstrap_mse), mse_ci, np.mean(bootstrap_rmse), rmse_ci

# Predicted utilities from earlier
predicted_utilities = predictions_data['PREDICTED UTILITY'].values

# DataFrame to store results per dimension
results_df = pd.DataFrame()

for dimension in eq5d_y_dimensions:
    # Get the observed utilities for each dimension
    observed_utilities = predictions_data[dimension].values

    # Calculate the number of parameters for AIC/BIC
    num_params = len(significant_predictors_per_dimension[dimension])

    # Calculate AIC and BIC
    aic, bic = calculate_aic_bic(observed_utilities, predicted_utilities, num_params)

    # Calculate accuracy measures with bootstrapped confidence intervals
    mae, mae_ci, mse, mse_ci, rmse, rmse_ci = bootstrap_metrics(observed_utilities, predicted_utilities)

    # Collect all measures and values in lists
    measures = ['AIC', 'BIC', 'Mean Absolute Error', 'MAE 95% CI Lower', 'MAE 95% CI Upper',
                'Mean Squared Error', 'MSE 95% CI Lower', 'MSE 95% CI Upper',
                'Root Mean Squared Error', 'RMSE 95% CI Lower', 'RMSE 95% CI Upper']

    values = [aic, bic, mae, mae_ci[0], mae_ci[1], mse, mse_ci[0], mse_ci[1], rmse, rmse_ci[0], rmse_ci[1]]

    # Ensure matching lengths of measures and values
    if len(measures) == len(values):
        # Create a DataFrame for this dimension's results
        results_for_dimension = pd.DataFrame({
            'Dimension': [dimension] * len(measures),
            'Measure': measures,
            'Value': values
        })

        # Append to the overall results DataFrame
        results_df = pd.concat([results_df, results_for_dimension], ignore_index=True)
    else:
        print(f"Warning: Length mismatch between 'measures' and 'values' for dimension: {dimension}")

# Utility calculations (global, not per dimension)
mean_predicted_utilities = np.mean(predicted_utilities)
max_predicted_utilities = np.max(predicted_utilities)
min_predicted_utilities = np.min(predicted_utilities)

# Add global utility results to the results DataFrame
utilities_df = pd.DataFrame({
    'Dimension': ['Global'] * 3,
    'Measure': ['Mean Predicted Utility', 'Max Predicted Utility', 'Min Predicted Utility'],
    'Value': [mean_predicted_utilities, max_predicted_utilities, min_predicted_utilities]
})

# Concatenate utility results with fit and accuracy measures
results_df = pd.concat([results_df, utilities_df], ignore_index=True)

# Save the complete results to an Excel file (replace file path)
results_df.to_excel(f"{output_path}Results_per_dimension.xlsx", sheet_name='Fit and Accuracy Measures', index=False)

print("Model fit and accuracy measures per dimension, and global utilities have been successfully saved to the Excel file.")
