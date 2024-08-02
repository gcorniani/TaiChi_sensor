'''
Code use to the the performances of the movement specific model when the first module of the hierarchical model wrongly recognise the movement.
'''
import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from mrmr import mrmr_classif
from utils import *
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import plotly.graph_objects as go
import pickle

# Set data paths and create results folder if it doesn't exist
data_folder = "../data"
data_file_name = "data.csv"
meta_data_file_name = "metadata.csv"
results_folder = "results/mrmr"
results_folder_path = os.path.join(data_folder, results_folder)
pathlib.Path(results_folder_path).mkdir(exist_ok=True)

# Load metadata and data files
metadata_path = os.path.join(data_folder, meta_data_file_name)
data_path = os.path.join(data_folder, data_file_name)
meta_data = pd.read_csv(metadata_path)
data = pd.read_csv(data_path)

# Clean data
data = clean_df(data, 0.2, 0.2)
data = data.dropna().reset_index(drop=True)
meta_data = meta_data.iloc[data.index].reset_index(drop=True)
groups = pd.factorize(meta_data['user_id'])[0]

# Load previously computed results
results_identifier_path = 'results/Results_UnifiedTask_Seeds20_feat378.pkl'
with open(results_identifier_path, 'rb') as file:
    result_identifier = pickle.load(file)

# Extract optimal features and selected results
optimal_features_identifier = result_identifier['optimal_features']
result_identifierSelected = result_identifier['resultsIteration'][result_identifier['resultsIteration']['selected_features_number'] == optimal_features_identifier].iloc[0]
df_formIdentification = pd.DataFrame({
    'y_true': result_identifierSelected['y_true'],
    'y_pred': result_identifierSelected['y_pred'],
    'uniqueID': result_identifierSelected['uniqueID'],
    'user_id': result_identifierSelected['user_id']
})

# Initialize a dictionary to store selected features for each exercise
features_selected = {}
previous_results_folder = '/Users/gc594/Partners HealthCare Dropbox/Giulia Corniani/TaiChi/Tai Chi Stefano Backup/Code Giulia/python/data/results/mrmr_reduced'
for exercise in set(list(meta_data['task'])):
    print(exercise)
    previous_results_file = f"Results_{exercise}_Seeds20_feat378.pkl"
    results_file_path = os.path.join(previous_results_folder, previous_results_file)
    with open(results_file_path, 'rb') as file:
        script_results = pickle.load(file)
    features_selected[exercise] = script_results['selected_features'][:script_results['optimal_features']]
    print(len(features_selected[exercise]))

# Number of seeds for random forest classifier
number_of_seeds = 5

# Loop through each unique exercise
for exercise in set(list(meta_data['task'])):
    print(exercise)
    save_results_file_identified = f"Results_{exercise}_Seeds{number_of_seeds}_feat_Hierarchical_identified.pkl"
    results_file_path_identified = os.path.join(data_folder, results_folder, save_results_file_identified)

    # Filter data for the current exercise
    mask = meta_data['task'] == exercise
    exercise_data = data[mask].reset_index(drop=True)
    exercise_meta_data = meta_data[mask].iloc[exercise_data.index].reset_index(drop=True)
    groups = pd.factorize(exercise_meta_data['user_id'])[0]

    # Identify incorrect predictions for the current exercise
    identified_exercise = df_formIdentification[(df_formIdentification['y_pred'] == exercise) & (df_formIdentification['y_true'] != exercise)]
    print(len(identified_exercise))
    mask = meta_data['uniqueID'].isin(identified_exercise['uniqueID'])
    exercise_data_identified = data[mask].reset_index(drop=True)
    exercise_meta_data_identified = meta_data[mask].reset_index(drop=True)

    rowsFold_identified = []
    rowsIterations_identified = []

    # Perform Leave-One-Group-Out cross-validation
    logo = LeaveOneGroupOut()
    X = exercise_data[features_selected[exercise]]
    y = exercise_meta_data['score']
    X2 = exercise_data_identified[features_selected[exercise]]
    y2 = exercise_meta_data_identified['score']
    logo.get_n_splits(X, y, groups)

    for seed in range(number_of_seeds):
        y_preds_identified = np.empty([0])
        y_trues_identified = np.empty([0])
        user_idForIteration = []  
        taskForIteration = []
        scoreForIteration = []

        for j, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
            model = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=seed)
            model.fit(X.iloc[train_index], y.iloc[train_index])

            mask_identified = exercise_meta_data_identified['user_id'].isin(exercise_meta_data.iloc[test_index]['user_id'])
            X_identified = exercise_data_identified[mask_identified].reset_index(drop=True)
            y_identified = exercise_meta_data_identified[mask_identified].reset_index(drop=True)

            # Ensure X_identified has the same columns as X
            X_identified = X_identified[X.columns]

            if X_identified.shape[0] == 0:
                continue  # Skip this iteration if no samples are identified

            y_pred_identified = model.predict(X_identified)

            y_preds_identified = np.concatenate([y_preds_identified, y_pred_identified])
            y_trues_identified = np.concatenate([y_trues_identified, y_identified['score']])
            f1_identified = f1_score(y_identified['score'], y_pred_identified, average='micro')

            row_identified = [len(features_selected[exercise]), f1_identified, seed, j, features_selected[exercise], y_identified['user_id'].tolist(), y_identified['task'].tolist(), y_identified['score'].tolist()]
            rowsFold_identified.append(row_identified)

        if y_preds_identified.size > 0 and y_trues_identified.size > 0:
            f1_iter_identified = f1_score(y_preds_identified, y_trues_identified, average='micro')
            row2_identified = [len(features_selected[exercise]), f1_iter_identified, seed, features_selected[exercise], y_preds_identified.tolist(), y_trues_identified.tolist(), user_idForIteration, taskForIteration, scoreForIteration]
            rowsIterations_identified.append(row2_identified)

    if rowsFold_identified and rowsIterations_identified:
        my_columnsFold_identified = ["selected_features_number", "f1", "seed", "fold", "selected_features", "user_id", "task", "score"]
        resultsFold_identified = pd.DataFrame(rowsFold_identified, columns=my_columnsFold_identified)

        my_columnsIteration_identified = ["selected_features_number", "f1", "seed", "selected_features", "y_pred", "y_true", "user_id", "task", "score"]
        resultsIteration_identified = pd.DataFrame(rowsIterations_identified, columns=my_columnsIteration_identified)

        total_correct = (y_preds_identified == y_trues_identified).sum()
        total_samples = len(y_trues_identified)
        print(f"{exercise}: {total_correct} correctly identified over {total_samples} total samples")

        mean_f1_per_fold_identified = resultsFold_identified.groupby(['selected_features_number', 'seed']).f1.mean().reset_index()
        aggregated_resultsFold_identified = mean_f1_per_fold_identified.groupby('selected_features_number').f1.agg(['mean', 'std', 'min', 'max']).reset_index()

        aggregated_resultsIteration_identified = resultsIteration_identified.groupby(['selected_features_number'])['f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
        print(aggregated_resultsIteration_identified)

        script_results_identified = {
            "selected_features": features_selected[exercise],
            "numbers": len(features_selected[exercise]),
            "number_of_seeds": number_of_seeds,
            "resultsFold": resultsFold_identified,
            "resultsIteration": resultsIteration_identified,
            "mean_f1_per_fold": mean_f1_per_fold_identified,
            "aggregated_resultsFold": aggregated_resultsFold_identified,
            "aggregated_resultsIteration": aggregated_resultsIteration_identified,
            "optimal_features": len(features_selected[exercise]),
            "data": exercise_data_identified,
            "meta_data": exercise_meta_data_identified
        }

        with open(results_file_path_identified, 'wb') as file:
            pickle.dump(script_results_identified, file)

    del exercise_data, exercise_meta_data, mask
