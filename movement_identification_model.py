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

# Define the number of features to be selected and the number of seeds
numbers = [1, 3, 5, 7, 9, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150, 200, 250, len(data.columns)]
number_of_seeds = 20

# Define the path and filename for saving results
save_results_file = f"Results_UnifiedTask_Seeds{number_of_seeds}_feat{max(numbers)}.pkl"
results_file_path = os.path.join(data_folder, results_folder, save_results_file)

# Perform feature selection using mRMR
selected_features = mrmr_classif(data, meta_data['score'], len(data.columns))
rowsFold = []
rowsIterations = []

# Loop through each specified number of features
for i, n in enumerate(numbers):
    print(i)
    logo = LeaveOneGroupOut()
    X = data[selected_features[:numbers[i]]]
    y = meta_data['task']
    groups = meta_data['user_id']
    logo.get_n_splits(X, y, groups)
    
    for seed in range(number_of_seeds):
        y_preds = np.empty([0])
        y_probs = np.empty([0, len(set(y))])
        y_trues = np.empty([0])
        user_idForIteration = []
        taskForIteration = []
        scoreForIteration = []
        uniqueIDForIteration = []
        
        # Perform Leave-One-Group-Out cross-validation
        for j, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
            model = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=seed)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = model.predict(X.iloc[test_index])
            y_preds = np.concatenate([y_preds, y_pred])
            y_trues = np.concatenate([y_trues, y.iloc[test_index]])
            y_prob = model.predict_proba(X.iloc[test_index])
            y_probs = np.concatenate([y_probs, y_prob])

            f1 = f1_score(y.iloc[test_index], y_pred, average='micro')
            test_metadata = meta_data.iloc[test_index]

            row = [numbers[i], f1, seed, j, selected_features[:numbers[i]], test_metadata.user_id, test_metadata.task, test_metadata.score, test_metadata.uniqueID]
            rowsFold.append(row)
            user_idForIteration.extend(test_metadata.user_id.tolist())
            taskForIteration.extend(test_metadata.task.tolist())
            scoreForIteration.extend(test_metadata.score.tolist())
            uniqueIDForIteration.extend(test_metadata.uniqueID.tolist())

        f1_iter = f1_score(y_preds, y_trues, average='micro')
        row2 = [numbers[i], f1_iter, seed, selected_features[:numbers[i]], y_preds.tolist(), y_trues.tolist(), y_probs.tolist(), user_idForIteration, taskForIteration, scoreForIteration, uniqueIDForIteration]
        rowsIterations.append(row2)
        print(f1_iter)

# DataFrame for Fold-level results
my_columnsFold = ["selected_features_number", "f1", "seed", "fold", "selected_features", "user_id", "task", "score", "uniqueID"]
resultsFold = pd.DataFrame(rowsFold, columns=my_columnsFold)

# DataFrame for Iteration-level results, including metadata
my_columnsIteration = ["selected_features_number", "f1", "seed", "selected_features", "y_pred", "y_true", "y_prob", "user_id", "task", "score", "uniqueID"]
resultsIteration = pd.DataFrame(rowsIterations, columns=my_columnsIteration)

# Aggregate results to get mean F1 scores
mean_f1_per_fold = resultsFold.groupby(['selected_features_number', 'seed']).f1.mean().reset_index()
aggregated_resultsFold = mean_f1_per_fold.groupby('selected_features_number').f1.agg(['mean', 'std', 'min', 'max']).reset_index()

aggregated_resultsIteration = resultsIteration.groupby(['selected_features_number'])['f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
print(aggregated_resultsIteration)

# Select optimal number of features
optimal_features = select_optimal_features(aggregated_resultsIteration, numbers, 0.95)
print("Optimal number of features:", optimal_features)
print(aggregated_resultsIteration[aggregated_resultsIteration['selected_features_number'] == optimal_features])

# Save results to a dictionary
script_results = {
    "selected_features": selected_features,
    "numbers": numbers,
    "number_of_seeds": number_of_seeds,
    "resultsFold": resultsFold,
    "resultsIteration": resultsIteration,
    "mean_f1_per_fold": mean_f1_per_fold,
    "aggregated_resultsFold": aggregated_resultsFold,
    "aggregated_resultsIteration": aggregated_resultsIteration,
    "optimal_features": optimal_features,
    "data": data,
    "meta_data": meta_data
}

# Save the dictionary to a file
with open(results_file_path, 'wb') as file:
    pickle.dump(script_results, file)

# Print optimal number of features
optimal_features = select_optimal_features(aggregated_resultsIteration, numbers, 0.95)
print("Optimal number of features:", optimal_features)
