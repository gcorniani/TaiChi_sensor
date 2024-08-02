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



data_folder = "../data"
data_file_name = "data.csv"
meta_data_file_name = "metadata.csv"
results_folder = "results/mrmr"
results_folder_path = os.path.join(data_folder, results_folder)
pathlib.Path(results_folder_path).mkdir(exist_ok=True)

metadata_path = os.path.join(data_folder, meta_data_file_name)
data_path = os.path.join(data_folder, data_file_name)

meta_data = pd.read_csv(metadata_path)
data = pd.read_csv(data_path)

data = clean_df(data, 0.2,.2)
data = data.dropna()
data = data.reset_index(drop=True)
meta_data = meta_data.iloc[data.index].reset_index(drop=True)
groups = pd.factorize(meta_data['user_id'])[0]


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14,16,18,20, 30, len(data.columns)] # number of features to test 
number_of_seeds = 20

# Loop through each unique exercise in the meta_data['task']
for exercise in set(list(meta_data['task'])):
    print(exercise)

    # Define the path and filename for saving results
    save_results_file = f"Results_{exercise}_Seeds{number_of_seeds}_feat{max(numbers)}.pkl"
    results_file_path = os.path.join(data_folder, results_folder, save_results_file)

    # Create a mask for the current exercise and filter data accordingly
    mask = meta_data['task'] == exercise
    exercise_data = data[mask].reset_index(drop=True)
    exercise_meta_data = meta_data[mask].iloc[exercise_data.index].reset_index(drop=True)
    groups = pd.factorize(exercise_meta_data['user_id'])[0]

    # Select features using mRMR feature selection
    selected_features = mrmr_classif(exercise_data, exercise_meta_data['score'], len(exercise_data.columns))

    rowsFold = []
    rowsIterations = []

    # Loop through each specified number of features
    for i, n in enumerate(numbers):
        logo = LeaveOneGroupOut()
        X = exercise_data[selected_features[:numbers[i]]]
        y = exercise_meta_data['score']
        logo.get_n_splits(X, y, groups)

        # Loop through each seed for model training
        for seed in range(number_of_seeds):
            y_preds = np.empty([0])
            y_probs = np.empty([0, 3])
            y_trues = np.empty([0])
            user_idForIteration = []
            taskForIteration = []
            scoreForIteration = []

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
                test_metadata = exercise_meta_data.iloc[test_index]

                row = [numbers[i], f1, seed, j, selected_features[:numbers[i]], test_metadata.user_id, test_metadata.task, test_metadata.score]
                rowsFold.append(row)

                # Collect metadata for this fold
                user_idForIteration.extend(test_metadata.user_id.tolist())
                taskForIteration.extend(test_metadata.task.tolist())
                scoreForIteration.extend(test_metadata.score.tolist())

            # Store iteration results
            f1_iter = f1_score(y_preds, y_trues, average='micro')
            row2 = [numbers[i], f1_iter, seed, selected_features[:numbers[i]], y_preds.tolist(), y_trues.tolist(), y_probs.tolist(), user_idForIteration, taskForIteration, scoreForIteration]
            rowsIterations.append(row2)

    # DataFrame for Fold-level results
    my_columnsFold = ["selected_features_number", "f1", "seed", "fold", "selected_features", "user_id", "task", "score"]
    resultsFold = pd.DataFrame(rowsFold, columns=my_columnsFold)

    # DataFrame for Iteration-level results, including metadata
    my_columnsIteration = ["selected_features_number", "f1", "seed", "selected_features", "y_pred", "y_true", "y_prob", "user_id", "task", "score"]
    resultsIteration = pd.DataFrame(rowsIterations, columns=my_columnsIteration)

    # Aggregate results to get mean F1 scores
    mean_f1_per_fold = resultsFold.groupby(['selected_features_number', 'seed']).f1.mean().reset_index()
    aggregated_resultsFold = mean_f1_per_fold.groupby('selected_features_number').f1.agg(['mean', 'std', 'min', 'max']).reset_index()
    aggregated_resultsIteration = resultsIteration.groupby(['selected_features_number'])['f1'].agg(['mean', 'std', 'min', 'max']).reset_index()

    # Select optimal number of features
    optimal_features = select_optimal_features(aggregated_resultsIteration, numbers, threshold_ratio=0.95)
    print(exercise + ": Optimal number of features:", optimal_features)
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
        "data": exercise_data,
        "meta_data": exercise_meta_data
    }

    # Save the dictionary to a file
    with open(results_file_path, 'wb') as file:
        pickle.dump(script_results, file)



for exercise in set(list(meta_data['task'])):
    save_results_file = "Results_"+exercise+"_Seeds" + str(number_of_seeds) +"_feat"+ str(max(numbers))+ ".pkl"
    results_file_path = os.path.join(data_folder,results_folder, save_results_file)
    with open(results_file_path, 'rb') as file:
            script_results = pickle.load(file)
    aggregated_resultsIteration = script_results['aggregated_resultsIteration']
    resultsIteration = script_results['resultsIteration']
    optimal_features = script_results['optimal_features']
    print(exercise + ": Optimal number of features:", optimal_features)
    plt.plot(np.arange(len(numbers)), aggregated_resultsIteration['mean'])
    plt.legend(set(list(meta_data['task'])))
    aggregated_resultsIteration = script_results['aggregated_resultsIteration']
    resultsIteration = script_results['resultsIteration']
# Selecting rows where 'selected_features_number' equals 'optimal_features' and compute finalF1 score 
    selected_rows_df = resultsIteration[resultsIteration['selected_features_number'] == optimal_features].reset_index(drop=True)
    totalF1 = calculate_totalf1(selected_rows_df)
    print(aggregated_resultsIteration[aggregated_resultsIteration['selected_features_number']==optimal_features])
    print(f'finale f1: {np.mean(totalF1), np.std(totalF1)} with {optimal_features}  features')
    print('')
    print(script_results['selected_features'][:optimal_features])