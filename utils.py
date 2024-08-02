import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

def clean_df(df, max_nan_percentage_col, max_nan_percentage_row):
    """
    Cleans the DataFrame by filling or dropping columns and rows based on NaN thresholds.
    
    Args:
    df (DataFrame): The input DataFrame to be cleaned.
    max_nan_percentage_col (float): Maximum allowed percentage of NaNs in a column.
    max_nan_percentage_row (float): Maximum allowed percentage of NaNs in a row.

    Returns:
    DataFrame: The cleaned DataFrame.
    """
    max_nans_per_column = int(max_nan_percentage_col * df.shape[0])
    max_nans_per_row = int(max_nan_percentage_row * df.shape[1])

    for col in df.columns:
        if df[col].isna().sum() <= max_nans_per_column:
            df[col].fillna(df[col].mean(), inplace=True)

    cols_to_remove = df.columns[df.isna().sum() > max_nans_per_column]
    df.drop(cols_to_remove, axis=1, inplace=True)

    rows_to_remove = df.index[df.isna().sum(axis=1) > max_nans_per_row]
    df.drop(rows_to_remove, inplace=True)

    return df

def select_optimal_features(aggregated_results, numbers, threshold_ratio=0.95):
    """
    Selects the optimal number of features based on the mean F1 score.
    
    Args:
    aggregated_results (DataFrame): DataFrame containing the aggregated results with 'mean' F1 scores.
    numbers (list): List of numbers representing the number of features.
    threshold_ratio (float): The ratio of the max mean F1 score to set as the threshold.

    Returns:
    int: The optimal number of features.
    """
    if len(aggregated_results) != len(numbers):
        raise ValueError("Length of 'aggregated_results' and 'numbers' must be the same.")

    max_mean_f1 = aggregated_results['mean'].max()
    f1_threshold = max_mean_f1 * threshold_ratio

    for num_features, mean_f1 in zip(numbers, aggregated_results['mean']):
        if mean_f1 >= f1_threshold:
            return num_features

    return numbers[-1]

def calculate_totalf1(selected_rows_df):
    f1TotalIterations = np.empty(len(selected_rows_df))
    for i in range(len(selected_rows_df)):

        users = np.array(selected_rows_df['user_id'].iloc[i])
        tasks = np.array(selected_rows_df['task'].iloc[i])
        scores = np.array(selected_rows_df['score'].iloc[i])
        yprobs = np.array(selected_rows_df['y_prob'].iloc[i])
        # Now, 'selected_rows_df' is your new DataFrame with the selected rows
        scoresTarget = np.empty([0])
        mostPred =np.empty([0])
        for u in np.unique(users):
            for t in np.unique(tasks):
                # Find indexes where 'user_id' equals u
                index = np.where((users == u) & (tasks == t))[0]
               
                # Check if the index array is not empty
                if index.size > 0:
                    scoresTarget = np.append(scoresTarget,np.unique(scores[index][0]))
                    yprobsPred = yprobs[index]
                    mostPred = np.append(mostPred,np.argmax(np.sum(yprobsPred,axis=0)))
        f1 = f1_score(scoresTarget, mostPred, average='micro')
        f1TotalIterations[i] = f1
    return f1TotalIterations
