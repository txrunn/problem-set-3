'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/genres.csv')
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    genre_list = genres_df['genre'].unique().tolist()

    # Create binary columns for each genre in model_pred_df
    for genre in genre_list:
        model_pred_df[f'true_{genre}'] = model_pred_df['actual genres'].apply(lambda x: genre in x)
        model_pred_df[f'pred_{genre}'] = model_pred_df['predicted'].apply(lambda x: genre in x)

    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}

    for idx, row in model_pred_df.iterrows():
        for genre in genre_list:
            true_val = int(row[f'true_{genre}'])
            pred_val = int(row[f'pred_{genre}'])
            if true_val == 1:
                genre_true_counts[genre] += 1
            if pred_val == 1 and true_val == 1:
                genre_tp_counts[genre] += 1
            if pred_val == 1 and true_val == 0:
                genre_fp_counts[genre] += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
