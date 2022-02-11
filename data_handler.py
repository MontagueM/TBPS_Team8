"""
Functions used to handle the external data files for use in the analysis
"""
import pickle
import pandas as pd


def import_pickle_by_name(filename: str) -> pd.DataFrame:
    """
    Imports a pickle file by name and returns the data
    """
    with open(filename, 'rb') as f:
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        data = pickle.load(f)
    return data


def select_data_by_q2range(accept_data, q2range):
    params = ["costhetal", "costhetak", "phi", "q2"]

    q2_range_data = {}
    for _, row in accept_data.iterrows():
        if q2range[0] <= row['q2'] <= q2range[1]:
            for param in params:
                if param not in q2_range_data.keys():
                    q2_range_data[param] = []
                q2_range_data[param].append(row[param])

    return q2_range_data