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