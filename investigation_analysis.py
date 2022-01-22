"""
Random stuff in here for testing
"""
import data_handler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def setup() -> pd.DataFrame:
    data = data_handler.import_pickle_by_name('data_files/total_dataset.pkl')
    return data


def plot_random_hist(data: pd.DataFrame) -> None:
    plt.hist(data['mu_plus_MC15TuneV1_ProbNNk'], bins=25, density=True)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    data = setup()
    plot_random_hist(data)
