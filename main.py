import data_handler
import pandas as pd


def main():
    data = data_handler.import_pickle_by_name('data_files/total_dataset.pkl')
    print(data)
    # determine_acceptance_function()
    # fit_function_to_data()
    # identify_results()


if __name__ == '__main__':
    main()
