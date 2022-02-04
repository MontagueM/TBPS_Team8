import data_handler
import plot_histograms
import acceptance_function

q2_ranges = [
    (0.1, 0.98),
    (18., 19.)
]

# Import acceptance data
accept_data = data_handler.import_pickle_by_name('data/acceptance_mc.pkl')
# Produce histograms of event count vs parameters
plot_histograms.plot_event_count_histograms(accept_data, q2_range=q2_ranges[1])
# Run through acceptance function to generate efficiency data

# Plot efficiency vs parameters


a = 0