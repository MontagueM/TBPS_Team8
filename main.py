import data_handler
import plot_histograms
import acceptance_function

q2_ranges = [
    (0.1, 0.98),
    (18., 19.)
]

# Import acceptance data
accept_data = data_handler.import_pickle_by_name('data/signal.pkl')
q2range = q2_ranges[1]
accept_data_range = data_handler.select_data_by_q2range(accept_data, q2range)
# Produce histograms of event count vs parameters
# plot_histograms.plot_event_count_histograms(accept_data_range)
# Run through acceptance function to generate efficiency data
accept_eff_data = acceptance_function.accept_costhetal(accept_data_range)
# Plot efficiency vs parameters


a = 0