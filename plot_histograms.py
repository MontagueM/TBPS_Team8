import matplotlib.pyplot as plt


def plot_event_count_histograms(accept_data, q2_range):
    params = ["costhetal", "costhetak", "phi", "q2"]

    q2_range_data = {}
    for _, row in accept_data.iterrows():
        if q2_range[0] <= row['q2'] <= q2_range[1]:
            for param in params:
                if param not in q2_range_data.keys():
                    q2_range_data[param] = []
                q2_range_data[param].append(row[param])

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, p in enumerate(q2_range_data):
        px = i // 2
        py = i % 2
        ax[px, py].hist(q2_range_data[p], bins=25)
        ax[px, py].set_xlabel(p)
        ax[px, py].set_ylabel("Events")

    plt.show()