import matplotlib.pyplot as plt


def plot_event_count_histograms(q2_range_data):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, p in enumerate(q2_range_data):
        px = i // 2
        py = i % 2
        ax[px, py].hist(q2_range_data[p], bins=25)
        ax[px, py].set_xlabel(p)
        ax[px, py].set_ylabel("Events")

    plt.show()