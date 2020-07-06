from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# For a series of lists e.g. pandas.Series<["a", "b"], ["a", "b", "c"]>,
# count the number of occurences of each element e.g.
# {"a": 2, "b": 2, "c": 1}
def count_series_list(series):
    flattened = sum(series.tolist(), [])
    counted = Counter(flattened)
    return dict(counted)

# For a dict of str => int, plot a horizontal bar chart
# e.g. {"a": 1, "b": 5, "c": 3} (used in conjuction with count_list)
def plot_dict(dict):
    bar = plt.barh(range(len(dict)), list(dict.values()), align="center")
    plt.yticks(range(len(dict)), list(dict.keys()))
    return bar

def plot_dicts(dicts):
    # Get all unique keys from dicts
    keys = list(set([key for dict in dicts for key in dict.keys()]))
    ind = np.arange(len(keys))

    return_values = []
    acc = np.zeros(len(keys))
    for dict in dicts:
        values = np.array([dict.get(key, 0) for key in keys])
        bar = plt.barh(ind, values, align="center", left=acc)
        return_values.append(bar)

        acc += values

    plt.yticks(ind, keys)

    return return_values
