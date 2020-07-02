from collections import Counter
import matplotlib.pyplot as plt

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
