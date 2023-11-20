import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def print_statistics(data):

    print('mean -----------')
    print(data.mean(numeric_only=True))
    print('\nmax -----------')
    print(data.max(numeric_only=True))
    print('\nmin -----------')
    print(data.min(numeric_only=True))
    print('\nmedian -----------')
    print(data.median(numeric_only=True))
    print('\nskew -----------')
    print(data.skew(numeric_only=True))
    print('\nstd -----------')
    print(data.std(numeric_only=True))
    print('\nkurtosis -----------')
    print(data.kurtosis(numeric_only=True))


def visualize_scatter(x, y):

    plt.scatter(x, y)
    plt.show()


def visualize_heatmap(corr_matrix):

    sns.heatmap(corr_matrix, annot=True)
    plt.show()


def visualize_boxplot(x):

    sns.boxplot(x=x)
    plt.show()


data = pd.read_csv('DatasetL5_2.csv')

print_statistics(data)


numeric_data = data.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_data.corr()
visualize_heatmap(corr_matrix)


visualize_scatter(data['val'], data['year'])


visualize_boxplot(data['upper'])
