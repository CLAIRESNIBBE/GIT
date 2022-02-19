import csv
# Evaluate using Shuffle Split Cross Validation
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy
import pandas as pd
filename = "Pima_Indians_Diabetes"
def main():
    filewritename = input("Enter file name: \n")
    fileoutput = open(filewritename, 'w')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dfmod = pd.read_csv(r"C:\Users\Claire\anaconda\envs\datascience\lib\site-packages\sklearn\datasets\data\boston_house_prices.csv",";")
    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Pima_Indians_Diabetes.csv", ";", names = names)
    data=read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Pima_Indians_Diabetes.csv", ";", names = names)
    types = data.dtypes
    print(types)
    set_option('display.width', 100)
    set_option('precision', 3)
    description = data.describe()
    print(description)
    correlations = data.corr(method='pearson')
    print(correlations)
    data.hist()
    pyplot.show()
    data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    pyplot.show()
    data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    pyplot.show()
    correlations = data.corr()
    # plot correlation matrix
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0, 9, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    pyplot.show()
    # correlations = data.corr()
    #plot correlation matrix
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    pyplot.show()
    scatter_matrix(data)
    pyplot.show()
    skew = data.skew()
    print(skew)

    if False:
        array = pd.DataFrame.values
        X = df[:, 0:8]
        Y = df[:, 8]
        n_splits = 10
        test_size = 0.33
        seed = 7
        kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        model = LogisticRegression(solver='liblinear')
        results = cross_val_score(model, X, Y, cv=kfold)
        print("Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
             'B', 'LSTAT', 'MEDV']
    dataframe = dfmod #read_csv(filename, delim_whitespace=True, names=names)
    array = dataframe.values
    X = array[:, 0:13]
    Y = array[:, 13]
    model = LinearRegression()
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = LinearRegression()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    print(results.mean())


if __name__ == "__main__":
   main()



