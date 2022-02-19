import csv
import pandas as pd
from pandas import read_csv
from pandas import set_option
import numpy as np
from matplotlib import pyplot
import os
import time
import scipy
from mlxtend.regressor import StackingCVRegressor
from pygam import GAM, LinearGAM, s, f, te
from xgboost import XGBRegressor
from mlens.ensemble import SuperLearner
from PyRapidML.datasets import extract_data
#boston = extract_data('boston')
from PyRapidML.regression import *
#exp_name = initializer(data = boston,  target = 'medv')
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn import tree
from sklearn.pipeline import make_pipeline,make_union
from sklearn.linear_model import LogisticRegression, RidgeCV, ElasticNetCV, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, FunctionTransformer, Normalizer
from sklearn.linear_model import RidgeCV, ElasticNetCV, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.svm import SVR
from math import sqrt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFwe, f_regression, SelectPercentile
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from tpot.builtins import StackingEstimator, ZeroCount
from copy import copy
from warfit_learn import metrics
from warfit_learn.metrics import score_pw20, score_r2, score_mae
from warfit_learn.metrics import confidence_interval
from statsmodels.imputation import mice
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
def variance(metric):
    meanvalue = np.mean(metric)
    sumsquares = 0
    for i in range(len(metric)):
        core = abs(metric[i] - meanvalue)
        sumsquares += np.square(core)
    variance = sumsquares / ((len(metric) - 1))
    return variance


def std_deviation(metric):
    return np.sqrt(variance(metric))


def confintlimit95(metric):
    return 1.96 * np.sqrt(variance(metric) / len(metric))


def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model) & (metrics[i]['metric'] == metric):
            container.append(metrics[i]['value'])
    return container


def collect_Results(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model):
            container.append(metrics[i][metric])
    return container


def MLAR(trueval, predval):
    # mean log of accuracy ratio
    sum = 0
    for i in range(len(trueval)):
        sum += np.log(predval[i] / trueval[i])
    return (np.exp(sum / len(trueval)) - 1) * 100


def MALAR(trueval, predval):
    # mean absolute log of accuracy ratio
    sum = 0
    for i in range(len(trueval)):
        sum += abs(np.log(predval[i] / trueval[i]))
    return (np.exp(sum / len(trueval)) - 1) * 100


def RSquared(trueval, predval):
    true_mean = np.mean(trueval)
    topsum = 0
    lowersum = 0
    for i in range(len(trueval)):
        topsum += np.square((predval[i] - true_mean))
        lowersum += np.square(trueval[i] - true_mean)
    return topsum / lowersum * 100


def BSA(height, weight):
    return 0.007184 * height ** 0.725 * weight ** 0.425


def ConvertYesNo(variable):
    if variable == "Yes":
        return 1
    elif variable == "No":
        return 0


def PercIn20(true, predicted):
    patients_in_20 = 0
    for i in range(len(true)):
        if abs(true[i] - predicted[i]) < 0.2 * true[i]:
            patients_in_20 += 1
    return 100 * patients_in_20 / len(true)


def INRThree(targetINR):
    if (targetINR >= 2.5).all() & (targetINR <= 3.5).all():
        return 1
    else:
        return 0


# create a list of base-models
def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(KNeighborsRegressor())
    models.append(AdaBoostRegressor())
    models.append(BaggingRegressor(n_estimators=10))
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(ExtraTreesRegressor(n_estimators=10))
    return models


# create the super learner
def get_super_learner(X):
    ensemble = SuperLearner(scorer=rmse, folds=10, shuffle=True, sample_size=len(X))
    # add base models
    models = get_models()
    ensemble.add(models)
    # add the meta model
    ensemble.add_meta(LinearRegression())
    return ensemble


def rmse(yreal, yhat):
    return sqrt(mean_squared_error(yreal, yhat))


def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1


def DoseRound(predicted):
    return round(predicted / 2.5, 0) * 2.5


def SList(series):
    return np.array(series.values.tolist())


def main():
    # Load dataset
    #boston = datasets.load_boston()
    #filename = "C:\users\claire\git_repo_1\cscthesispy\boston_house_prices.csv"
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
    #dataset = pd.read_csv(r"C:\users\claire\git_repo_1\cscthesispy\boston_house_prices.csv",";")
    validation_size = 0.20
    seed = 7

    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    filelist = [f for f in os.listdir(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations") if f.endswith(".csv")]
    for f in filelist:
        os.remove(os.path.join(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations", f))
    impNumber = 100
    for imp in range(impNumber):
        counter = imp + 1
        dfcurrent = df.loc[df[".imp"] == counter]
        # dfcurrent = df
        suffix = str(counter).zfill(3)
        dfcurrent.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_" + suffix + ".csv", ";")

    filesImp = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv'):
                filesImp.append(file)

        for file in filesImp:
            dfnew = pd.read_csv(root + '\\' + file, ";")
            dfnew2 = pd.read_csv(root + '\\' + file, ";", header=None)
            df = filesImp.index(file) + 1
            print("on imputation ", df)

            dfmod = dfnew
            dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
            dfmod["INR_Three"] = np.where(dfmod["Target_INR"] == "Three", 1, 0)
            dfmod["Target_INR"] = np.where(dfmod["Target_INR"] == "Three", 3.0,
                                       np.where(dfmod["Target_INR"] == "aTwo_point_five", 2.5, 2.0))
            dfmod["Target_INR"] = dfmod['Target_INR'].astype("float")
            dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
            dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
            dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
            dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
            dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)
            dfmod["AgeYears"] = dfmod["Age_years"]
            dfmod["AgeDecades"] = np.floor(dfmod["AgeYears"] * 0.1).astype("int")
            dfmod["AgeYears"] = dfmod["Age_years"]
            dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
            dfmod.drop(['Age_years'], axis=1, inplace=True)
            dfmod["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
            dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
            dfmod.drop(["HIV_status"], axis=1, inplace=True)
            dfmod.drop(['.id', '.imp', 'Unnamed: 0', 'Unnamed: 0.1', 'AgeDecades', 'INR_Three'], axis=1, inplace=True)
            dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
            print(dfmod.shape)
            print(dfmod.head(20))
        # Split-out validation dataset
            X =  dfmod[dfmod.columns.difference(['Dose_mg_week'])]
            Y =  dfmod['Dose_mg_week']

            X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)
            num_folds = 10
            scoring = 'neg_mean_absolute_error'

        # Spot-Check Algorithms

            models = []
            models.append(('LR', LinearRegression()))
            models.append(('LASSO', Lasso()))
            models.append(('EN', ElasticNet()))
            models.append(('KNN', KNeighborsRegressor()))
            models.append(('CART', DecisionTreeRegressor()))
            models.append(('SVR', SVR()))
    # evaluate each model in turn
            results = []
            names = []
            for name, model in models:
                kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
                cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                print(msg)
        # Compare Algorithms
        #fig = pyplot.figure()
        #fig.suptitle('Algorithm Comparison')
        #ax = fig.add_subplot(111)
        #pyplot.boxplot(results)
        #ax.set_xticklabels(names)
        #pyplot.show()
        # Standardize the dataset
            pipelines = []
            pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR',
                                                                       LinearRegression())])))
            pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO',
                                                                          Lasso())])))

            pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
                ElasticNet())])))
            pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
                KNeighborsRegressor())])))
            pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
                DecisionTreeRegressor())])))
            pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
            results = []
            names = []
            for name, model in pipelines:
                kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
                cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                print(msg)
        # Compare Algorithms
        #fig = pyplot.figure()
        #fig.suptitle('Scaled Algorithm Comparison')
        #ax = fig.add_subplot(111)
        #pyplot.boxplot(results)
        #ax.set_xticklabels(names)
        #pyplot.show()

    # KNN Algorithm tuning
     if False:
        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        k_values = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
        param_grid = dict(n_neighbors=k_values)
        model = KNeighborsRegressor()
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(rescaledX, Y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    # ensembles
        ensembles = []
        ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB',
                                                                           AdaBoostRegressor())])))
        ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM',
                                                                            GradientBoostingRegressor())])))
        ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF',
                                                                           RandomForestRegressor(n_estimators=10))])))
        ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET',
                                                                           ExtraTreesRegressor(n_estimators=10))])))
        results = []
        names = []
        for name, model in ensembles:
            kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

    #fig = pyplot.figure()
    #fig.suptitle('Scaled Ensemble Algorithm Comparison')
    #ax = fig.add_subplot(111)
    #pyplot.boxplot(results)
    #ax.set_xticklabels(names)
    #pyplot.show()


    # Tune scaled GBM
    if False:
        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        param_grid = dict(n_estimators=np.array([50, 100, 150, 200, 250, 300, 350, 400]))
        model = GradientBoostingRegressor(random_state=seed)
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(rescaledX, Y_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        # FINALISE MODEL ! prepare the model
        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
        model.fit(rescaledX, Y_train)
        # transform the validation dataset
        rescaledValidationX = scaler.transform(X_validation)
        predictions = model.predict(rescaledValidationX)
        print('MAE of finalised model',score_mae(Y_validation, predictions))


if __name__ == "__main__":
    main()
