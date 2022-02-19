import sklearn
import csv
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from warfit_learn import datasets, preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier # for extreme gradient boosting model
from warfit_learn.estimators import Estimator
from warfit_learn.evaluation import evaluate_estimators
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from warfit_learn.metrics.scoring import confidence_interval
from scipy.stats import norm
from mlxtend.regressor import StackingCVRegressor
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import ChangedBehaviorWarning
from sklearn.exceptions import DataConversionWarning
from cubist import Cubist
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ChangedBehaviorWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.reset_option('all')
from sklearn.feature_selection import SelectFwe, f_regression, SelectPercentile
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, \
    MinMaxScaler, FunctionTransformer, Normalizer
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoLarsCV, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, make_union
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_approximation import RBFSampler, Nystroem
from tpot.builtins import StackingEstimator, ZeroCount
#import tensorflow as tf
#from tensorflow import keras
from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense
from copy import copy
def format_summary(df_res):
    df_summary = df_res.groupby(['Estimator']).mean()
    df_summary.reset_index(inplace=True)
    for alg in df_res['Estimator'].unique():
        for metric in ['PW20', 'MAE']:
            lo, hi = confidence_interval(df_res[metric][df_res['Estimator'] == alg].values, )
            mean = df_res[metric][df_res['Estimator'] == alg].mean()
            for v in [mean, lo, hi]:
                if not (-10000 < v < 10000):
                    print('nan applied: ', alg, metric, lo, hi, mean)
                    mean, lo, hi = np.nan, np.nan, np.nan
                conf = f"{mean:.2f}({lo:.2f}-{hi:.2f})"
                print(alg, metric, lo, hi, mean, conf)
                df_summary[metric][df_summary['Estimator'] == alg] = conf
    return df_summary


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


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

impNumber = 100
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('expand_frame_repr', False)
pd.set_option("display.max_rows", False)
df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
for imp in range(impNumber):
    counter = imp + 1
    dfcurrent = df.loc[df[".imp"] == counter]
    suffix = str(counter).zfill(3)
    dfcurrent.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_" + suffix + ".csv", ";")
    filesImp = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv'):
                filesImp.append(file)
    for file in filesImp:
        dfnew = pd.read_csv(root + '\\' + file, ";")
        dfnew.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\DataAccessed\\" + file,";")
        df = filesImp.index(file) + 1
        dfmod = dfnew
        dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
        #dfmod["INR_Three"] = np.where(dfmod["Target_INR"] == "Three", 1, 0)
        dfmod["Target_INR"] = np.where(dfmod["Target_INR"] == "Three", 3.0,
                                       np.where(dfmod["Target_INR"] == "aTwo_point_five", 2.5, 2.0))
        dfmod["Target_INR"] = dfmod['Target_INR'].astype("float")
        dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
        dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
        dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
        dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
        dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)
        #dfmod["AgeDecades"] = np.floor(dfmod["Age_years"] * 0.1).astype("int")
        dfmod["AgeYears"] = dfmod["Age_years"]
        dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
        dfmod["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
        dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
        dfmod.drop(["HIV_status"], axis=1, inplace=True)
        #dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
        #dfmod.drop(["Height_cm"], axis = 1, inplace = True)
        #dfmod.drop(["Weight_kg"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0"],axis=1, inplace = True)
        dfmod.drop(["Unnamed: 0.1"],axis=1, inplace = True)
        dfmod.drop(["Age_years"], axis = 1, inplace = True)
        dfmod.drop([".imp"], axis=1, inplace=True)
        dfmod.drop([".id"], axis=1, inplace=True)
        if df == 1:
            print("On imputation ", df)
            data = dfmod

            print(data.shape)
            test_size = 0.2
            target_column = 'Dose_mg_week'
            train, test = train_test_split(data, test_size=test_size)
            y_train = train[target_column].values
            x_train = train.drop([target_column], axis=1).values
            scaler = StandardScaler().fit(x_train)
            rescaledX = scaler.transform(x_train)
            num_folds = 10
            seed = 7
            y_test = test[target_column].values
            x_test = test.drop([target_column], axis=1).values
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            base_model = RandomForestRegressor(n_estimators=10, random_state=42)
            rf = base_model
            num_folds = 10
            seed = 7
            kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
            if False:
                base_model.fit(x_train, y_train)
                base_accuracy = evaluate(base_model, x_test, y_test)
                print(base_accuracy)
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
                rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=kfold, verbose=2,
                                            random_state=42, n_jobs=-1)
            # Fit the random search model
                rf_random.fit(x_train, y_train)
                print(rf_random.best_params_)
                best_random = rf_random.best_estimator_
                random_accuracy = evaluate(best_random, x_test, y_test)
                print(random_accuracy)
                print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

            # Create the parameter grid based on the results of random search
            param_grid = {
                'bootstrap': [True],
                'max_depth': [6,8,10],
                'max_features': ['sqrt'],
                'min_samples_leaf': [4,6,8,10],
                'min_samples_split': [2,4,6,8],
                'n_estimators': [400,600,800]
            }

            # Instantiate the grid search model
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=2)

            # Fit the grid search to the data
            grid_search.fit(x_train,y_train)
            print(grid_search.best_params_)
            best_grid = grid_search.best_estimator_
            grid_accuracy = evaluate(best_grid, x_test,y_test)
            base_accuracy = 59.39291859218764
            print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))
            print('the end')