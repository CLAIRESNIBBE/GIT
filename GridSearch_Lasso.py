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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier # for extreme gradient boosting model
from warfit_learn.estimators import Estimator
from warfit_learn.evaluation import evaluate_estimators
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from warfit_learn.metrics.scoring import confidence_interval
from sklearn.metrics import r2_score, get_scorer
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, \
    MinMaxScaler, FunctionTransformer, Normalizer, PolynomialFeatures
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


def regmodel_param_test(
        alphas_to_try, solvers_to_try,X, y, cv, scoring='r2',
        model_name='LASSO', X_test=None, y_test=None,
        draw_plot=False, filename=None):
    validation_scores = []
    train_scores = []
    results_list = []
    if X_test is not None:
        test_scores = []
        scorer = get_scorer(scoring)
    else:
        test_scores = None

    for curr_alpha in alphas_to_try:
        for curr_solver in solvers_to_try:
            if model_name == 'LASSO':
              regmodel = Lasso(alpha=curr_alpha)
            elif model_name == 'Ridge':
              regmodel = Ridge(alpha=curr_alpha, solver = curr_solver)
            else:
              return None
        results = cross_validate(
            regmodel, X, y, scoring=scoring, cv=cv,
            return_train_score=True)

        validation_scores.append(np.mean(results['test_score']))
        train_scores.append(np.mean(results['train_score']))
        results_list.append(results)

        if X_test is not None:
            regmodel.fit(X, y)
            y_pred = regmodel.predict(X_test)
            test_scores.append(scorer(regmodel, X_test, y_test))

    chosen_alpha_id = np.argmax(validation_scores)
    chosen_alpha = alphas_to_try[chosen_alpha_id]
    chosen_solver = solvers_to_try[chosen_alpha_id]
    max_validation_score = np.max(validation_scores)
    if X_test is not None:
        test_score_at_chosen_alpha = test_scores[chosen_alpha_id]
    else:
        test_score_at_chosen_alpha = None

    if draw_plot:
        regmodel_param_plot(
            validation_scores, train_scores, alphas_to_try, chosen_alpha,
            scoring, model_name, test_scores, filename)

    return chosen_alpha, chosen_solver, max_validation_score, test_score_at_chosen_alpha

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
            my_random_state = 6
            train, test = train_test_split(data, test_size=test_size,random_state=my_random_state)
            y_train = train[target_column].values
            x_train = train.drop([target_column], axis=1).values
            y_test = test[target_column].values
            x_test = test.drop([target_column], axis=1).values
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            x_train_poly = poly.fit_transform(x_train)
            sc = StandardScaler()
            x_train_poly_scaled = sc.fit_transform(x_train_poly)
            x_test_poly = poly.transform(x_test)
            x_test_poly_scaled = sc.transform(x_test_poly)
            cv = KFold(n_splits=5, shuffle=True, random_state=my_random_state)

            lasso_alphas = np.linspace(0, 0.02, 11)
            ridge_solvers = []
            ridge_solvers.append('svd')
            ridge_solvers.append('cholesky')
            ridge_solvers.append('lsqr')
            ridge_solvers.append('sag')

            chosen_alpha, chosen_solver, max_validation_score, test_score_at_chosen_alpha = \
            regmodel_param_test(lasso_alphas, ridge_solvers, x_train_poly_scaled, y_train,cv, scoring='r2', model_name='Ridge',
            X_test=x_test_poly_scaled, y_test=y_test, draw_plot=False, filename='ridge_wide_search')
            print("Chosen alpha: %.5f" % chosen_alpha)
            print("Chosen solver: ", chosen_solver)
            print("Validation score: %.5f" % max_validation_score)
            print("Test score at chosen alpha: %.5f" % test_score_at_chosen_alpha)

            lasso_alphas = np.linspace(0, 0.002, 11)
            chosen_alpha, max_validation_score, test_score_at_chosen_alpha = \
            regmodel_param_test(lasso_alphas, x_train_poly_scaled, y_train,cv, scoring='r2', model_name='Ridge',
            X_test=x_test_poly_scaled, y_test=y_test, draw_plot=False, filename='ridge_wide_search')
            print("Chosen alpha: %.5f" % chosen_alpha)
            print("Validation score: %.5f" % max_validation_score)
            print("Test score at chosen alpha: %.5f" % test_score_at_chosen_alpha)















