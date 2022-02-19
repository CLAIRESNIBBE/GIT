import csv
import pandas as pd
import numpy as np
import os
import time
import scipy
from mlxtend.regressor import StackingCVRegressor
from pygam import GAM, LinearGAM, s, f, te
from xgboost import XGBRegressor
from mlens.ensemble import SuperLearner
from PyRapidML.datasets import extract_data
boston = extract_data('boston')
from PyRapidML.regression import *
exp_name = initializer(data = boston,  target = 'medv')
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
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
    filewritename = input("Enter file name: \n")
    fileoutput = open(filewritename, 'w')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    tdf = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
#    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLCombine.csv", ";")
    filelist = [f for f in os.listdir(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations") if f.endswith(".csv")]
    for f in filelist:
        os.remove(os.path.join(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations", f))

    results = []
    std_Dev = []
    std_Dev_Summ = ({'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0})

    impNumber = 100
    for imp in range(impNumber):
        counter = imp + 1
        #dfcurrent = df.loc[df[".imp"] == counter]
        dfcurrent = df
        dfcurrent.to_csv()
        suffix = str(counter).zfill(3)
        dfcurrent.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_" + suffix + ".csv", ";")

    filesImp = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv'):
                filesImp.append(file)


    for file in filesImp:
        dfnew = pd.read_csv(root + '\\' + file, ";")
        dfnew2 = pd.read_csv(root + '\\' + file, ";",header=None)
        df = filesImp.index(file) + 1
        print("on imputation ", df)
        impResults = []
        models = []
        MLearning = []
        dfmod = dfnew
        #dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
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
        dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)


        dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnWARPATH.csv", ";")
        X = dfmod[dfmod.columns.difference(['Dose_mg_week'])]
        y = dfmod['Dose_mg_week']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
        print('Train', X_train.shape, y_train.shape, 'Test', X_test.shape, y_test.shape)
        y_test = y_test.tolist()
        models = []
        models.append(('LR', LinearRegression()))
        models.append(('LASSO', Lasso()))
        models.append(('EN', ElasticNet()))
        models.append(('KNN', KNeighborsRegressor()))
        models.append(('CART', DecisionTreeRegressor()))
        models.append(('SVR', SVR()))


        MLearning.append({'model': 'Linear', 'y-true': y_test, 'y-pred': y_pred_Linear})
        MLearning.append({'model': 'Lasso', 'y-true': y_test, 'y-pred': y_pred_Lasso})
        MLearning.append({'model': 'Ridge', 'y-true': y_test, 'y-pred': y_pred_Ridge})
        MLearning.append({'model': 'GBR', 'y-true': y_test, 'y-pred': y_pred_GBR})
        MLearning.append({'model': 'SVR', 'y-true': y_test, 'y-pred': y_pred_SVR})
        MLearning.append({'model': 'DTR', 'y-true': y_test, 'y-pred': y_pred_DTR})
        MLearning.append({'model': 'GBT', 'y-true': y_test, 'y-pred': y_pred_GBT})
        MLearning.append({'model': 'RF', 'y-true': y_test, 'y-pred': y_pred_RF})
        MLearning.append({'model': 'Stack', 'y-true': y_test, 'y-pred': y_pred_stack})
        MLearning.append({'model': 'Stack2', 'y-true': y_test, 'y-pred': y_pred_stack2})
        MLearning.append({'model': 'Stack3', 'y-true': y_test, 'y-pred': y_pred_stack3})
        MLearning.append({'model': 'Stack4', 'y-true': y_test, 'y-pred': y_pred_stack4})
        MLearning.append({'model': 'Stack5', 'y-true': y_test, 'y-pred': y_pred_stack5})
        MLearning.append({'model': 'Tpot2', 'y-true': y_test, 'y-pred': y_pred_tpot2})
        MLearning.append({'model': 'Tpot10', 'y-true': y_test, 'y-pred': y_pred_tpot10})
        MLearning.append({'model': 'Tpot17', 'y-true': y_test, 'y-pred': y_pred_tpot17})
        MLearning.append({'model': 'Tpot_06_12', 'y-true': y_test, 'y-pred': y_pred_tpot_06_12})
        MLearning.append({'model': 'Tpot_06_12_02', 'y-true': y_test, 'y-pred': y_pred_tpot_06_12_02 })
        MLearning.append({'model': 'Tpot_06_12_03', 'y-true': y_test, 'y-pred': y_pred_tpot_06_12_03})



        #Mlearning.append({'model': 'Super', 'y-true': y_test, 'y-pred': y_pred_Super})

        for m in range(len(MLearning)):
            skmodel = MLearning[m]['model']
            y_test = MLearning[m]['y-true']
            y_pred = MLearning[m]['y-pred']
            pw20 = PercIn20(y_test, y_pred)
            mae = score_mae(y_test, y_pred)
            R2 = RSquared(y_test, y_pred)
            mape = MALAR(y_test, y_pred)
            bias = MLAR(y_test, y_pred)
            impResults.append(
                {'Imp': df, 'model': skmodel, 'MAE': mae, 'PW20': pw20, 'R2': R2, 'MLAR': bias, 'MALAR': mape,
                 'y-true': y_test, 'y-pred': y_pred})

        suffix = str(df).zfill(3)
        impResultsPD = pd.DataFrame(impResults)
        impResultsPD.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARMLearn" + suffix + ".csv", ";")

        if (find(models, 'model', 'Linear') == -1):
            models.append({'model': 'Linear', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Lasso', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Ridge', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'GBR', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'SVR', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'DTR', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'GBT', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'RF', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Stack', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Stack2', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Stack3', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Stack4', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Stack5', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Tpot2', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Tpot10', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Tpot17', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Tpot_06_12', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Tpot_06_12_02', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Tpot_06_12_03', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})








            #models.append({'model': 'Super', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})

        for k in range(len(impResults)):
            a = impResults[k]['model']
            b = impResults[k]['Imp']
            c = impResults[k]['MAE']
            d = impResults[k]['PW20']
            e = impResults[k]['R2']
            g = impResults[k]['MLAR']
            h = impResults[k]['MALAR']
            results.append({'Imp': b, 'model': a, 'MAE': c, 'PW20': d, 'R2': e, 'MLAR': g, 'MALAR': h})

    if df == impNumber:
        for k in range(len(models)):
            fieldname = models[k]['model']
            for m in range(len(results)):
                if results[m]['model'] == fieldname:
                    models[k]['MAE'] += results[m]['MAE']
                    models[k]['PW20'] += results[m]['PW20']
                    models[k]['R2'] += results[m]['R2']
                    models[k]['MLAR'] += results[m]['MLAR']
                    models[k]['MALAR'] += results[m]['MALAR']

        Bfactor = (impNumber + 1) / impNumber
        for k in range(len(models)):
            fieldname = models[k]['model']
            # stdpos = find(std_Dev_Summ, 'model', fieldname)
            mae_value = models[k]['MAE'] / impNumber
            mae_list = collect_Results(results, fieldname, 'MAE')
            mae_variance = variance(mae_list) * Bfactor
            mae_std_dev = np.sqrt(mae_variance)
            # mae_std_dev = std_Dev_Summ[stdpos]['MAE'] / impNumber
            mae_CI_minus = mae_value
            mae_CI_plus = mae_value
            # mae_CI_minus = round(mae_value - 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
            # mae_CI_plus = round(mae_value + 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
            pw20_value = models[k]['PW20'] / impNumber
            pw20_list = collect_Results(results, fieldname, 'PW20')
            pw20_variance = variance(pw20_list) * Bfactor
            pw20_std_dev = np.sqrt(pw20_variance)
            pw20_CI_plus = pw20_value
            pw20_CI_minus = pw20_value
            # pw20_std_dev = std_Dev_Summ[stdpos]['PW20'] / impNumber
            # pw20_CI_minus = round(pw20_value - 1.96 * np.sqrt(pw20_std_dev + pw20_variance), 4)
            # pw20_CI_plus = round(pw20_value + 1.96 * np.sqrt(pw20_std_dev + pw20_variance), 4)
            R2_value = models[k]['R2'] / impNumber
            R2_list = collect_Results(results, fieldname, 'R2')
            R2_variance = variance(R2_list) * Bfactor
            R2_CI_plus = R2_value
            R2_CI_minus = R2_value
            R2_std_dev = np.sqrt(R2_variance)
            # R2_CI_minus = round(R2_value - 1.96 * np.sqrt(R2_std_dev + R2_variance), 4)
            # R2_CI_plus = round(R2_value + 1.96 * np.sqrt(R2_std_dev + R2_variance), 4)
            MALAR_value = models[k]['MALAR'] / impNumber
            MALAR_list = collect_Results(results, fieldname, 'MALAR')
            MALAR_variance = variance(MALAR_list) * Bfactor
            MALAR_CI_plus = MALAR_value
            MALAR_CI_minus = MALAR_value
            MALAR_std_dev = np.sqrt(MALAR_variance)
            # MALAR_CI_minus = round(MALAR_value - 1.96 * np.sqrt(MALAR_std_dev + MALAR_variance), 4)
            # MALAR_CI_plus = round(MALAR_value + 1.96 * np.sqrt(MALAR_std_dev + MALAR_variance), 4)
            MLAR_value = models[k]['MLAR'] / impNumber
            MLAR_list = collect_Results(results, fieldname, 'MLAR')
            MLAR_variance = variance(MLAR_list) * Bfactor
            MLAR_CI_minus = MLAR_value
            MLAR_CI_plus = MLAR_value
            MLAR_std_dev = np.sqrt(MLAR_variance)
            # MLAR_CI_minus = round(MLAR_value - 1.96 * np.sqrt(MLAR_std_dev + MLAR_variance), 4)
            # MLAR_CI_plus = round(MLAR_value + 1.96 * np.sqrt(MLAR_std_dev + MLAR_variance), 4)
            if False:
                print(fieldname, 'MAE:', round(mae_value, 6), "StdDev:", round(mae_std_dev, 6), "B: ",
                      round(mae_variance, 4), "  CI: [", mae_CI_minus, mae_CI_plus, "]", file=fileoutput)
                print(fieldname, 'PW20:', round(pw20_value, 6), "StdDev:", round(pw20_std_dev, 6), "B: ",
                      round(pw20_variance, 4), " CI: [", pw20_CI_minus, pw20_CI_plus, "]", file=fileoutput)
                print(fieldname, 'R2:', round(R2_value, 6), "StdDev:", round(R2_std_dev, 6), "B: ",
                      round(R2_variance, 4), " CI: [", R2_CI_minus, R2_CI_plus, "]", file=fileoutput)
                print(fieldname, 'MALAR:', round(MALAR_value, 6), "StdDev:", round(MALAR_std_dev, 6), "B: ",
                      round(MALAR_variance, 4), " CI: [", MALAR_CI_minus, MALAR_CI_plus, "]", file=fileoutput)
                print(fieldname, 'MLAR:', round(MLAR_value, 6), "StdDev", round(MLAR_std_dev, 6), "B :",
                      round(MLAR_variance, 4), " CI:", MLAR_CI_minus, MLAR_CI_plus, "]", file=fileoutput)
            else:
                print(fieldname, 'MAE:', round(mae_value, 6), file=fileoutput)
                print(fieldname, 'PW20:', round(pw20_value, 6), file=fileoutput)
                print(fieldname, 'R2:', round(R2_value, 6), file=fileoutput)
                print(fieldname, 'MALAR:', round(MALAR_value, 6), file=fileoutput)
                print(fieldname, 'MLAR:', round(MLAR_value, 6), file=fileoutput)


if __name__ == "__main__":
    main()
