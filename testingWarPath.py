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
from warfit_learn.estimators import Estimator
from warfit_learn.evaluation import evaluate_estimators
from warfit_learn.metrics.scoring import confidence_interval
from scipy.stats import norm
from mlxtend.regressor import StackingCVRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import ChangedBehaviorWarning
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ChangedBehaviorWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
from sklearn.feature_selection import SelectFwe, f_regression, SelectPercentile
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, \
    MinMaxScaler, FunctionTransformer, Normalizer
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoLarsCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, make_union
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_approximation import RBFSampler, Nystroem
from tpot.builtins import StackingEstimator, ZeroCount
from copy import copy

def format_summary(df_res):
    df_summary = df_res.groupby(['Estimator']).mean()
    df_summary.reset_index(inplace=True)
    for alg in df_res['Estimator'].unique():
        for metric in ['PW20','MAE']:
            lo, hi = confidence_interval(df_res[metric][df_res['Estimator'] == alg].values,)
            mean = df_res[metric][df_res['Estimator'] == alg].mean()
            for v in [mean,lo,hi]:
                if not (-10000 < v < 10000):
                    print('nan applied: ', alg, metric, lo, hi, mean)
                    mean, lo, hi = np.nan,np.nan,np.nan
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
        if (metrics[i]['model']== model):
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

def main():
    impNumber = 100
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_rows", False)
    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    for imp in range(impNumber):
        counter = imp+1
        dfcurrent = df.loc[df[".imp"] == counter]
        suffix = str(counter).zfill(3)
        dfcurrent.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATHIMP\ImpWarPATH_" + suffix + ".csv", ";")
    filesImp = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv'):
                filesImp.append(file)
    for file in filesImp:
        dfnew = pd.read_csv(root + '\\' + file, ";")
        df = filesImp.index(file) + 1
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
        dfmod["AgeDecades"] = np.floor(dfmod["Age_years"] * 0.1).astype("int")
        dfmod["AgeYears"] = dfmod["Age_years"]
        dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
        dfmod["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
        dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
        dfmod.drop(["HIV_status"], axis=1, inplace=True)
        dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
        if df==1:
            print(data.shape)
            data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
            estimates=[]
            estimates.append(Estimator(GradientBoostingRegressor(loss='ls', learning_rate = 0.1,
                n_estimators = 100),'BRT'))
            GBT = GradientBoostingRegressor(learning_rate = 0.1, loss = 'lad',    max_depth = 4)
            RR = Ridge(alpha= 1.0)
            NN = MLPRegressor(hidden_layer_sizes=(100,),activation='logistic',  solver='lbfgs', max_iter=1000)
            SV = SVR(kernel = 'linear',cache_size=1000)
            estimates.append(Estimator(GBT,'GBT'))
            estimates.append(Estimator(LinearRegression(normalize='deprecated', fit_intercept=True), 'LR'))
            estimates.append( Estimator(NN,'NN'))
            estimates.append(Estimator(RR,'RR'))
            estimates.append(Estimator(SV,'SV'))
            estimates.append(Estimator(LinearSVR(epsilon=0.0,tol=0.0001, C=1.0,loss='epsilon_insensitive'), 'SVR'))
            estimates.append(Estimator(StackingCVRegressor(regressors=[GBT,RR,NN],meta_regressor=RR, cv=5,),'Stacked_RR'))
            estimates.append(Estimator(StackingCVRegressor(regressors=[GBT,SV,NN], meta_regressor=SV, cv=5,),'Stacked_SV'))




    tpot2 = make_pipeline(
    StackingEstimator(
        estimator=LinearSVR(
            C=1.0,
            dual=True,
            epsilon=0.01,
            loss="epsilon_insensitive",
            tol=0.001,)),
    StackingEstimator(
        estimator=ElasticNetCV(l1_ratio=0.6000000000000001, tol=0.01, cv=5)),
    RobustScaler(),
    StackingEstimator(estimator=RidgeCV()),
    ExtraTreesRegressor(
        bootstrap=True,
        max_features=1.0,
        min_samples_leaf=20,
        min_samples_split=2,
        n_estimators=100,)
    )

    tpot10 = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(
        bootstrap=True, max_features=0.05,
        min_samples_leaf=18, min_samples_split=10,
        n_estimators=100)),
    MaxAbsScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(
        bootstrap=True, max_features=0.05,
        min_samples_leaf=18, min_samples_split=10, n_estimators=100)),
    LassoLarsCV(normalize=True, cv=3)
    )

    tpot17 = make_pipeline(
    make_union(
        FunctionTransformer(copy, validate=True),
        MaxAbsScaler()
    ),
    StackingEstimator(estimator=RidgeCV()),
    ZeroCount(),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="lad",
                              max_depth=3, max_features=0.9000000000000001,
                              min_samples_leaf=20, min_samples_split=8,
                              n_estimators=100, subsample=0.55)
    )
    estimates.append(Estimator(tpot2, 'TPOT2'))
    estimates.append(Estimator(tpot10, 'TPOT10'))
    estimates.append(Estimator(tpot17, 'TPOT17'))
    warpath_results = evaluate_estimators( estimates,
    data,
    target_column='Therapeutic Dose of Warfarin' #@param {type:"string"}
    ,scale=True
    ,resamples = 100 #@param {type:"slider", min:5, max:200, step:1}
    ,test_size=0.2
    ,squaring = True #@param ["True", "False"] {type:"raw"}
    ,technique = 'mccv' #@param ["'bootstrap'", "'mccv'"] {type:"raw"}
    ,parallelism = 0.8 #@param {type:"slider", min:0.1, max:1.0, step:0.05}
    )
    print(warpath_results)
    summary = warpath_results.groupby('Estimator').apply(np.mean)
    print(summary)
    iwpc_formatted = format_summary(warpath_results)
    df_final = pd.concat([iwpc_formatted], axis=1, keys = ['IWPC'])
    print(df_final)

    df_final.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH'001dec.csv",";")



if __name__ == "__main__":
    main()
