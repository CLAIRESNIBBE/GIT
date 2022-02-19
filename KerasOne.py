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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier # for extreme gradient boosting model
from warfit_learn.estimators import Estimator
from warfit_learn.evaluation import evaluate_estimators

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
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, \
    MinMaxScaler, FunctionTransformer, Normalizer
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoLarsCV, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, make_union
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_approximation import RBFSampler, Nystroem
from tpot.builtins import StackingEstimator, ZeroCount
import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
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


def INRThree(targetINR):
    if (targetINR >= 2.5).all() & (targetINR <= 3.5).all():
        return 1
    else:
        return 0


def main():
    # Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
    #  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                            inter_op_parallelism_threads=2,
                            allow_soft_placement=True,
                            device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

    session = tf.Session(config=config)


    dftemplate = pd.DataFrame()
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
        dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
        dfmod.drop(["Height_cm"], axis = 1, inplace = True)
        dfmod.drop(["Weight_kg"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0"],axis=1, inplace = True)
        dfmod.drop(["Unnamed: 0.1"],axis=1, inplace = True)
        dfmod.drop(["Age_years"], axis = 1, inplace = True)
        dfmod.drop([".imp"], axis=1, inplace=True)
        dfmod.drop([".id"], axis=1, inplace=True)
        if True:
            print("On imputation ", df)
            data = dfmod
            print(data.shape)
            data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
            estimates = []
            #LR = LinearRegression(fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False)
            LR = LinearRegression()
            BRT = GradientBoostingRegressor(loss = 'ls', learning_rate=0.1,n_estimators=100)
            GBT = GradientBoostingRegressor(learning_rate=0.1, loss='lad', max_depth=4)
            RR = Ridge(alpha=1.0)
            NN = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", random_state=1, max_iter=2000)
            #NN = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', max_iter=1000)
            SV = SVR(kernel='linear', cache_size=1000)
            RF = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
            CUBE = Cubist()
            LAS = Lasso(alpha=0.1)
            EL = ElasticNet(random_state=0)
            n_trees = 100
            mdepth = 6
            gamma = 1
            lam = 1
            XG = XGBClassifier(
                learning_rate=0.1,
                n_estimators=1000,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                nthread=4,
                scale_pos_weight=1,
                seed=27,
                num_class=3,
            )
            #BAG = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)

            #XG = XGBClassifier(use_label_encoder=False,
            #                      booster='gbtree',  # boosting algorithm to use, default gbtree, othera: gblinear, dart
            #                      n_estimators=n_trees,  # number of trees, default = 100
            #                      eta=0.3,  # this is learning rate, default = 0.3
            #                      max_depth=mdepth,  # maximum depth of the tree, default = 6
            #                      gamma=gamma,
            #                      # used for pruning, if gain < gamma the branch will be pruned, default = 0
            #                      reg_lambda=lam,  # regularization parameter, defautl = 1
            #                      # min_child_weight=0 # this refers to Cover which is also responsible for pruning if not set to 0
            #                      )
            estimates.append(Estimator(LR, 'LR'))
            estimates.append(Estimator(BRT,'BRT'))
            estimates.append(Estimator(GBT, 'GBT'))
            estimates.append(Estimator(RF, 'RF'))
            estimates.append(Estimator(CUBE, 'Cubist'))
            estimates.append(Estimator(XG, 'XGB'))
            estimates.append(Estimator(NN, 'NN'))
            estimates.append(Estimator(RR, 'RR'))
            estimates.append(Estimator(SV, 'SV'))
            estimates.append(Estimator(EL,'EL'))
            estimates.append(Estimator(LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive'), 'SVR'))
            estimates.append(
                Estimator(StackingCVRegressor(regressors=[GBT, RR, NN], meta_regressor=RR, cv=5, ), 'Stacked_RR'))
            estimates.append(
                Estimator(StackingCVRegressor(regressors=[GBT, SV, NN], meta_regressor=SV, cv=5, ), 'Stacked_SV'))
            #estimates.append(Estimator(BAG, 'Bag'))
            estimates.append(Estimator(LAS, 'Lasso'))
            tpot2 = make_pipeline(
                StackingEstimator(
                estimator=LinearSVR(
                C=1.0,
                dual=True,
                epsilon=0.01,
                loss="epsilon_insensitive",
                tol=0.001, )),
                StackingEstimator(
                estimator=ElasticNetCV(l1_ratio=0.6000000000000001, tol=0.01, cv=5)),
                RobustScaler(),
                StackingEstimator(estimator=RidgeCV()),
                ExtraTreesRegressor(
                bootstrap=True,
                max_features=1.0,
                min_samples_leaf=20,
                min_samples_split=2,
                n_estimators=100, )
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
                                      n_estimators=100, subsample=0.55)
            )
            estimates.append(Estimator(tpot2, 'TPOT2'))
            estimates.append(Estimator(tpot10, 'TPOT10'))
            estimates.append(Estimator(tpot17, 'TPOT17'))
            estimates = []
            estimates.append(Estimator(LR, 'LR'))
            # define the keras model
            model = Sequential()
            model.add(Dense(12, input_dim=8, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # compile the keras model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X, y, epochs=150, batch_size=10, verbose=0)
            # make class predictions with the model
            predictions = (model.predict(X) > 0.5).astype(int)
            # summarize the first 50 cases
            for i in range(50):
              print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))



            warpath_results = evaluate_estimators(estimates,
                                          data,
                                          target_column='Dose_mg_week'
                                          , scale = True
                                          , test_size=0.2
                                          , squaring=True
                                          , technique='mccv'
                                          , parallelism=0.8
                                          )
            print(warpath_results)
            summary = warpath_results.groupby('Estimator').apply(np.mean)
            print(summary)
            dftemplate=dftemplate.append(summary)
            warpath_formatted = format_summary(warpath_results)
            df_final = pd.concat([warpath_formatted], axis=1, keys=['WARPATH'])
            print(df_final)
            suffix = str(df).zfill(3)
            df_final.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_" + suffix + ".csv", ";")
            #model.fit(X, y, epochs=150, batch_size=10, verbose=0)
            # make class predictions with the model
            #predictions = (model.predict(X) > 0.5).astype(int)
            # summarize the first 50 cases
            #for i in range(50):
             #   print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))




    dftemplate.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dftemplate" + ".csv", ";")
    dfSummary = dftemplate.groupby('Estimator').apply(np.mean)
    dfSummary_formatted = format_summary(dfSummary)

if __name__ == "__main__":
    main()
