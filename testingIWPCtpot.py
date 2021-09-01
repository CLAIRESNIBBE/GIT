import sklearn
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

raw_iwpc = datasets.load_iwpc()
data = preprocessing.prepare_iwpc(raw_iwpc)
print(data.shape)
#data.drop(['INR on Reported Therapeutic Dose of Warfarin'], axis='columns', inplace=True)
data['Therapeutic Dose of Warfarin'] = data['Therapeutic Dose of Warfarin'].apply(np.sqrt)
estimates=[]
estimates.append(Estimator(GradientBoostingRegressor(loss='ls', learning_rate = 0.1,
    n_estimators = 100),'BRT'))
GBT = GradientBoostingRegressor(learning_rate = 0.1, loss = 'lad',
    max_depth = 4)
RR = Ridge(alpha= 1.0)
#NN = MLPRegressor(hidden_layer_sizes=(100,),activation='logistic',/
#  solver='lbfgs')
SV = SVR(kernel = 'linear',cache_size=1000)
estimates.append(Estimator(GBT,'GBT'))
estimates.append(Estimator(LinearRegression(normalize=False, fit_intercept=True), 'LR'))
#estimates.append( Estimator(NN,'NN'))
estimates.append(Estimator(RR,'RR'))
estimates.append(Estimator(SV,'SV'))
estimates.append(Estimator(LinearSVR(epsilon=0.0,tol=0.0001, C=1.0,
loss='epsilon_insensitive'), 'SVR'))
#estimates.append(Estimator(StackingCVRegressor(regressors=[GBT,RR,NN],meta_regressor=RR,
 #   cv=5,),'Stacked_RR'))
#estimates.append(Estimator(StackingCVRegressor(regressors=[GBT,SV,NN], meta_regressor=SV,
 #  cv=5,),'Stacked_SV'))

iwpc_results = evaluate_estimators(
   estimates,
   data,
  target_column='Therapeutic Dose of Warfarin' #@param {type:"string"}
 ,scale=True
   ,resamples = 100 #@param {type:"slider", min:5, max:200, step:1}
   ,test_size=0.2
 ,squaring = True #@param ["True", "False"] {type:"raw"}
  ,technique = 'mccv' #@param ["'bootstrap'", "'mccv'"] {type:"raw"}
 ,parallelism = 0.8 #@param {type:"slider", min:0.1, max:1.0, step:0.05}
)

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


iwpc_formatted = format_summary(iwpc_results)
df_final = pd.concat([iwpc_formatted], axis=1, keys = ['IWPC'])
print(df_final)


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


# Trained on PathCare data
tpot_06_12 = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=56),
    MinMaxScaler(),
    RBFSampler(gamma=0.75),
    MinMaxScaler(),
    PCA(iterated_power=2, svd_solver="randomized"),
    PCA(iterated_power=2, svd_solver="randomized"),
    KNeighborsRegressor(n_neighbors=96, p=1, weights="distance")
)

# Trained on PathCare data
tpot_06_12_02 = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_union(
            make_pipeline(
                Normalizer(norm="l1"),
                Nystroem(gamma=0.65, kernel="sigmoid", n_components=3),
                SelectFwe(score_func=f_regression, alpha=0.006)
            ),
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            )
        )
    ),
    StackingEstimator(estimator=LinearSVR(C=0.0001, dual=True,
                                          epsilon=0.0001,
                                          loss="epsilon_insensitive",
                                          tol=1e-05)),
    MinMaxScaler(),
    SelectFwe(score_func=f_regression, alpha=0.047),
    Nystroem(gamma=0.9, kernel="polynomial", n_components=6),
    ZeroCount(),
    Nystroem(gamma=0.9500000000000001, kernel="linear", n_components=7),
    KNeighborsRegressor(n_neighbors=99, p=1, weights="distance")
)

# Trained on PathCare data
tpot_06_12_03 = make_pipeline(
    StandardScaler(),
    SelectFwe(score_func=f_regression, alpha=0.047),
    MinMaxScaler(),
    Nystroem(gamma=0.5, kernel="poly", n_components=5),
    StackingEstimator(estimator=DecisionTreeRegressor(
        max_depth=2, min_samples_leaf=12, min_samples_split=5)),
    MaxAbsScaler(),
    SelectFwe(score_func=f_regression, alpha=0.007),
    KNeighborsRegressor(n_neighbors=100, p=1, weights="distance")
)

# Trained on PathCare data (sqrted)
tpot_06_13_02 = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    SelectPercentile(score_func=f_regression, percentile=49),
    MinMaxScaler(),
    RBFSampler(gamma=0.45),
    StackingEstimator(estimator=KNeighborsRegressor(
        n_neighbors=43, p=1, weights="distance")),
    MaxAbsScaler(),
    KNeighborsRegressor(n_neighbors=97, p=1, weights="distance")
)

# Trained on PathCare data (sqrted + not-LITE)
tpot_06_13_01 = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.011),
    StackingEstimator(estimator=ElasticNetCV(
        l1_ratio=0.7000000000000001, tol=0.01)),
    MaxAbsScaler(),
    Nystroem(gamma=0.2, kernel="linear", n_components=3),
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=100, p=1, weights="distance")
)

estimates.append(Estimator(tpot2, 'TPOT2'))
estimates.append(Estimator(tpot10, 'TPOT10'))
estimates.append(Estimator(tpot17, 'TPOT17'))
estimates.append(Estimator(tpot_06_12, 'TPOT_06_12'))
estimates.append(Estimator(tpot_06_12_02, 'TPOT_06_12_02'))
estimates.append(Estimator(tpot_06_12_03, 'TPOT_06_12_03'))
estimates.append(Estimator(tpot_06_13_01, 'TPOT_06_13_01'))
estimates.append(Estimator(tpot_06_13_02, 'TPOT_06_13_02'))
#iwpc_results = evaluate_estimators(
 #   estimates,
  #  data,
  #  parallelism=0.5,
   # resamples=100,
#)
iwpc_results = evaluate_estimators(
   estimates,
   data,
  target_column='Therapeutic Dose of Warfarin' #@param {type:"string"}
 ,scale=True
   ,resamples = 100 #@param {type:"slider", min:5, max:200, step:1}
   ,test_size=0.2
 ,squaring = True #@param ["True", "False"] {type:"raw"}
  ,technique = 'mccv' #@param ["'bootstrap'", "'mccv'"] {type:"raw"}
 ,parallelism = 0.8 #@param {type:"slider", min:0.1, max:1.0, step:0.05}
)
print(iwpc_results)
summary = iwpc_results.groupby('Estimator').apply(np.mean)
print(summary)





