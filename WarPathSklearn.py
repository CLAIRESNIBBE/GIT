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
from copy import copy

train = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarTrainErrors"  + ".csv", ";")
test = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarTestErrors"  + ".csv", ";")
target_column = "Dose_mg_week"
ytrainfile = open(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARTRAINy"  + ".csv")
y_train = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARTRAINy"  + ".csv",";")
xtrainfile = open(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARTRAINx"  + ".csv")
x_train = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARTRAINx"  + ".csv",";")
x_test = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARTESTx"  + ".csv",";")
#y_train = train[target_column].values
#x_train = train.drop([target_column], axis=1).values
#x_test = test.drop([target_column], axis=1).values
#y_test = test[target_column].values
#y_test = np.square(y_test)
estimator = Estimator(estimator=LinearRegression(), identifier='LR')
estimator.fit(x_train, y_train)
predicts = estimator.predict(x_test)
predicts = np.square(predicts)
predicts = predicts[:,1]
print(max(predicts))
predictsDF = pd.DataFrame(predicts)
predictsDF.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPredicts"  + ".csv", ";")









