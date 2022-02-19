import numpy as np
import pandas as pd
import warfit_learn
from warfit_learn.evaluation import evaluate_estimators
from warfit_learn import preprocessing
from warfit_learn.estimators import Estimator
from warfit_learn.metrics import score_mae, score_pw20, score_r2
import os

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler, Normalizer,QuantileTransformer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline, make_pipeline,make_union
from sklearn.svm import SVR
from math import sqrt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier # for extreme gradient boosting model
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

def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1

def DoseRound(predicted):
    return round(predicted / 2.5, 0) * 2.5

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

    number_of_samples = 1000
    number_of_imps = 100
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
            print(dfmod.info())
            X = dfmod
            Y = X['Dose_mg_week']

            X.drop(['Unnamed: 0', 'Unnamed: 0.1','.imp','.id','Dose_mg_week'], axis = 1, inplace = True)
            dfOriginal = pd.concat([X, Y],  axis=1, sort=False)
            validation_size = 0.20
            seed = 7
            X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,
                                                                        random_state=seed)
            num_folds = 10
            scoring = 'neg_mean_absolute_error'
            Ycopy = Y
            Xcopy = X
            Xcopy.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ScaleData\UnScaled.csv", ";")
            if False:
                mapper = DataFrameMapper([(Xcopy.columns, StandardScaler())])
                scaled_features = mapper.fit_transform(Xcopy.copy(), 4)
                scaled_features_df = pd.DataFrame(scaled_features, index=Xcopy.index, columns=Xcopy.columns)
                scaled_features_df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ScaleData\StdScalar.csv", ";")
            mapper = DataFrameMapper([(Xcopy.columns, MinMaxScaler())])
            scaled_features = mapper.fit_transform(Xcopy.copy(), 4)
            scaled_features_df = pd.DataFrame(scaled_features, index=Xcopy.index, columns=Xcopy.columns)
            scaled_features_df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ScaleData\MinMaxScalar.csv", ";")
        #mapper = DataFrameMapper([(Y.name, MinMaxScaler())])
        #scaled_response = mapper.fit_transform(Y.copy(), 4)
        #scaled_response_df = pd.DataFrame(scaled_response, index=Y.index, columns=Y.name)

            if False:
                mapper = DataFrameMapper([(Xcopy.columns, MaxAbsScaler())])
                scaled_features = mapper.fit_transform(Xcopy.copy(), 4)
                scaled_features_df = pd.DataFrame(scaled_features, index=Xcopy.index, columns=Xcopy.columns)
                scaled_features_df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ScaleData\MaxAbsScalar.csv", ";")
                mapper = DataFrameMapper([(Xcopy.columns, QuantileTransformer(output_distribution="uniform"))])
                scaled_features = mapper.fit_transform(Xcopy.copy(), 4)
                scaled_features_df = pd.DataFrame(scaled_features, index=Xcopy.index, columns=Xcopy.columns)
                scaled_features_df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ScaleData\QTUniform.csv", ";")
                mapper = DataFrameMapper([(Xcopy.columns, QuantileTransformer(output_distribution="normal"))])
                scaled_features = mapper.fit_transform(Xcopy.copy(), 4)
                scaled_features_df = pd.DataFrame(scaled_features, index=Xcopy.index, columns=Xcopy.columns)
                scaled_features_df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ScaleData\QTNormal.csv", ";")
                mapper = DataFrameMapper([(Xcopy.columns,Normalizer())])
                scaled_features = mapper.fit_transform(Xcopy.copy(), 4)
                scaled_features_df = pd.DataFrame(scaled_features, index=Xcopy.index, columns=Xcopy.columns)
                scaled_features_df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ScaleData\Normalize.csv", ";")

            models = []
            models.append(('LR', LinearRegression()))
            models.append(('LASSO', Lasso()))
            models.append(('ElasticNet', ElasticNet()))
            models.append(('KNN', KNeighborsRegressor()))
            models.append(('ClassificationAndRegressionTrees', DecisionTreeRegressor()))
            models.append(('SupportVectorRegression', SVR()))
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

            pipelines = []
            pipelines.append(('ScaledLR', Pipeline([('MinMaxScaler', MinMaxScaler()),('LR', LinearRegression())])))
            pipelines.append(('ScaledLASSO', Pipeline([('MinMaxScaler', MinMaxScaler()),('LASSO', Lasso())])))
            pipelines.append(('ScaledElasticNet', Pipeline([('MinMaxScaler', MinMaxScaler()), ('EN', ElasticNet())])))
            pipelines.append(('ScaledKNN', Pipeline([('MinMaxScaler', MinMaxScaler()), ('KNN', KNeighborsRegressor())])))
            pipelines.append(('ScaledClassificationAndRegressionTrees', Pipeline([('MinMaxScaler', MinMaxScaler()), ('CART', DecisionTreeRegressor())])))
            pipelines.append(('ScaledSupportVectorRegression', Pipeline([('MinMaxScaler', MinMaxScaler()), ('SVR', SVR())])))

            results = []
            names = []
            for name, model in pipelines:
                kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
                cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                print(msg)

            ests = []
            ests.append(Estimator(LinearRegression(),'LR'))
            ests.append(Estimator(Lasso(),'LASSO'))
            ests.append(Estimator(ElasticNet(), 'ElasticNet'))
            ests.append(Estimator(KNeighborsRegressor(), 'KNN'))
            ests.append(Estimator(DecisionTreeRegressor(), 'ClassificationAndRegressionTrees'))
            ests.append(Estimator(SVR(),'SupportVectorRegression'))

            evaluate_estimators(
                ests,
                dfOriginal,
                target_column='Dose_mg_week'  # @param {type:"string"}
                , scale=True
                , scaleMethod= 'MinMaxScaler()'
                , resamples=100  # @param {type:"slider", min:5, max:200, step:1}
                , test_size=0.2
                , squaring=False  # @param ["True", "False"] {type:"raw"}
                , technique='mccv'  # @param ["'bootstrap'", "'mccv'"] {type:"raw"}
                , parallelism=0.8  # @param {type:"slider", min:0.1, max:1.0, step:0.05}
            )

"""
A toolkit for reproducible research in warfarin dose estimation.
Copyright (C) 2019 Gianluca Truda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import List
from warfit_learn.estimators import Estimator
from warfit_learn.metrics import score_mae, score_pw20, score_r2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from tabulate import tabulate


def evaluate_estimator(estimator: Estimator,
                       data: pd.DataFrame,
                       technique='mccv',
                       target_column='Therapeutic Dose of Warfarin',
                       resamples=100,
                       test_size=0.2,
                       squaring=False,
                       parallelism=0.5):
    """Evaluation function for a single estimator.

    NOTE: You would typically not call this function directly, unless
    you are evaluating only a single type of estimator.

    Parameters
    ----------
    estimator : Estimator
        The Estimator object to train-evaluate.
    data : pd.DataFrame
        The data on which to train and evaluate.
    technique : str, optional
        The CV method to use. Either 'mccv' for monte-carlo CV or
        'bootstrap' for bootstrap resampling, by default 'mccv'.
    target_column : str, optional
        The name of the target column in the provided data,
        by default 'Therapeutic Dose of Warfarin'
    resamples : int, optional
        The number of times to resample and evaluate, by default 100.
        The more resamples performed, the more reliable the aggregated
        results.
    test_size : float, optional
        The fraction of the data to be used as the test/evaluation set,
        by default 0.2
    squaring : bool, optional
        Whether the predictions and truth values must be squared before
        comparson, by default False. Only enable this if you
        square-rooted your target variable to un-skew the distribution.
    parallelism : float, optional
        The fraction of your processors to parallelise the evaluation
        over, by default 0.5. Setting this to 1.0 will probably give you
        the fastest evaluation, but will demand all your CPU resources.

    Returns
    -------
    Dictionary
        A dictionary of lists, with results from each trial. Keys of the
        dictionary are ['PW20', 'MAE', 'R2'].
    """

    assert(technique in ['mccv', 'bootstrap'])
    results = {'PW20': [], 'MAE': [], 'R2': []}

    avail_cores = multiprocessing.cpu_count()
    num_cores = max(int(avail_cores * parallelism), 1)

    try:
        if technique == 'bootstrap':
            replace = True
        elif technique == 'mccv':
            replace = False

        res = Parallel(n_jobs=num_cores)(delayed(_train_eval)(
            estimator,
            data,
            target_column=target_column,
            test_size=test_size,
            squaring=squaring,
            replace=replace) for i in range(resamples))

        results['PW20'] = [r[0] for r in res]
        results['MAE'] = [r[1] for r in res]
        results['R2'] = [r[2] for r in res]

        return results

    except Exception as e:
        print("Error occurred:", e)
        return results


def evaluate_estimators(estimators: List[Estimator],
                        data: pd.DataFrame,
                        target_column='Therapeutic Dose of Warfarin',
                        scale=True,
                        scaleMethod = 'StandardScaler()',
                        parallelism=0.5,
                        *args,
                        **kwargs):
    """Evaluation function for a list of Estimators.

    Parameters
    ----------
    estimators : List[Estimator]
        A list of Estimator objects.
    data : pd.DataFrame
        The data on which to train and evaluate.
    target_column : str, optional
        The name of the target column in the provided data,
        by default 'Therapeutic Dose of Warfarin'
    scale : bool, optional
        Whether or not to scale the input features prior to training,
        by default True.
    parallelism : float, optional
        The fraction of your processors to parallelise the evaluation
        over, by default 0.5. Setting this to 1.0 will probably give you
        the fastest evaluation, but will demand all your CPU resources.
    technique : str, optional
        The CV method to use. Either 'mccv' for monte-carlo CV or
        'bootstrap' for bootstrap resampling, by default 'mccv'.
    resamples : int, optional
        The number of times to resample and evaluate, by default 100.
        The more resamples performed, the more reliable the aggregated
        results.
    test_size : float, optional
        The fraction of the data to be used as the test/evaluation set,
        by default 0.2
    squaring : bool, optional
        Whether the predictions and truth values must be squared before
        comparson, by default False. Only enable this if you
        square-rooted your target variable to un-skew the distribution.

    Returns
    -------
    pd.DataFrame
        Dataframe of results with the name of the Estimator, and the
        results in terms of MAE, PW20, and R2.
    """

    _data = data.copy()

    avail_cores = multiprocessing.cpu_count()
    num_cores = max(int(avail_cores * parallelism), 1)
    print(f'Using {num_cores} / {avail_cores} CPU cores...')

    if scale:
        x_cols = list(_data.columns)
        x_cols.remove(target_column)
        scaler = MinMaxScaler()
        _data[x_cols] = scaler.fit_transform(_data[x_cols])

    results = []
    for _, est in enumerate(estimators):
            print(f'\n{est.identifier}...')
            res = evaluate_estimator(est, _data,
                                     target_column=target_column,
                                     *args, **kwargs)
            res_dict = {
                'Estimator': [est.identifier for x in range(len(res['PW20']))],
                'PW20': res['PW20'],
                'MAE': res['MAE'],
                'R2': res['R2'],
            }
            prog = {k: [np.mean(res_dict[k])]
                    for k in list(res_dict.keys())[1:]}
            print(tabulate(prog, headers=prog.keys()))
            results.append(res_dict)

    # Compile results to single DF
    df_res = pd.DataFrame()
    for res in results:
        df_res = df_res.append(pd.DataFrame.from_dict(res))

    print(f"\n\n{df_res.groupby(['Estimator']).agg(np.mean)}\n")

    return df_res


def _train_eval(estimator: Estimator,
                data,
                test_size,
                target_column,
                squaring,
                replace=False):
    """Trains and evaluates a single Estimator for one iteration.

    NOTE: This should not be called directly by the user.
    """

    train, test = train_test_split(data, test_size=test_size)
    if replace:
        # Bootstrap resampling
        train = resample(train, replace=True)
    y_train = train[target_column].values
    x_train = train.drop([target_column], axis=1).values
    y_test = test[target_column].values
    # Square the dose (to undo upstream sqrt call)
    if squaring:
        y_test = np.square(y_test)
    x_test = test.drop([target_column], axis=1).values
    estimator.fit(x_train, y_train)
    predicts = estimator.predict(x_test)
    suffix   = estimator.identifier.strip()
    print(suffix, 'DOSETRUE',y_test)
    print(suffix, 'DOSEPRED', predicts)
    if squaring:
        predicts = np.square(predicts)
    pd.DataFrame(y_test).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\DoseTrue" + suffix + ".csv", ";")
    pd.DataFrame(predicts).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\DosePred" + suffix + ".csv", ";")

    return (
        score_pw20(y_test, predicts),
        score_mae(y_test, predicts),
        RSquared(y_test, predicts))

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.validation import check_array, check_consistent_length, _num_samples
from sklearn.utils.validation import column_or_1d
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.stats import _weighted_percentile

def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """:math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a :math:`R^2` score of 0.0.

    Read more in the :ref:`User Guide <r2_score>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, \
            array-like of shape (n_outputs,) or None, default='uniform_average'

        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".

        'raw_values' :
            Returns a full set of scores in case of multioutput input.

        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

        .. versionchanged:: 0.19
            Default value of multioutput is 'uniform_average'.

    Returns
    -------
    z : float or ndarray of floats
        The :math:`R^2` score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function.

    Unlike most other scores, :math:`R^2` score may be negative (it need not
    actually be the square of a quantity R).

    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_

    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred,
    ...          multioutput='variance_weighted')
    0.938...
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [2, 2, 2]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [3, 2, 1]
    >>> r2_score(y_true, y_pred)
    -3.0
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    check_consistent_length(y_true, y_pred, sample_weight)


    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")


    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.0


    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (
        weight * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2
    ).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput


    return np.average(output_scores, weights=avg_weights)




if __name__ == "__main__":
    main()


