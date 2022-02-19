import csv
import pandas as pd
import numpy as np
import os
import time
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import tree
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
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


def MAPE(trueval, predval):
    topsum = 0
    for i in range(len(trueval)):
        topsum += abs((predval[i] - trueval[i]) / trueval[i])
    return topsum / len(trueval) * 100


def MLAR(trueval, predval):
    # mean log of accuracy ratio
    sum = 0
    for i in range(len(trueval)):
        sum += np.log(predval[i] / trueval[i])
    return (np.exp(sum / len(trueval)) - 1) * 100


def MALAR(trueval, predval):
    # mean absolute log of absolute ratio
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


def format_summary(df_res):
    df_summary = df_res.groupby(['Estimator']).mean()
    df_summary.reset_index(inplace=True)
    for alg in df_res['Estimator'].unique():
        for metric in ['PW20', 'MAE', 'R2']:
            lo, hi = confidence_interval(
                df_res[metric][df_res['Estimator'] == alg].values,
            )
            mean = df_res[metric][df_res['Estimator'] == alg].mean()

            for v in [mean, lo, hi]:
                if not -10000 < v < 10000:
                    mean, lo, hi = np.nan, np.nan, np.nan

            conf = f"{mean:.2f} ({lo:.2f}–{hi:.2f})"
            df_summary[metric][df_summary['Estimator'] == alg] = conf

    return df_summary


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


def SList(series):
    return np.array(series.values.tolist())


def main():
    filewritename = input("Enter file name: \n")
    fileoutput = open(filewritename, 'w')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    dfmod = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Blacks.csv", ";")
    dfmod.drop(['Gender'], axis=1, inplace=True)

    factor_IWPC = {"Age": -0.2546, "Height": 0.0118, "Weight": 0.0134, "Inducer": 1.2799, "Amiodarone": -0.5695,
                   "Intercept": 4.4436}
    factor_Gage = {"Age": -0.0075, "BSA": 0.425, "TargetINR": 0.216, "Smoker": 0.108, "Amiodarone": -0.257,
                   "Indication": 0.0784, "Intercept": 0.769}
    factor_Fixed = {"Intercept": 35}
    factor_WarPath = {'Intercept': 23.6886, 'Age': -0.0656, 'Weight': 0.2178, 'TargetINR': 7.3190}
    number_of_samples = 100
    number_of_imps = 100
    filelist = [f for f in os.listdir(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL") if f.endswith(".csv")]
    for f in filelist:
        os.remove(os.path.join(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL", f))
    results = []
    std_Dev = []
    std_Dev_Summ = ({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                    {'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                    {'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                    {'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0})

    dfmod["AgeLower"] = dfmod["Age_years"].str.split('_', expand=True)[0]
    dfmod['AgeLower'] = dfmod['AgeLower'].str.replace('+', '')
    dfmod["AgeDecades"] = np.floor(dfmod["AgeLower"].astype("float") * 0.1).astype("int")
    dfmod["AgeYears"] = dfmod["AgeLower"].astype("float")
    dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
    dfmod.drop(['AgeLower'], axis=1, inplace=True)
    dfmod.drop(['Age_years'], axis=1, inplace=True)
    dfmod["Weight_kg"] = dfmod["Weight_kg"].astype("float")
    dfmod["Height_cm"] = dfmod["Height_cm"].astype("float")
    dfmod["Target_INR"] = dfmod["Target_INR"].astype("float")
    dfmod["INR_Three"] = np.where(dfmod["Target_INR"] >= 3.0, 1, 0)
    dfmod["Dose_mg_week"] = dfmod["Dose_mg_week"].astype("float")
    dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
    dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
    dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
    dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
    dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)

    imp = mice.MICEData(dfmod)
    resultsImp = []
    for j in range(number_of_imps):
        imp.update_all()
        x = imp.next_sample()
        resultsImp.append(x)
        suffix = str(j).zfill(3)
        x.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL\IWPC_MICE_" + str(suffix) + ".csv", ";")

    filesImp = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL"):
        for file in files:
            if file.endswith('.csv'):
                filesImp.append(file)

    impNumber = len(filesImp)
    rng = np.random.RandomState(1)
    modelsk_LR = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    modelsk_LAR = linear_model.Lasso(alpha=0.1, fit_intercept=True)
    modelsk_RR = linear_model.Ridge(alpha=0.1, fit_intercept=True)
    modelsk_GBR = ensemble.GradientBoostingRegressor(learning_rate=0.1, loss="lad", max_depth=4, )
    modelsk_DTR = tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=12, min_samples_split=5)
    modelsk_GBT = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=4), n_estimators=300,
                                             random_state=rng)
    modelsk_SVR = make_pipeline(StandardScaler(), svm.SVR(C=1.0, epsilon=0.2))
    modelsk_RF = RandomForestRegressor(n_estimators=100, random_state=1)
    modelsk_NN = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs')
    MLearning = []

    for file in filesImp:
        dfnew = pd.read_csv(root + '\\' + file, ";")
        df = filesImp.index(file) + 1
        print("on imputation", df)
        suffix = str(df).zfill(3)
        dfmod = dfnew
        impResults = []
        models = []
        MLearning = []
        sc = StandardScaler()
        dfmod.drop(['Unnamed: 0'], axis = 1, inplace = True)
        dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_" +suffix +".csv",";")
        X = dfmod[dfmod.columns.difference(['Dose_mg_week'])]
        y = dfmod['Dose_mg_week']
        X.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_X_" + suffix +".csv", ";")
        y.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_y_" + suffix + ".csv", ";")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
        pd.DataFrame(X_train).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_Xtrain_" + suffix + ".csv", ";")
        pd.DataFrame(X_test).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_Xtest_" + suffix + ".csv", ";")
        pd.DataFrame(y_test).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_ytest_" + suffix + ".csv", ";")
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        pd.DataFrame(X_train).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_Xtrainscaled_" + suffix + ".csv", ";")
        pd.DataFrame(X_test).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_Xtestscaled_" + suffix + ".csv", ";")
        y_train = y_train.values.reshape((len(y_train), 1))  # reshaping to fit the scaler
        y_train = sc.fit_transform(y_train)
        y_train = y_train.ravel()
        pd.DataFrame(y_train).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_ytrainscaled_" + suffix + ".csv", ";")
        pd.DataFrame(X_train).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_XtrainPreFit_" + suffix + ".csv", ";")
        pd.DataFrame(y_train).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_ytrainPreFit_" + suffix + ".csv", ";")

        modelsk_LR.fit(X_train, y_train)
        modelsk_LAR.fit(X_train, y_train)
        modelsk_RR.fit(X_train, y_train)
        modelsk_GBR.fit(X_train, y_train)
        modelsk_DTR.fit(X_train, y_train)
        modelsk_GBT.fit(X_train, y_train)
        modelsk_SVR.fit(X_train, y_train)
        modelsk_RF.fit(X_train, y_train)
        modelsk_NN.fit(X_train, y_train)

        y_test = y_test.tolist()

        y_predicted_Linear = modelsk_LR.predict(X_test)
        y_pred_Linear = y_predicted_Linear.tolist()
        intercept = modelsk_LR.intercept_
        coefficients = modelsk_LR.coef_.tolist()
        roundlist = [round(num, 4) for num in coefficients]
        #print("Imputation ", df, " Linear coefficients ", roundlist, " Intercept ", intercept, file=fileoutput)
        y_predicted_Lasso = modelsk_LAR.predict(X_test)
        y_pred_Lasso = y_predicted_Lasso.tolist()
        coefficients = modelsk_LAR.coef_.tolist()
        roundlist = [round(num, 4) for num in coefficients]
        intercept = modelsk_LAR.intercept_
        # print("Lasso coefficients ", roundlist, " Intercept ", intercept, file=fileoutput)
        y_predicted_Ridge = modelsk_RR.predict(X_test)
        y_pred_Ridge = y_predicted_Ridge.tolist()
        coefficients = modelsk_RR.coef_.tolist()
        roundlist = [round(num, 4) for num in coefficients]
        intercept = modelsk_RR.intercept_
        # print("Ridge coefficients ", roundlist, " Intercept ", intercept, file=fileoutput)
        y_predicted_GBR = modelsk_GBR.predict(X_test)
        y_pred_GBR = y_predicted_GBR.tolist()
        # coefficients = modelsk_GBR.coef_
        # print("GBR coefficients ", coefficients)
        y_predicted_SVR = modelsk_SVR.predict(X_test)
        y_pred_SVR = y_predicted_SVR.tolist()
        # coefficients = modelsk_SVR
        # print("SVR coefficients ", coefficients)
        y_predicted_DTR = modelsk_DTR.predict(X_test)
        y_pred_DTR = y_predicted_DTR.tolist()
        # coefficients = modelsk_DTR.coef_
        # print("DTR coefficients ", coefficients)
        y_predicted_GBT = modelsk_GBT.predict(X_test)
        y_pred_GBT = y_predicted_GBT.tolist()
        y_predicted_RF = modelsk_RF.predict(X_test)
        y_pred_RF = y_predicted_RF.tolist()

        pd.DataFrame(y_pred_RF).to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearnIWPC_ypredictedRF_" + suffix + ".csv", ";")

        # coefficients = modelsk_GBT
        # print("GBT coefficients ", coefficients)
        # Initializing Level One Regressorsxgbr = XGBRegressor()
        #
        xgbr = XGBRegressor()
        rf = RandomForestRegressor(n_estimators=100, random_state=1)
        SV = SVR(kernel='linear', cache_size=1000)
        # Stacking the various regressors initialized before
        stack = StackingCVRegressor(regressors=(xgbr, rf, modelsk_LR), meta_regressor=xgbr,
                                    use_features_in_secondary=True)


        stack.fit(X_train, y_train)
        y_pred_stack = stack.predict(X_test)
        stack2 = StackingCVRegressor(regressors=(rf, modelsk_NN, modelsk_DTR), meta_regressor=rf, use_features_in_secondary=True)
        stack2.fit(X_train, y_train)
        y_pred_stack2 = stack2.predict(X_test)
        stack3 = StackingCVRegressor(regressors=(rf, modelsk_DTR), meta_regressor=rf, use_features_in_secondary=True)
        stack3.fit(X_train, y_train)
        y_pred_stack3 = stack3.predict(X_test)
        stack4 = StackingCVRegressor(regressors=(modelsk_GBR, modelsk_NN, modelsk_RR), meta_regressor=modelsk_RR,
                                     use_features_in_secondary=True)
        stack4.fit(X_train, y_train)
        y_pred_stack4 = stack4.predict(X_test)
        stack5 = StackingCVRegressor(regressors=(SV, modelsk_GBR, modelsk_NN), meta_regressor=SV,
                                     use_features_in_secondary=True)
        stack5.fit(X_train, y_train)
        y_pred_stack5 = stack5.predict(X_test)

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
                {'Imp': df, 'model': skmodel, 'MAE': mae, 'PW20': pw20, 'R2': R2, 'MLAR': bias, 'MALAR': mape,'y-true':y_test, 'y-pred':y_pred})

        suffix = str(df).zfill(3)
        impResultsPD = pd.DataFrame(impResults)
        impResultsPD.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MLearn" + suffix+".csv",";")

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


        for k in range(len(impResults)):
            a = impResults[k]['model']
            b = impResults[k]['Imp']
            c = impResults[k]['MAE']
            d = impResults[k]['PW20']
            e = impResults[k]['R2']
            g = impResults[k]['MLAR']
            h = impResults[k]['MALAR']
            results.append({'Imp': b, 'model': a, 'MAE': c, 'PW20': d, 'R2': e,  'MLAR': g, 'MALAR': h})

        if False:
            while boot < number_of_samples:
                print("imputation ", df, " on sample ", boot)
                dfsample = dfmod.sample(n=690, frac=None, replace=True)
                dfMetricfactors = dfsample[["WarPATH_Pred", "IWPC_Pred", "Fixed_Pred", "Gage_Pred", "Dose_mg_week"]]
                dfsample.to_csv(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imp" + str(df) + "Samp" + str(boot) + ".csv", ";")
                boot = boot + 1
                samples.append(dfMetricfactors)
                for m in range(len(listmodels)):
                    model = listmodels[m]
                    doseTrue = dfMetricfactors["Dose_mg_week"]
                    dosePred2 = dfMetricfactors[model + '_Pred']
                    curr_MAE = score_mae(doseTrue, dosePred2)
                    curr_PW20 = PercIn20(doseTrue, dosePred2)
                    curr_R2 = RSquared(doseTrue, dosePred2)
                    curr_MALAR = MALAR(doseTrue, dosePred2)
                    curr_MLAR = MLAR(doseTrue, dosePred2)
                    smpResults.append({'Imp': df, 'Sample': boot, 'model': model, 'MAE': curr_MAE, 'PW20': curr_PW20,
                                       'R2': curr_R2, 'MALAR': curr_MALAR, 'MLAR': curr_MLAR})

            for m in range(len(listmodels)):
                modelinlist = listmodels[m]
                for l in range(len(metric_columns)):
                    metric = metric_columns[l]
                    for j in range(len(smpResults)):
                        model = smpResults[j]['model']
                        metric_value = smpResults[j][metric]
                        if model == modelinlist:
                            metrics.append({'model': model, 'metric': metric, 'value': metric_value})

            for i in range(len(metric_columns)):
                current_metric = metric_columns[i]
                df_WARPATH[current_metric] = np.array(collect_Metrics(metrics, 'WarPATH', current_metric))
                std = np.square(std_deviation(df_WARPATH[current_metric]))
                var = variance(df_WARPATH[current_metric])
                std_Dev.append({'model': 'WarPATH', 'metric': current_metric, 'SD': std, 'VAR': var})

                df_IWPC[current_metric] = np.array(collect_Metrics(metrics, 'IWPC', current_metric))
                std = np.square(std_deviation(df_IWPC[current_metric]))
                var = variance(df_IWPC[current_metric])
                std_Dev.append({'model': 'IWPC', 'metric': current_metric, 'SD': std, 'VAR': var})

                df_GAGE[current_metric] = np.array(collect_Metrics(metrics, 'Gage', current_metric))
                std = np.square(std_deviation(df_GAGE[current_metric]))
                var = variance(df_GAGE[current_metric])
                std_Dev.append({'model': 'Gage', 'metric': current_metric, 'SD': std, 'VAR': var})

                df_FIXED[current_metric] = np.array(collect_Metrics(metrics, 'Fixed', current_metric))
                std = np.square(std_deviation(df_FIXED[current_metric]))
                var = variance(df_FIXED[current_metric])
                std_Dev.append({'model': 'Fixed', 'metric': current_metric, 'SD': std, 'VAR': var})

            df_WARPATH.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPATH_samples" + ".csv", ";")
            df_IWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_samples" + ".csv", ";")
            df_GAGE.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Gage_samples" + ".csv", ";")
            df_FIXED.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Fixed_samples" + ".csv", ";")

    if df == impNumber:
        for k in range(len(std_Dev)):
            model = std_Dev[k]['model']
            metric = std_Dev[k]['metric']
            Wfactor = std_Dev[k]['SD']
            Bfactor = std_Dev[k]['VAR']
            modelpos = find(std_Dev_Summ, 'model', model)
            std_Dev_Summ[modelpos][metric] += Wfactor

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
            mae_value = models[k]['MAE'] / impNumber
            mae_list = collect_Results(results, fieldname, 'MAE')
            mae_variance = variance(mae_list) * Bfactor
            stdpos = find(std_Dev_Summ, 'model', fieldname)

            mae_std_dev = std_Dev_Summ[stdpos]['MAE'] / impNumber
            mae_CI_minus = round(mae_value - 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
            mae_CI_plus = round(mae_value + 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
            pw20_value = models[k]['PW20'] / impNumber
            pw20_list = collect_Results(results, fieldname, 'PW20')
            pw20_variance = variance(pw20_list) * Bfactor
            pw20_std_dev = std_Dev_Summ[stdpos]['PW20'] / impNumber

            pw20_CI_minus = round(pw20_value - 1.96 * np.sqrt(pw20_std_dev + pw20_variance), 4)
            pw20_CI_plus = round(pw20_value + 1.96 * np.sqrt(pw20_std_dev + pw20_variance), 4)
            R2_value = models[k]['R2'] / impNumber
            R2_list = collect_Results(results, fieldname, 'R2')
            R2_variance = variance(R2_list) * Bfactor

            R2_std_dev = std_Dev_Summ[stdpos]['R2'] / impNumber
            R2_CI_minus = round(R2_value - 1.96 * np.sqrt(R2_std_dev + R2_variance), 4)
            R2_CI_plus = round(R2_value + 1.96 * np.sqrt(R2_std_dev + R2_variance), 4)
            MALAR_value = models[k]['MALAR'] / impNumber
            MALAR_list = collect_Results(results, fieldname, 'MALAR')
            MALAR_variance = variance(MALAR_list) * Bfactor

            MALAR_std_dev = std_Dev_Summ[stdpos]['MALAR'] / impNumber
            MALAR_CI_minus = round(MALAR_value - 1.96 * np.sqrt(MALAR_std_dev + MALAR_variance), 4)
            MALAR_CI_plus = round(MALAR_value + 1.96 * np.sqrt(MALAR_std_dev + MALAR_variance), 4)
            MLAR_value = models[k]['MLAR'] / impNumber
            MLAR_list = collect_Results(results, fieldname, 'MLAR')
            MLAR_variance = variance(MLAR_list) * Bfactor
            MLAR_std_dev = std_Dev_Summ[stdpos]['MLAR'] / impNumber

            MLAR_CI_minus = round(MLAR_value - 1.96 * np.sqrt(MLAR_std_dev + MLAR_variance), 4)
            MLAR_CI_plus = round(MLAR_value + 1.96 * np.sqrt(MLAR_std_dev + MLAR_variance), 4)
            if False:
                print(fieldname, 'MAE:', round(mae_value, 6), "StdDev:", round(mae_std_dev, 6), "B: ",
                  round(mae_variance, 4), "  CI: [", mae_CI_minus, mae_CI_plus, "]", file=fileoutput)
                print(fieldname, 'PW20:', round(pw20_value, 6), "StdDev:", round(pw20_std_dev, 6), "B: ",
                  round(pw20_variance, 4), " CI: [", pw20_CI_minus, pw20_CI_plus, "]", file=fileoutput)
                print(fieldname, 'R2:', round(R2_value, 6), "StdDev:", round(R2_std_dev, 6), "B: ", round(R2_variance, 4),
                  " CI: [", R2_CI_minus, R2_CI_plus, "]", file=fileoutput)
                print(fieldname, 'MALAR:', round(MALAR_value, 6), "StdDev:", round(MALAR_std_dev, 6), "B: ",
                  round(MALAR_variance, 4), " CI: [", MALAR_CI_minus, MALAR_CI_plus, "]", file=fileoutput)
                print(fieldname, 'MLAR:', round(MLAR_value, 6), "StdDev", round(MLAR_std_dev, 6), "B :",
                  round(MLAR_variance, 4), " CI:", MLAR_CI_minus, MLAR_CI_plus, "]", file=fileoutput)
            else:
                print(fieldname, 'MAE:', round(mae_value, 6), file=fileoutput)
                print(fieldname, 'PW20:', round(pw20_value, 6), file=fileoutput)
                print(fieldname, 'R2:', round(R2_value, 6), file=fileoutput)
                print(fieldname, 'MAPE/MALAR:', round(MALAR_value, 6), file=fileoutput)
                print(fieldname, 'Bias/MLAR:', round(MLAR_value, 6), file=fileoutput)



if __name__ == "__main__":
    main()