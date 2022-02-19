import csv
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score
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

def SList(series):
    return np.array(series.values.tolist())

def main():
    filewritename = input("Enter file name: \n")
    fileoutput = open(filewritename, 'w')
    impNumber = 100
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    filelist = [f for f in os.listdir(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations") if f.endswith(".csv")]
    for f in filelist:
        os.remove(os.path.join(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations", f))

    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    for imp in range(impNumber):
        counter = imp+1
        dfcurrent = df.loc[df[".imp"] == counter]
        dfcurrent.to_csv()
        suffix = str(counter).zfill(3)
        dfcurrent.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_" + suffix + ".csv", ";")

    factor_IWPC = {"Age": -0.2546, "Height": 0.0118, "Weight": 0.0134, "Inducer": 1.2799, "Amiodarone": -0.5695,
                   "Intercept": 4.4436}
    factor_Gage = {"Age": -0.0075, "BSA": 0.425, "TargetINR": 0.216, "Smoker": 0.108, "Amiodarone": -0.257,
                   "Indication": 0.0784, "Intercept": 0.769}
    factor_Fixed = {"Intercept": 35}
    factor_WarPath = {'Intercept': 20.2832, 'Age': -0.0656, 'Weight': 0.2178, 'TargetINR': 7.3190,
                      'HIVPositive': 8.7973, 'HIVUnknown': 3.4054}

    number_of_samples = 100
    number_of_imps = 100
    filesImp = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv'):
                filesImp.append(file)

    impNumber = len(filesImp)
    results = []
    std_Dev = []
    std_Dev_Summ = ({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                    {'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                    {'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                    {'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0})

    for file in filesImp:
        dfnew = pd.read_csv(root + '\\' + file, ";")
        df = filesImp.index(file)+1
        dfmod = dfnew
        impResults = []
        models = []
        boot = 0
        samples = []
        metrics = []
        smpResults = []
        metric_columns = ['MAE', 'PW20', 'R2', 'MALAR', 'MLAR']
        listmodels = ['WarPATH', 'IWPC', 'Gage', 'Fixed']

        dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
        dfmod["INR_Three"] = np.where(dfmod["Target_INR"] == "Three", 1, 0)
        dfmod["Target_INR"] = np.where(dfmod["Target_INR"] == "Three", 3.0, np.where(dfmod["Target_INR"] == "aTwo_point_five", 2.5, 2.0))
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
        dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
        if (find(models, 'model', 'IWPC') == -1):
            models.append({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
            models.append({'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
        IWPCbase = factor_IWPC["Intercept"] + factor_IWPC["Age"] * dfmod["AgeDecades"] + factor_IWPC["Height"] * dfmod[
            "Height_cm"] + \
            factor_IWPC["Weight"] * dfmod["Weight_kg"] + factor_IWPC["Amiodarone"] * dfmod["Amiodarone"] + \
            factor_IWPC["Inducer"] * dfmod["Inducer"]
        dfmod['IWPC_Pred'] = np.square(IWPCbase)
        dfmod['IWPC_Pred_Round'] = DoseRound(dfmod['IWPC_Pred'])
        print("on imputation ", df)
        dfKey = dfmod[['IWPC_Pred', 'IWPC_Pred_Round', 'Dose_mg_week']]
        impResults.append({'Imp': df, 'model': 'IWPC', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0,
                           'MALAR': 0})
        dfmod["Gage_Pred"] = 7 * np.exp(factor_Gage["Intercept"] + factor_Gage["Age"] * dfmod["AgeYears"] +
                            factor_Gage["BSA"] * dfmod["BSA"] + factor_Gage["TargetINR"] * dfmod["Target_INR"] +
                            factor_Gage["Amiodarone"] * dfmod["Amiodarone"] +
                            factor_Gage["Indication"] * dfmod["Indicationflag"] + factor_Gage["Smoker"] * dfmod["Smoker"])
        dfmod['Gage_Pred_Round'] = DoseRound(dfmod['Gage_Pred'])
        dfKey = dfmod[["Gage_Pred", "Gage_Pred_Round", "Dose_mg_week"]]
        impResults.append(
            {'Imp': df, 'model': 'Gage', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
        dfmod["WarPATH_Pred"] = factor_WarPath['Intercept'] + factor_WarPath['Age'] * dfmod['AgeYears'] + \
                                factor_WarPath['Weight'] * dfmod['Weight_kg'] + \
                                factor_WarPath['TargetINR'] * dfmod['INR_Three'] + \
                                factor_WarPath['HIVPositive'] * dfmod['HIVPositive'] + factor_WarPath['HIVUnknown'] * \
                                dfmod['HIVUnknown']

        dfmod["WarPATH_Pred_Round"] = DoseRound(dfmod['WarPATH_Pred'])
        dfKey = dfmod[["WarPATH_Pred", "WarPATH_Pred_Round", "Dose_mg_week"]]
        impResults.append(
            {'Imp': df, 'model': 'WarPATH', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
        dfmod["Fixed_Pred"] = factor_Fixed["Intercept"]
        dfmod["Fixed_Pred_Round"] = DoseRound(dfmod["Fixed_Pred"])
        dfKey = dfmod[["Fixed_Pred", "Fixed_Pred_Round", "Dose_mg_week"]]
        impResults.append(
            {'Imp': df, 'model': 'Fixed', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MLAR': 0, 'MALAR': 0})
        suffix = str(df).zfill(3)
        dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARMICEPRED\WARRImputed_" + suffix + ".csv", ";")

        for k in range(len(impResults)):
            dfKey = impResults[k]['predactual']
            model = impResults[k]['model']
            imputation = impResults[k]['Imp']
            doseTrue = dfKey["Dose_mg_week"]
            dosePred2 = dfKey[model + '_Pred']
            dosePred = dfKey[model + '_Pred_Round']
            impResults[k]['MAE'] = score_mae(doseTrue, dosePred)
            impResults[k]['PW20'] = PercIn20(doseTrue, dosePred)
            impResults[k]['R2'] = RSquared(doseTrue, dosePred2)
            impResults[k]['MLAR'] = MLAR(doseTrue, dosePred)
            impResults[k]['MALAR'] = MALAR(doseTrue, dosePred)

        for k in range(len(impResults)):
            a = impResults[k]['model']
            b = impResults[k]['Imp']
            c = impResults[k]['MAE']
            d = impResults[k]['PW20']
            e = impResults[k]['R2']
            g = impResults[k]['MLAR']
            h = impResults[k]['MALAR']
            results.append({'Imp': b, 'model': a, 'MAE': c, 'PW20': d, 'R2': e, 'MLAR': g, 'MALAR': h})

        resultsdf = pd.DataFrame(results)
        suffix = str(df).zfill(3)
        resultsdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARMICEResults\WAR_Results_" + suffix + ".csv", ";")
        boot = 0
        samples = []
        metrics = []
        smpResults = []

        data = np.random.randint(10, size=(number_of_samples, 5))
        df_WARPATH = pd.DataFrame(data=data, columns=metric_columns)
        df_IWPC = pd.DataFrame(data=data, columns=metric_columns)
        df_GAGE = pd.DataFrame(data=data, columns=metric_columns)
        df_FIXED = pd.DataFrame(data=data, columns=metric_columns)
        if True:
            while boot < number_of_samples:
                print("imputation ", df, " on sample ", boot)
                dfsample = dfmod.sample(n=364, frac=None, replace=True)
                dfsample = dfsample.reset_index(drop=True)
                dfMetricfactors = dfsample[
                    ["WarPATH_Pred", "WarPATH_Pred_Round", "IWPC_Pred", "IWPC_Pred_Round", "Fixed_Pred",
                     "Fixed_Pred_Round", "Gage_Pred", "Gage_Pred_Round", "Dose_mg_week"]]
                suffix = str(df).zfill(3)
                dfsample.to_csv(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WAR_Imp" + suffix + "Samp" + str(boot) + ".csv", ";")
                boot = boot + 1
                samples.append(dfMetricfactors)
                for m in range(len(listmodels)):
                    model = listmodels[m]
                    doseTrue = dfMetricfactors["Dose_mg_week"]
                    dosePred2 = dfMetricfactors[model + '_Pred']
                    dosePred = dfMetricfactors[model + '_Pred_Round']
                    curr_MAE = score_mae(doseTrue, dosePred)
                    curr_PW20 = PercIn20(doseTrue, dosePred)
                    curr_R2 = RSquared(doseTrue, dosePred2)
                    curr_MALAR = MALAR(doseTrue, dosePred)
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
                stdpre = std_deviation(df_WARPATH[current_metric])
                std_Dev.append({'model': 'WarPATH', 'metric': current_metric, 'SD': std, 'STDEV': stdpre})

                df_IWPC[current_metric] = np.array(collect_Metrics(metrics, 'IWPC', current_metric))
                std = np.square(std_deviation(df_IWPC[current_metric]))
                stdpre = std_deviation(df_IWPC[current_metric])
                std_Dev.append({'model': 'IWPC', 'metric': current_metric, 'SD': std, 'STDEV': stdpre})

                df_GAGE[current_metric] = np.array(collect_Metrics(metrics, 'Gage', current_metric))
                std = np.square(std_deviation(df_GAGE[current_metric]))
                stdpre = std_deviation(df_GAGE[current_metric])
                std_Dev.append({'model': 'Gage', 'metric': current_metric, 'SD': std, 'STDEV': stdpre})

                df_FIXED[current_metric] = np.array(collect_Metrics(metrics, 'Fixed', current_metric))
                std = np.square(std_deviation(df_FIXED[current_metric]))
                stdpre = std_deviation(df_FIXED[current_metric])
                std_Dev.append({'model': 'Fixed', 'metric': current_metric, 'SD': std, 'STDEV': stdpre})

            df_WARPATH.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPATH_samples" + ".csv", ";")
            df_IWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_samples" + ".csv", ";")
            df_GAGE.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Gage_samples" + ".csv", ";")
            df_FIXED.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Fixed_samples" + ".csv", ";")

        if df == impNumber:

            std_Dev_df = pd.DataFrame(std_Dev)
            std_Dev_df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\StdDeviationSquares" + ".csv", ";")
            for k in range(len(std_Dev)):
                model = std_Dev[k]['model']
                metric = std_Dev[k]['metric']
                Wfactor = std_Dev[k]['SD']
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


if __name__ == "__main__":
    main()