import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score
from warfit_learn import metrics
from warfit_learn.metrics import score_pw20, score_r2, score_mae
from warfit_learn.metrics import confidence_interval
from scipy.stats import norm
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose, dec
import warnings

warnings.filterwarnings("ignore")

def variance(metric):
    meanvalue = np.mean(metric)
    sumsquares = 0
    for i in range(len(metric)):
        core = (metric[i] - meanvalue)
        sumsquares += np.square(core)
    variance = sumsquares / (len(metric - 1))
    return variance

def std_deviation(metric):
    return np.sqrt(variance(metric))

def confintlimit95(metric):
    return 1.96 * np.sqrt(variance(metric) / len(metric))

def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['metric']) == metric:
            if (metrics[i]['model'] == model):
                container.append(metrics[i]['value'])
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


def AdultCheck(Age):
    return max(Age.all(), 18)


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
    # only applies to the War-PATH data!
    return round(predicted / 2.5, 0) * 2.5


def main():
    file_name = "IWPCImputations.csv"

    try:
        fileinput = open(file_name, 'r')
        file_content = fileinput.readlines()
        if file_content:
            print(file_name, ' does contain content ')
        else:
            print(file_name, ' has no content')
        dfnew = pd.read_csv(file_name, ";")
        implist = []
        implist.append(dfnew)
        if True:
            dfnew.rename(columns={'.imp': 'Imp', '.id': 'PatID'}, inplace=True)
            patients = 690
            print(dfnew.info())
            implist = []
            imp = 1
            while imp < 100:
                endrow = (imp + 1) * patients
                beginrow = imp * patients
                df = dfnew.iloc[beginrow:endrow, :]
                print("imputation", imp)
                ##print(df)
                implist.append({"DataFrame": df})
                imp = imp + 1

        results = []
        impResults = []
        models = []
        for df in range(len(implist)):
            dfcurrent = implist[df]["DataFrame"]
            # dfcurrent = implist[df]
            # print(dfcurrent)
            dfcurrent = dfcurrent.reset_index(drop=True)
            # print(dfcurrent)
            dfmod = dfcurrent
            dfmod = dfcurrent.drop(['Gender'], axis=1)
            dfgen = dfcurrent['Gender']
            ##print(dfmod)
            impResults = []
            metric_columns = ['MAE', 'PW20', 'R2', 'MALAR', 'MLAR']
            listmodels = ['WarPATH', 'IWPC', 'Gage', 'Fixed']
            dfmod["Dose_mg_week"] = dfmod["Dose_mg_week"].astype("float")
            dfmod["AgeLower"] = dfmod['Age_years'].str.split('_', expand=True)[0]
            dfmod['AgeLower'] = dfmod['AgeLower'].str.replace('+', '')
            dfmod["AgeDecades"] = dfmod["AgeLower"].astype("float") * 0.1
            dfmod["AgeYears"] = dfmod["AgeLower"].astype("float")
            dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
            dfmod.drop(['AgeLower'], axis=1, inplace=True)
            dfmod.drop(['Age_years'], axis=1, inplace=True)
            dfmod["Weight_kg"] = dfmod["Weight_kg"].astype("float")
            dfmod["Height_cm"] = dfmod["Height_cm"].astype("float")
            dfmod["Target_INR"] = dfmod["Target_INR"].astype("float")
            dfmod["INR_Three"] = np.where(dfmod["Target_INR"] >= 3.0, 1, 0)
            # dfmod["INR_Three"] = dfmod.apply(lambda x: INRThree(dfmod["Target_INR"]), axis=1)
            dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
            dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
            dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
            dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
            dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)
            factor_IWPC = {"Age": -0.2546, "Height": 0.0118, "Weight": 0.0134, "Inducer": 1.2799, "Amiodarone": -0.5695,
                           "Intercept": 4.4436}
            factor_Gage = {"Age": -0.0075, "BSA": 0.425, "TargetINR": 0.216, "Smoker": 0.108, "Amiodarone": -0.257,
                           "Indication": 0.0784, "Intercept": 0.769}
            factor_Fixed = {"Intercept": 35}
            factor_WarPath = {'Intercept': 23.6886, 'Age': -0.0656, 'Weight': 0.2178, 'TargetINR': 7.3190}
            if (find(models, 'model', 'IWPC') == -1):
                models.append({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0,
                               'MAE_sd' : 0, 'PW20_sd' :0, 'R2_sd' : 0, 'MALAR_sd' :0, 'MLAR_sd' :0})
                models.append({'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0,
                               'MAE_sd' : 0, 'PW20_sd' :0, 'R2_sd' : 0, 'MALAR_sd' :0, 'MLAR_sd' :0})
                models.append({'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0,
                               'MAE_sd' : 0, 'PW20_sd' :0, 'R2_sd' : 0, 'MALAR_sd' :0, 'MLAR_sd' :0})
                models.append({'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0,
                               'MAE_sd' : 0, 'PW20_sd' :0, 'R2_sd' : 0, 'MALAR_sd' :0, 'MLAR_sd' :0})
            dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
            dfmod['IWPC_Pred'] = np.square(
                factor_IWPC["Intercept"] + factor_IWPC["Age"] * dfmod["AgeDecades"] + factor_IWPC["Height"] * dfmod[
                    "Height_cm"] + factor_IWPC["Weight"] * dfmod["Weight_kg"] + factor_IWPC["Amiodarone"] * dfmod[
                    "Amiodarone"] + factor_IWPC["Inducer"] * dfmod["Inducer"])
            dfmod['IWPC_PredRound'] = dfmod.apply(lambda x: DoseRound(x['IWPC_Pred']), axis=1)
            dfKey = dfmod[['IWPC_PredRound', 'IWPC_Pred', 'Dose_mg_week']]
            dfKey.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_PredActual_" + str(df) + ".csv", ";")
            impResults.append(
                {'Imp': df, 'model': 'IWPC', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0, 'MAE_sd':0, 'PW20_sd':0, 'R2_sd':0, 'MALAR_sd':0, 'MLAR_sd':0})
            dfmod["Gage_Pred"] = 7 * np.exp(factor_Gage["Intercept"] + factor_Gage["Age"] * dfmod["AgeYears"] +
                                            factor_Gage["BSA"] * dfmod["BSA"] + factor_Gage["TargetINR"] * dfmod[
                                                "Target_INR"] +
                                            factor_Gage["Amiodarone"] * dfmod["Amiodarone"] +
                                            factor_Gage["Indication"] * dfmod["Indicationflag"] + factor_Gage[
                                                "Smoker"] * dfmod["Smoker"])
            # dfmod["Gage_PredRound"] = dfmod.apply(lambda x: DoseRound(x["Gage_Pred"]), axis=1)
            dfKey = dfmod[["Gage_Pred", "Dose_mg_week"]]
            impResults.append(
                {'Imp': df, 'model': 'Gage', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0, 'MAE_sd':0, 'PW20_sd':0, 'R2_sd':0, 'MALAR_sd':0, 'MLAR_sd':0})
            dfmod["WarPATH_Pred"] = factor_WarPath['Intercept'] + factor_WarPath['Age'] * dfmod['AgeYears'] + \
                                    factor_WarPath['Weight'] * dfmod['Weight_kg'] + factor_WarPath['TargetINR'] * dfmod[
                                        'INR_Three']
            # dfmod["WarPATH_PredRound"] = dfmod.apply(lambda x: DoseRound(x["WarPATH_Pred"]), axis=1)
            dfKey = dfmod[["WarPATH_Pred", "Dose_mg_week"]]
            impResults.append(
                {'Imp': df, 'model': 'WarPATH', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0, 'MAE_sd':0, 'PW20_sd':0, 'R2_sd':0, 'MALAR_sd':0, 'MLAR_sd':0})
            dfmod["Fixed_Pred"] = factor_Fixed["Intercept"]
            # dfmod["Fixed_PredRound"] = dfmod.apply(lambda x: DoseRound(x["Fixed_Pred"]), axis=1)
            dfKey = dfmod[["Fixed_Pred", "Dose_mg_week"]]
            impResults.append(
                {'Imp': df, 'model': 'Fixed', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0, 'MAE_sd':0, 'PW20_sd':0, 'R2_sd':0, 'MALAR_sd':0, 'MLAR_sd':0})
            dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imputed_" + str(df) + ".csv", ";")
            boot = 0
            samples = []
            metrics = []
            smpResults = []
            data = np.random.randint(10, size=(5, 5))
            df_WARPATH = pd.DataFrame(data=data, columns=metric_columns)
            df_IWPC = pd.DataFrame(data=data, columns=metric_columns)
            df_GAGE = pd.DataFrame(data=data, columns=metric_columns)
            df_FIXED = pd.DataFrame(data=data, columns=metric_columns)
            if False:
                while boot < 5:
                    print("imputation ", df, " on sample ", boot)
                    dfsample = dfmod.sample(n=690,frac=None,replace = True)
                    dfsample = dfsample.reset_index(drop=True)
                    dfMetricfactors = dfsample[["WarPATH_Pred","IWPC_Pred","Fixed_Pred","Gage_Pred","Dose_mg_week"]]
                    dfsample.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imp" + str(df) + "Samp" + str(boot) + ".csv", ";")
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
                        smpResults.append({'Imp': df, 'Sample': boot, 'model': model,'MAE': curr_MAE,'PW20': curr_PW20,'R2': curr_R2,'MALAR': curr_MALAR,'MLAR': curr_MLAR})

                for m in range(len(listmodels)):
                    modelinlist = listmodels[m]
                    for l in range(len(metric_columns)):
                       metric = metric_columns[l]
                       for j in range(len(smpResults)):
                         model = smpResults[j]['model']
                         metric_value = smpResults[j][metric]
                         if model == modelinlist:
                           metrics.append({'model': model, 'metric': metric, 'value': metric_value})

            #for m in range(len(smpResults)):
             #   print("imputation", smpResults['Imp'], smpResults["Sample"], smpResults["model"], smpResults["MAE"],
                for i in range(len(metric_columns)):
                    current_metric = metric_columns[i]
                    df_WARPATH[current_metric] = np.array(collect_Metrics(metrics, 'WarPATH', current_metric))
                    #print("Confidence Interval Factor",'WARPATH', current_metric, confintlimit95(df_WARPATH[current_metric]))
                    #print(df_WARPATH[current_metric])
                    df_IWPC[current_metric] = np.array(collect_Metrics(metrics, 'IWPC', current_metric))
                    #print("Confidence Interval Factor",'IWPC', current_metric, confintlimit95(df_IWPC[current_metric]))
                    #print(df_IWPC[current_metric])
                    df_GAGE[current_metric] = np.array(collect_Metrics(metrics, 'Gage', current_metric))
                    #print("Confidence Interval Factor",'GAGE', current_metric, confintlimit95(df_GAGE[current_metric]))
                    #print(df_GAGE[current_metric])
                    df_FIXED[current_metric] = np.array(collect_Metrics(metrics, 'Fixed', current_metric))
                    #print("Confidence Interval Factor", 'FIXED', current_metric, confintlimit95(df_FIXED[current_metric]))
                    #print(df_FIXED[current_metric])

                df_WARPATH.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPATH_samples" + ".csv", ";")
                df_IWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_samples" + ".csv", ";")
                df_GAGE.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Gage_samples" + ".csv", ";")
                df_FIXED.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Fixed_samples" + ".csv", ";")
                stdDev = []
                for i in range(len(metric_columns)):
                   current_metric = metric_columns[i]
                   df_WARPATH['Mean' + current_metric] = np.mean(df_WARPATH[current_metric])
                   meanvalue = np.mean(df_WARPATH[current_metric])
                   sd = confintlimit95(df_WARPATH[current_metric])
                   stdDev.append({'model':'WarPATH', 'metric':current_metric, 'SD':sd})
                   #print('WARPATH ', current_metric, meanvalue, sd)
                   for i in range(len(metric_columns)):
                      current_metric = metric_columns[i]
                      meanvalue = np.mean(df_IWPC[current_metric])
                      sd = confintlimit95(df_IWPC[current_metric])
                      stdDev.append({'model': 'IWPC', 'metric': current_metric, 'SD': sd})
                      #print('IWPC ', current_metric, meanvalue, sd)
                   for i in range(len(metric_columns)):
                        current_metric = metric_columns[i]
                        meanvalue = np.mean(df_GAGE[current_metric])
                        sd =confintlimit95(df_GAGE[current_metric])
                        stdDev.append({'model': 'Gage', 'metric': current_metric, 'SD': sd})
                        #print('GAGE ', current_metric, meanvalue, sd)
                   for i in range(len(metric_columns)):
                        current_metric = metric_columns[i]
                        meanvalue = np.mean(df_FIXED[current_metric])
                        sd = confintlimit95(df_FIXED[current_metric])
                        stdDev.append({'model': 'Fixed', 'metric': current_metric, 'SD': sd})
                        #print('Fixed Dose ', current_metric, meanvalue, sd)

                df_WARPATH.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPATH_samples" + ".csv", ";")
                df_IWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_samples" + ".csv", ";")
                df_GAGE.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Gage_samples" + ".csv", ";")
                df_FIXED.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Fixed_samples" + ".csv", ";")

                for k in range(len(impResults)):
                 dfKey = impResults[k]['predactual']
                 model = impResults[k]['model']
                 imputation = impResults[k]['Imp']
             # print(dfKey, model, imputation)
                 doseTrue = dfKey["Dose_mg_week"]
            # dosePred = dfKey[model + '_PredRound']
                 dosePred2 = dfKey[model + '_Pred']
            # print (k, doseTrue, dosePred)
                 impResults[k]['MAE'] = score_mae(doseTrue, dosePred2)
                 impResults[k]['PW20'] = PercIn20(doseTrue, dosePred2)
                 impResults[k]['R2'] = RSquared(doseTrue, dosePred2)
                 impResults[k]['MAPE'] = MAPE(doseTrue, dosePred2)
                 impResults[k]['MLAR'] = MLAR(doseTrue, dosePred2)
                 impResults[k]['MALAR'] = MALAR(doseTrue, dosePred2)
                 if False:
                    for s in range(len(stdDev)):
                        if stdDev[s]['model'] == model:
                         for m in range(len(metric_columns)):
                            metric = metric_columns[m]
                            if stdDev[s]['metric'] == metric:
                                impResults[k][metric+'_sd'] = stdDev[s]['SD']


            for k in range(len(impResults)):
                 a = impResults[k]['model']
                 b = impResults[k]['Imp']
                 c = impResults[k]['MAE']
                 d = impResults[k]['PW20']
                 e = impResults[k]['R2']
                 f = impResults[k]['MAPE']
                 g = impResults[k]['MLAR']
                 h = impResults[k]['MALAR']
                 i = impResults[k]['MAE_sd']
                 j = impResults[k]['PW20_sd']
                 l = impResults[k]['R2_sd']
                 m = impResults[k]['MALAR_sd']
                 n = impResults[k]['MLAR_sd']
                 results.append({'Imp': b, 'model': a, 'MAE': c, 'PW20': d, 'R2': e, 'MAPE': f, 'MLAR': g, 'MALAR': h,
                                'MAE_sd': i, 'PW20_sd': j, 'R2_sd':l, 'MALAR_sd':m, 'MLAR_sd':n})

            for k in range(len(models)):
                 fieldname = models[k]['model']
                 for m in range(len(results)):
                   if results[m]['model'] == fieldname:

                    models[k]['MAE'] += results[m]['MAE']
                    models[k]['PW20'] += results[m]['PW20']
                    models[k]['R2'] += results[m]['R2']
                    models[k]['MAPE'] += results[m]['MAPE']
                    models[k]['MLAR'] += results[m]['MLAR']
                    models[k]['MALAR'] += results[m]['MALAR']
                    models[k]['MAE_sd'] += results[m]['MAE_sd']
                    models[k]['PW20_sd'] += results[m]['PW20_sd']
                    models[k]['R2_sd'] += results[m]['R2_sd']
                    models[k]['MALAR_sd'] += results[m]['MALAR_sd']
                    models[k]['MLAR_sd'] += results[m]['MLAR_sd']
                    #print(fieldname, results[m]['MAE'], results[m]['PW20'], results[m]['R2'], results[m]['MAPE'],
                    #     results[m]['MLAR'], results[m]['MALAR'])

            # print('***', m, fieldname, results[m]['MAE'], results[m]['PW20'], results[m]['R2'], results[m]['MAPE'],results[m]['MLAR'],results[m]['MALAR'])

            for k in range(len(models)):
                  fieldname = models[k]['model']
                  mae_value = round(models[k]['MAE'] / 100, 4)
                  print(fieldname, 'MAE',models[k]['MAE'])
                  pw20_value = round(models[k]['PW20'] / 100, 4)
                  print(fieldname,'PW20',models[k]['PW20'])
                  R2_value = round(models[k]['R2'] / 100, 4)
                  print(fieldname, 'R2',models[k]['R2'])
                  MALAR_value = round(models[k]['MALAR'] / 100, 4)
                  print(fieldname, 'MALAR',models[k]['MALAR'])
                  MLAR_value = round(models[k]['MLAR'] / 100, 4)
                  print(fieldname, 'MLAR',models[k]['MLAR'])
                  MALAR_value = round(models[k]['MALAR'] / 100, 4)
                  print(fieldname, models[k]['MAE'], models[k]['PW20'], models[k]['R2'],  models[k]['MLAR'], models[k]['MALAR'])
                  print(fieldname, 'MAE:', mae_value, 'PW20:', pw20_value, 'R2:', R2_value,  'MLAR:', MLAR_value, 'MALAR:', MALAR_value)


        if False:
                print("Confidence Intervals for ", fieldname)
                factor = models[k]['MAE_sd']/100
                print("MAE 1.96 * std deviation/(sqrt(n) ", factor )
                MAE_avg = models[k]['MAE']/100
                MAE_upper = MAE_avg+factor
                MAE_lower = MAE_avg-factor
                print(fieldname, 'MAE', MAE_avg, MAE_lower, MAE_upper)
                factor = models[k]['PW20_sd']/100
                print("PW20 1.96 * std deviation/(sqrt(n) ", factor)
                PW20_avg = models[k]['PW20']/100
                PW20_upper = round(PW20_avg+factor,4)
                PW20_lower = round(PW20_avg-factor,4)
                print(fieldname, 'PW20',PW20_avg, PW20_lower,PW20_upper)
                factor = models[k]['R2_sd']/100
                print("R2 1.96 * std deviation/(sqrt(n) ", factor)
                R2_avg = models[k]['R2']/100
                R2_upper = round(R2_avg + factor,4)
                R2_lower = round(R2_avg - factor,4)
                print(fieldname, 'R2', R2_avg,R2_lower, R2_upper)
                factor = models[k]['MALAR_sd']/100
                print("MALAR 1.96 * std deviation/(sqrt(n) ", factor)
                MALAR_avg = models[k]['MALAR']/100
                MALAR_upper = round(MALAR_avg + factor,4)
                MALAR_lower = round(MALAR_avg - factor,4)
                print(fieldname, 'MALAR', MALAR_avg, MALAR_lower, MALAR_upper)
                factor = models[k]['MLAR_sd']/100
                print("MLAR 1.96 * std deviation/(sqrt(n) ", factor)
                MLAR_avg = models[k]['MLAR']/100
                MLAR_upper = round(MLAR_avg + factor,4)
                MLAR_lower = round(MLAR_avg - factor,4)
                print(fieldname, 'MLAR', MLAR_avg,MLAR_lower, MLAR_upper)

    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)

if __name__ == "__main__":
    main()