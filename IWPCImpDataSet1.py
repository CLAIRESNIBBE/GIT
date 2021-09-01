import csv
import time
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
    variance = sumsquares / ((len(metric) - 1))
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
        topsum += abs((predval[i] - trueval[i])/trueval[i])
    return topsum/len(trueval)*100

def MLAR(trueval, predval):
    # mean log of accuracy ratio
    sum = 0
    for i in range(len(trueval)):
        sum += np.log(predval[i]/trueval[i])
    return (np.exp(sum/len(trueval))-1)*100

def MALAR(trueval, predval):
    # mean absolute log of absolute ratio
    sum = 0
    for i in range(len(trueval)):
        sum += abs(np.log(predval[i]/trueval[i]))
    return (np.exp(sum/len(trueval))-1)*100

def RSquared(trueval,predval):
    true_mean = np.mean(trueval)
    topsum = 0
    lowersum  = 0
    for i in range(len(trueval)):
       topsum += np.square((predval[i] - true_mean))
       lowersum += np.square(trueval[i] - true_mean)
    return topsum/lowersum*100


def BSA(height, weight):
    return 0.007184*height**0.725*weight**0.425


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

    return max(Age.all(),18)

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
    return round(predicted/2.5, 0) * 2.5

def main():
    file_name = "DataSet1_CI.csv"


    try:
        fileinput = open(file_name, 'r')
        file_content = fileinput.readlines()
        if file_content:
            print(file_name, ' does contain content ')
        else:
            print(file_name, ' has no content')
        dfnew = pd.read_csv(file_name, ";")
        print(dfnew)
        implist=[]
        implist.append(dfnew)
        if False:
            dfnew.rename(columns={'.imp': 'Imp', '.id': 'PatID'},inplace=True)
            patients = 690
            print(dfnew.info())
            implist = []
            imp = 1
            while imp<=100:
                endrow = (imp+1)*patients
                beginrow = imp*patients
                df = dfnew.iloc[beginrow:endrow,:]
                print("imputation", imp)
                #print(df)
                implist.append({"DataFrame":df})
                imp = imp + 1

        results = []
        impResults = []
        models = []
        std_Dev = []
        for df in range(len(implist)):
            #dfcurrent = implist[df]["DataFrame"]
            dfcurrent = implist[df]
            print(dfcurrent)
            #dfcurrent = dfcurrent.reset_index(drop = True)
            #print(dfcurrent)
            dfmod = dfcurrent
            #dfmod = dfcurrent.drop(['Gender'], axis=1)
            #dfgen = dfcurrent['Gender']
            print(dfmod)
            impResults = []
            boot = 0
            samples = []
            metrics = []
            smpResults = []
            print(dfmod.info())
            #time.sleep(20)
            metric_columns = ['MAE', 'PW20', 'R2', 'MALAR', 'MLAR']
            listmodels = ['WarPATH', 'IWPC', 'Gage', 'Fixed']

            #dfmod["Dose_mg_week"] = dfmod["Dose_mg_week"].str.replace(",",".")
            #dfmod["Dose_mg_week"] = dfmod["Dose_mg_week"].astype("float")
            #dfmod["AgeLower"] = dfmod['Age_years'].str.split('_', expand=True)[0]
            #dfmod['AgeLower'] = dfmod['AgeLower'].str.replace('+', '')
            dfmod["AgeDecades"] = dfmod["Age_years"] * 0.1
            #dfmod["AgeYears"] = dfmod["AgeLower"].astype("float")
            dfmod['AgeYears'] = np.where((dfmod['Age_years'] <= 18), 18, dfmod['Age_years'])
            ##dfmod.drop(['AgeLower'], axis=1, inplace=True)
            dfmod.drop(['Age_years'], axis=1, inplace=True)
            dfmod["Weight_kg"] = dfmod["Weight_kg"].astype("float")
            dfmod["Height_cm"] = dfmod["Height_cm"].astype("float")
            dfmod["Target_INR"] = dfmod["Target_INR"].astype("float")
            dfmod["INR_Three"] = np.where(dfmod["Target_INR"] >= 3.0, 1, 0)
            #dfmod["INR_Three"] = dfmod.apply(lambda x: INRThree(dfmod["Target_INR"]), axis=1)
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
            if (find(models,'model','IWPC')== -1):
                models.append({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
                models.append({'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
                models.append({'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
                models.append({'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
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
                 'MALAR': 0})
            dfmod["Target_INR"] =  np.array(dfmod["Target_INR"], dtype=int)
            dfmod["Gage_Pred"] = 7 * np.exp(factor_Gage["Intercept"] + factor_Gage["Age"] * dfmod["AgeYears"] +
                factor_Gage["BSA"] * dfmod["BSA"] + factor_Gage["TargetINR"] * dfmod["Target_INR"] +
                factor_Gage["Amiodarone"] * dfmod["Amiodarone"] +
                factor_Gage["Indication"] * dfmod["Indicationflag"] + factor_Gage["Smoker"] * dfmod["Smoker"])
            #dfmod["Gage_PredRound"] = dfmod.apply(lambda x: DoseRound(x["Gage_Pred"]), axis=1)
            dfKey = dfmod[["Gage_Pred", "Dose_mg_week"]]
            impResults.append(
                {'Imp': df, 'model': 'Gage', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0})
            dfmod["WarPATH_Pred"] =  factor_WarPath['Intercept'] + factor_WarPath['Age'] * dfmod['AgeYears'] + factor_WarPath['Weight'] *  dfmod['Weight_kg'] + factor_WarPath['TargetINR'] * dfmod['INR_Three']
            #dfmod["WarPATH_PredRound"] = dfmod.apply(lambda x: DoseRound(x["WarPATH_Pred"]), axis=1)
            dfKey = dfmod[[ "WarPATH_Pred", "Dose_mg_week"]]
            impResults.append(
                {'Imp': df, 'model': 'WarPATH', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0})
            dfmod["Fixed_Pred"] = factor_Fixed["Intercept"]
            #dfmod["Fixed_PredRound"] = dfmod.apply(lambda x: DoseRound(x["Fixed_Pred"]), axis=1)
            dfKey = dfmod[[ "Fixed_Pred", "Dose_mg_week"]]
            impResults.append(
                {'Imp': df, 'model': 'Fixed', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0})
            dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imputed_" + str(df) + ".csv", ";")
            for k in range(len(impResults)):
                dfKey = impResults[k]['predactual']
                model = impResults[k]['model']
                imputation = impResults[k]['Imp']
                #print(dfKey, model, imputation)
                doseTrue = dfKey["Dose_mg_week"]
                #dosePred = dfKey[model + '_PredRound']
                dosePred2 = dfKey[model + '_Pred']
                #print (k, doseTrue, dosePred)
                impResults[k]['MAE'] = score_mae(doseTrue, dosePred2)
                impResults[k]['PW20'] = PercIn20(doseTrue, dosePred2)
                impResults[k]['R2'] = RSquared(doseTrue, dosePred2)
                impResults[k]['MAPE'] = MAPE(doseTrue, dosePred2)
                impResults[k]['MLAR'] = MLAR(doseTrue, dosePred2)
                impResults[k]['MALAR'] = MALAR(doseTrue, dosePred2)
                #print('**', dfKey, model, imputation, results[k]['MAE'], results[k]['PW20'], results[k]['R2'],
                #      results[k]['MAPE'], results[k]['MLAR'], results[k]['MALAR'])


            for k in range(len(impResults)):
                a=impResults[k]['model']
                b=impResults[k]['Imp']
                c=impResults[k]['MAE']
                d=impResults[k]['PW20']
                e=impResults[k]['R2']
                f=impResults[k]['MAPE']
                g=impResults[k]['MLAR']
                h=impResults[k]['MALAR']

                results.append({'Imp':b,'model':a,'MAE' : c, 'PW20' : d, 'R2': e, 'MAPE':f, 'MLAR':g, 'MALAR':h})

            boot = 0
            samples = []
            metrics = []
            smpResults = []
            std_Dev_Summ = []

            data = np.random.randint(10, size=(1000, 5))
            df_WARPATH = pd.DataFrame(data=data, columns=metric_columns)
            df_IWPC = pd.DataFrame(data=data, columns=metric_columns)
            df_GAGE = pd.DataFrame(data=data, columns=metric_columns)
            df_FIXED = pd.DataFrame(data=data, columns=metric_columns)
            if True:
              while boot < 1000:
                 print("imputation ", df, " on sample ", boot)
                 dfsample = dfmod.sample(n=690, frac=None, replace=True)
                 dfsample = dfsample.reset_index(drop=True)
                 dfMetricfactors = dfsample[["WarPATH_Pred", "IWPC_Pred", "Fixed_Pred", "Gage_Pred", "Dose_mg_week"]]
                 dfsample.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imp" + str(df) + "Samp" + str(boot) + ".csv",";")
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
                df_WARPATH[current_metric].to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPATH_metrics" + ".csv", ";")
                std = std_deviation(df_WARPATH[current_metric])
                var = variance(df_WARPATH[current_metric])

                std_Dev.append({'model': 'WarPATH', 'metric': current_metric, 'SD': std, 'VAR': var})
                df_IWPC[current_metric] = np.array(collect_Metrics(metrics, 'IWPC', current_metric))
                std = std_deviation(df_IWPC[current_metric])
                var = variance(df_IWPC[current_metric])

                std_Dev.append({'model': 'IWPC', 'metric': current_metric, 'SD': std,'VAR': var })
                df_GAGE[current_metric] = np.array(collect_Metrics(metrics, 'Gage', current_metric))
                std = std_deviation(df_GAGE[current_metric])
                var = variance(df_GAGE[current_metric])

                std_Dev.append({'model': 'Gage', 'metric': current_metric, 'SD': std, 'VAR': var})
                df_FIXED[current_metric] = np.array(collect_Metrics(metrics, 'Fixed', current_metric))
                std = std_deviation(df_FIXED[current_metric])
                var = variance(df_FIXED[current_metric])

                std_Dev.append({'model': 'Fixed', 'metric': current_metric, 'SD': std,'VAR': var})

              df_WARPATH.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPATH_samples" + ".csv", ";")
              df_IWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_samples" + ".csv", ";")
              df_GAGE.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Gage_samples" + ".csv", ";")
              df_FIXED.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Fixed_samples" + ".csv", ";")

        std_Dev_Summ=( {'model':'IWPC','MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                       {'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                       {'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                       {'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0} )

        variance_Summ = ({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                        {'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                        {'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                        {'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0})




        for k in range(len(std_Dev)):
            model = std_Dev[k]['model']
            metric = std_Dev[k]['metric']
            Wfactor = std_Dev[k]['SD']
            Bfactor = std_Dev[k]['VAR']
            modelpos = find(std_Dev_Summ,'model',model)
            std_Dev_Summ[modelpos][metric] += Wfactor**2
            modelpos2 = find(variance_Summ,'model', model)
            variance_Summ[modelpos2][metric] += Bfactor

        for k in range(len(models)):
            fieldname = models[k]['model']
            for m in range(len(results)):
                if results[m]['model'] == fieldname:
                    models[k]['MAE'] += results[m]['MAE']
                    models[k]['PW20'] += results[m]['PW20']
                    models[k]['R2'] += results[m]['R2']
                    models[k]['MLAR'] += results[m]['MLAR']
                    models[k]['MALAR'] += results[m]['MALAR']

        for k in range(len(models)):
                fieldname = models[k]['model']
                mae_value = models[k]['MAE'] / 100
                stdpos = find(std_Dev_Summ, 'model', fieldname)
                varpos = find(variance_Summ, 'model', fieldname)
                mae_var = variance_Summ[varpos]['MAE']*101/100
                mae_std_dev = std_Dev_Summ[stdpos]['MAE']/100
                mae_CI_minus = round(mae_value - np.sqrt(mae_std_dev + mae_var),4)
                mae_CI_plus = round(mae_value + np.sqrt(mae_std_dev + mae_var),4)
                pw20_value = models[k]['PW20'] / 100
                pw20_std_dev = std_Dev_Summ[stdpos]['PW20']/100
                pw20_var = variance_Summ[varpos]['MAE']*101/100
                pw20_CI_minus = round(pw20_value-np.sqrt(pw20_std_dev + pw20_var),4)
                pw20_CI_plus = round(pw20_value +np.sqrt(pw20_std_dev + pw20_var),4)
                R2_value = models[k]['R2'] / 100
                R2_var = variance_Summ[varpos]['R2']*101/100
                R2_std_dev = std_Dev_Summ[stdpos]['R2']/100
                R2_CI_minus = round(R2_value - np.sqrt(R2_std_dev + R2_var),4)
                R2_CI_plus = round(R2_value + np.sqrt(R2_std_dev + R2_var),4)
                MALAR_value = models[k]['MALAR'] / 100
                MALAR_var = variance_Summ[varpos]['MALAR']*101/100
                MALAR_std_dev = std_Dev_Summ[stdpos]['MALAR']/100
                MALAR_CI_minus = round(MALAR_value -np.sqrt (MALAR_std_dev+MALAR_var),4)
                MALAR_CI_plus = round(MALAR_value + np.sqrt(MALAR_std_dev+MALAR_var),4)
                MLAR_value = round(models[k]['MLAR'] / 100, 4)
                MLAR_std_dev = std_Dev_Summ[stdpos]['MLAR']/100
                MLAR_var = variance_Summ[varpos]['MLAR']*101/100
                MLAR_CI_minus = round(MLAR_value - np.sqrt(MLAR_std_dev+MLAR_var),4)
                MLAR_CI_plus = round(MLAR_value + np.sqrt(MLAR_std_dev+MLAR_var),4)

                print(fieldname, 'MAE:', round(mae_value,1), mae_CI_minus, mae_CI_plus)
                print(fieldname, 'PW20:', round(pw20_value,1),pw20_CI_minus, pw20_CI_plus)
                print(fieldname, 'R2:', round(R2_value,1), R2_CI_minus, R2_CI_plus)
                print(fieldname, 'MALAR:', round(MALAR_value,1), MALAR_CI_minus, MALAR_CI_plus)
                print(fieldname, 'MLAR:', round(MLAR_value,1), MLAR_CI_minus, MLAR_CI_plus)



    except FileNotFoundError:
      print("Sorry, could not find file " + file_name)

if __name__ == "__main__":
    main()