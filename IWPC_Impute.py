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
    return round(0.007194*height**0.725*weight**0.425,1)


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
    #ile_name = "IWPCImputations.csv"
    file_name = "TestingData.csv"

    try:
        fileinput = open(file_name, 'r')
        file_content = fileinput.readlines()
        if file_content:
            print(file_name, ' does contain content ')
        else:
            print(file_name, ' has no content')
        dfnew = pd.read_csv(file_name, ";")
        implist=[]
        implist.append(dfnew)
        if True:
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
                print(df)
                implist.append({"DataFrame":df})
                imp = imp + 1

        results = []
        impResults = []
        models = []
        for df in range(len(implist)):
           dfcurrent = implist[df]["DataFrame"]
           #print(dfcurrent)
           # dfcurrent = dfcurrent.reset_index(drop = True)
           # print(dfcurrent)
           # dfmod = dfcurrent
           dfmod = dfcurrent.drop(['Gender'], axis=1)
           dfgen = dfcurrent['Gender']
           print(dfmod)
           impResults = []
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
           dfmod["INR_Three"] = dfmod.apply(lambda x: INRThree(dfmod["Target_INR"]), axis=1)
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
           factor_WarPath = {'Intercept': 20.2832, 'Age': -0.0656, 'Weight': 0.2178, 'TargetINR': 7.3190}
           if (find(models, 'model', 'IWPC') == -1):
               models.append({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
               models.append({'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
               models.append({'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
               models.append({'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
           dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
           dfmod['IWPC_Pred'] = round(np.square(
               factor_IWPC["Intercept"] + factor_IWPC["Age"] * dfmod["AgeDecades"] + factor_IWPC["Height"] * dfmod[
                   "Height_cm"] + factor_IWPC["Weight"] * dfmod["Weight_kg"] + factor_IWPC["Amiodarone"] * dfmod[
                   "Amiodarone"] + factor_IWPC["Inducer"] * dfmod["Inducer"]), 5)
           dfmod['IWPC_PredRound'] = dfmod.apply(lambda x: DoseRound(x['IWPC_Pred']), axis=1)
           dfKey = dfmod[['IWPC_PredRound', 'IWPC_Pred', 'Dose_mg_week']]
           dfKey.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_PredActual_" + str(df) + ".csv", ";")
           impResults.append(
               {'Imp': df, 'model': 'IWPC', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                'MALAR': 0})
           dfmod["Gage_Pred"] = round(7 * np.exp(
               factor_Gage["Intercept"] + factor_Gage["Age"] * dfmod["AgeYears"] +
               factor_Gage["BSA"] * dfmod["BSA"] + factor_Gage["TargetINR"] * dfmod["Target_INR"] +
               factor_Gage["Amiodarone"] * dfmod["Amiodarone"] +
               factor_Gage["Indication"] * dfmod["Indicationflag"] + factor_Gage["Smoker"] * dfmod["Smoker"]), 5)
           # dfmod["Gage_PredRound"] = dfmod.apply(lambda x: DoseRound(x["Gage_Pred"]), axis=1)
           dfKey = dfmod[["Gage_Pred", "Dose_mg_week"]]
           impResults.append(
               {'Imp': df, 'model': 'Gage', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                'MALAR': 0})
           dfmod["WarPATH_Pred"] = round(
               factor_WarPath['Intercept'] + factor_WarPath['Age'] * dfmod['AgeYears'] + factor_WarPath['Weight'] *
               dfmod['Weight_kg'] + factor_WarPath['TargetINR'] * dfmod['INR_Three'], 5)
           # dfmod["WarPATH_PredRound"] = dfmod.apply(lambda x: DoseRound(x["WarPATH_Pred"]), axis=1)
           dfKey = dfmod[["WarPATH_Pred", "Dose_mg_week"]]
           impResults.append(
               {'Imp': df, 'model': 'WarPATH', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                'MALAR': 0})
           dfmod["Fixed_Pred"] = factor_Fixed["Intercept"]
           # dfmod["Fixed_PredRound"] = dfmod.apply(lambda x: DoseRound(x["Fixed_Pred"]), axis=1)
           dfKey = dfmod[["Fixed_Pred", "Dose_mg_week"]]
           impResults.append(
               {'Imp': df, 'model': 'Fixed', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                'MALAR': 0})
           dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imputed_" + str(df) + ".csv", ";")
           #dfcurrent = dfcurrent.reset_index(drop = True)
           #print(dfcurrent)
           #dfmod = dfcurrent
           impResults = []
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
           dfmod["INR_Three"] = dfmod.apply(lambda x: INRThree(dfmod["Target_INR"]), axis=1)
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
            factor_WarPath = {'Intercept': 20.2832, 'Age': -0.0656, 'Weight': 0.2178, 'TargetINR': 7.3190}
            if (find(models,'model','IWPC')== -1):
                models.append({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
                models.append({'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
                models.append({'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
                models.append({'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0, 'MALAR': 0})
            dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
            dfmod['IWPC_Pred'] = round(np.square(
                factor_IWPC["Intercept"] + factor_IWPC["Age"] * dfmod["AgeDecades"] + factor_IWPC["Height"] * dfmod[
                    "Height_cm"] + factor_IWPC["Weight"] * dfmod["Weight_kg"] + factor_IWPC["Amiodarone"] * dfmod[
                    "Amiodarone"] + factor_IWPC["Inducer"] * dfmod["Inducer"]), 5)
            dfmod['IWPC_PredRound'] = dfmod.apply(lambda x: DoseRound(x['IWPC_Pred']), axis=1)
            dfKey = dfmod[['IWPC_PredRound', 'IWPC_Pred', 'Dose_mg_week']]
            dfKey.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_PredActual_" + str(df) + ".csv", ";")
            impResults.append(
                {'Imp': df, 'model': 'IWPC', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0})
            dfmod["Gage_Pred"] = round(7 * np.exp(
                factor_Gage["Intercept"] + factor_Gage["Age"] * dfmod["AgeYears"] +
                factor_Gage["BSA"] * dfmod["BSA"] + factor_Gage["TargetINR"] * dfmod["Target_INR"] +
                factor_Gage["Amiodarone"] * dfmod["Amiodarone"] +
                factor_Gage["Indication"] * dfmod["Indicationflag"] + factor_Gage["Smoker"] * dfmod["Smoker"]), 5)
            #dfmod["Gage_PredRound"] = dfmod.apply(lambda x: DoseRound(x["Gage_Pred"]), axis=1)
            dfKey = dfmod[["Gage_Pred", "Dose_mg_week"]]
            impResults.append(
                {'Imp': df, 'model': 'Gage', 'predactual': dfKey, 'MAE': 0, 'PW20': 0, 'R2': 0, 'MAPE': 0, 'MLAR': 0,
                 'MALAR': 0})
            dfmod["WarPATH_Pred"] = round(
                factor_WarPath['Intercept'] + factor_WarPath['Age'] * dfmod['AgeYears'] + factor_WarPath['Weight'] *
                dfmod['Weight_kg'] + factor_WarPath['TargetINR'] * dfmod['INR_Three'], 5)
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
                print(dfKey, model, imputation)
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
                    print(fieldname,results[m]['MAE'], results[m]['PW20'], results[m]['R2'], results[m]['MAPE'],results[m]['MLAR'],results[m]['MALAR'])

            #print('***', m, fieldname, results[m]['MAE'], results[m]['PW20'], results[m]['R2'], results[m]['MAPE'],results[m]['MLAR'],results[m]['MALAR'])

        for k in range(len(models)):
            fieldname = models[k]['model']
            mae_value = round(models[k]['MAE'] / 100, 4)
            pw20_value = round(models[k]['PW20'] / 100, 4)
            R2_value = round(models[k]['R2'] / 100, 4)
            MAPE_value = round(models[k]['MAPE'] / 100, 4)
            MLAR_value = round(models[k]['MLAR'] / 100, 4)
            MALAR_value = round(models[k]['MALAR'] / 100, 4)
            print(fieldname, models[k]['MAE'], models[k]['PW20'], models[k]['R2'], models[k]['MAPE'],models[k]['MLAR'],models[k]['MALAR'])
            print(fieldname, 'MAE:', mae_value, 'PW20:', pw20_value, 'R2:', R2_value, 'MAPE:', MAPE_value, 'MLAR:',
                      MLAR_value, 'MALAR:', MALAR_value)

    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)

if __name__ == "__main__":
    main()