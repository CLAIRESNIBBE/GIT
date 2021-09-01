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


def inducer(drug1, drug2, drug3):
    if (drug1 > 0 or drug2 > 0 or drug3 > 0):
        return 1
    else:
        return 0

def BSA(height, weight):
    return 0.007194*height**0.725*weight**0.425


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
    for i, _ in enumerate(true):
        if 0.8 * true[i] < predicted[i] < 1.2 * true[i]:
            patients_in_20 += 1
    return float(100 * patients_in_20 / len(true))


def main():
    file_name = "IWPCEthnic.csv"

    try:
        fileinput = open(file_name, 'r')
        file_content = fileinput.readlines()
        if file_content:
            print(file_name, ' does contain content ')
        else:
            print(file_name, ' has no content')
        df = pd.read_csv(file_name, ";")
        dfmod = df.drop(['Gender'], axis = 1)
        dfgen = df['Gender']
        print(dfmod)
        dfmod["Dose_mg_week"] = dfmod["Dose_mg_week"].astype("float")
        dfmod["AgeLower"] = dfmod['Age_years'].str.split('_', expand=True)[0]
        dfmod['AgeLower'] = dfmod['AgeLower'].str.replace('+', '')
        dfmod["AgeDecades"] = dfmod["AgeLower"].astype("float") * 0.1
        dfmod["AgeYears"] = dfmod["AgeLower"].astype("float")
        dfmod.drop(['AgeLower'], axis=1, inplace=True)
        dfmod.drop(['Age_years'], axis = 1, inplace = True)
        dfmod["Weight_kg"] = dfmod["Weight_kg"].astype("float")
        dfmod["Height_cm"] = dfmod["Height_cm"].astype("float")
        dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
        dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
        dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
        dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
        dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)
        factor_IWPC = {"Age": -0.2546, "Height": 0.0118, "Weight": 0.0134, "Inducer": 1.2799, "Amiodarone": -0.5695,"Intercept": 4.4436}
        factor_Gage = {"Age": -0.0075,"BSA": 0.425,"TargetINR": 0.216, "Smoker": 0.108,"Amiodarone": -0.257,"Indication": 0.0784, "Intercept": 0.769}
        factor_Fixed = {"Intercept": 35}
        factor_WarPath = {'Intercept': 20.2832, 'Age': -0.0656, 'Weight': 0.2178, 'TargetINR': 0.73190}
        print("Columns to be imputed")
        print(dfmod.info())
        imp_data = mice.MICEData(dfmod)
        models = []
        models.append('IWPC')
        models.append('WarPATH')
        models.append('Gage')
        models.append('Fixed')
        results = []
        for m in range(100):
            x = imp_data.next_sample()
            imp_data.data.to_csv('data%02d.csv' % m)
            dfmodnew = x
            dftemp = pd.concat([dfmodnew, dfgen], axis=1)
            dftemp['BSA'] = dftemp.apply(lambda x: BSA(x["Height_cm"],x["Weight_kg"]), axis=1)
            dftemp['IWPC_Pred'] = round(np.square(factor_IWPC["Intercept"] + factor_IWPC["Age"] * dftemp["AgeDecades"]+factor_IWPC["Height"]*dftemp["Height_cm"]+factor_IWPC["Weight"]*dftemp["Weight_kg"]+factor_IWPC["Amiodarone"]*dftemp["Amiodarone"]+factor_IWPC["Inducer"]*dftemp["Inducer"]),0)
            dfKey = dftemp[['IWPC_Pred', 'Dose_mg_week']]
            results.append({'Imp ':m,'model': 'IWPC','predactual':dfKey, 'MAE': 0  ,'PW20': 0})
            dftemp["Gage_Pred"] = round(7*np.exp(
                     factor_Gage["Intercept"] + factor_Gage["Age"] * dftemp["AgeYears"] +
                     factor_Gage["BSA"]*dftemp["BSA"] + factor_Gage["TargetINR"] * dftemp["Target_INR"] +
                     factor_Gage["Amiodarone"] * dftemp["Amiodarone"] +
                     factor_Gage["Indication"] * dftemp["Indicationflag"] + factor_Gage["Smoker"]*dftemp["Smoker"]),0)
            dfKey = dftemp[["Gage_Pred","Dose_mg_week"]]
            results.append({'Imp ': m, 'model': 'Gage', 'predactual': dfKey, 'MAE' : 0, 'PW20':0})
            dftemp["WarPATH_Pred"] = round(factor_WarPath['Intercept']+factor_WarPath['Age']*dftemp['AgeYears']+factor_WarPath['Weight']*dftemp['Weight_kg']+factor_WarPath['TargetINR']*dftemp['Target_INR'],0)
            dfKey = dftemp[["WarPATH_Pred", "Dose_mg_week"]]
            results.append({'Imp ': m,'model':'WarPATH','predactual' :dfKey,'MAE' : 0, 'PW20':0})
            dftemp["Fixed_Pred"] = factor_Fixed["Intercept"]
            dfKey = dftemp[["Fixed_Pred","Dose_mg_week"]]
            results.append({'Imp ': m, 'model':'Fixed','predactual' :dfKey, 'MAE' : 0, 'PW20':0})
            dftemp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_" + str(m) + ".csv", ";")
            for k in range(len(models)):
               fieldname=models[k]
               actual = dftemp["Dose_mg_week"]
               predict = dftemp[fieldname+"_Pred"]
               mae_current=score_mae(actual,predict)
               pw20_current = score_pw20(actual, predict)
               print(m, fieldname, actual, predict, mae_current, pw20_current)
               for j in range(len(results)):
                  if (results[j]['model'] == fieldname):
                     results[j]['MAE']+=mae_current
                     results[j]['PW20']+=pw20_current
            imp_data = mice.MICEData(dfmod)


    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)

if __name__ == "__main__":
    main()