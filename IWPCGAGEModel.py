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

def targetINR2(targetINR):
    INRlist = list(targetINR.split("-"))
    if '3.0' in INRlist or '3,0' in INRlist:
        return 3.0
    else:
        if '2.5' in INRlist or '2,5' in INRlist:
            return 2.5
        else:
            return targetINR

def DWTorPE(indication):
    if str(indication) in '1' or str(indication) in '2':
        return 1
    else:
        return 0

def BSA(height, weight):
    return 0.007194*height**0.725*weight**0.425

def PercIn20(true, predicted):
   patients_in_20 = 0
   for i, _ in enumerate(true):
      if 0.8 * true[i] <= predicted[i] <= 1.2 * true[i]:
        patients_in_20 += 1
   return float(100 * patients_in_20 / len(true))

def main():
    file_name = "IWPC_War-PATHFinal.csv"

    try:
        fileinput = open(file_name, 'r')
        file_content = fileinput.readlines()
        if file_content:
            print(file_name, ' does contain content ')
        else:
            print(file_name, ' has no content')
        df = pd.read_csv(file_name,";")
        dfmod = df.iloc[:, [0, 8, 9, 10, 11, 27, 35, 38, 40]]
        print(dfmod)
        factory = {"AgeInYears": -0.0075, "BSA": 0.425, "Target INR": 0.216, "Smoker": 0.108, "Amiodarone": -0.257,
                   "Indication": 0.0784, "Intercept": 0.769}
        dfmod.rename(columns={'Height (cm)': 'Height', 'Weight (kg)': 'Weight'}, inplace=True)
        dfmod.rename(columns={'Amiodarone (Cordarone)': 'Amiodarone','Current Smoker':'Smoker'}, inplace=True)
        dfmod["AgeLower"] = dfmod.Age.str.split('-', expand=True)[0]
        dfmod['AgeLower'] = dfmod['AgeLower'].str.replace('+', '')
        dfmod["AgeInYears"] = dfmod["AgeLower"].astype("float")
        dfmod.drop(['AgeLower'], axis=1, inplace=True)
        dfmod.drop(['Age'], axis=1, inplace = True)
        dfmod["Height"] = dfmod["Height"].str.replace(",", ".")
        dfmod["Weight"] = dfmod["Weight"].str.replace(",", ".")
        dfmod["Weight"] = dfmod["Weight"].astype("float")
        dfmod["Height"] = dfmod["Height"].astype("float")
        dfmod.rename(columns={'Therapeutic Dose of Warfarin': 'WarfarinDose'}, inplace=True)
        dfmod.rename(columns={'Indication for Warfarin Treatment': 'Indication'}, inplace=True)
        dfmod["WarfarinDose"] = dfmod["WarfarinDose"].str.replace(",", ".")
        dfmod["WarfarinDose"] = dfmod["WarfarinDose"].astype("float")
        dfmod.rename(columns={'PharmGKB Subject ID': 'PatientID'}, inplace=True)
        #dfmod['Indication'] = dfmod['Indication'].astype("str")
        dfmod['Indication'] = dfmod.apply(lambda x: DWTorPE(x["Indication"]), axis=1)
        #dfmod["Indication"] = dfmod["Indication"].str.replace(",", ".")
        dfmod["Indication"] = dfmod["Indication"].astype("float")
        dfmod["Target_INR2"] = dfmod.apply(lambda x: targetINR2(x["Target INR"]), axis=1)
        dfmod["Target_INR2"] = dfmod["Target_INR2"].astype("float")
        print(dfmod)
        dfimp = dfmod.iloc[:, [1, 2, 4, 7]]
        print("Columns to be imputed")
        print(dfimp)
        dfmod.drop(['Smoker'], axis=1, inplace=True)
        dfmod.drop(['Height'], axis=1, inplace=True)
        dfmod.drop(['Weight'], axis=1, inplace=True)
        dfmod.drop(['Amiodarone'], axis=1, inplace = True)


        imp_data = mice.MICEData(dfimp)
        imp_data.update_all(100)
        results = []
        for m in range(100):
            for j in range(50):
                 x = imp_data.next_sample()
                 imp_data.data.to_csv('data%02d.csv' % j)
                 dftempimp = x
                 dftemp = pd.concat([dftempimp, dfmod], axis = 1)
                 dftemp["BSA"] = dftemp.apply(lambda x: BSA(x["Height"], x["Weight"]), axis=1)
                 dftemp["Smoker"] = dftemp["Smoker"].astype("float")
                 dftemp["WarfarinPredicted"] = np.exp(
                     factory["Intercept"] + factory["AgeInYears"] * dftemp["AgeInYears"] +
                     factory["BSA"] * dftemp["BSA"] + factory["Target INR"] * dftemp["Target_INR2"] +
                     factory["Amiodarone"] * dftemp["Amiodarone"] +
                     factory["Indication"] * dftemp["Indication"] + factory["Smoker"] * dftemp["Smoker"])
                 dftemp.loc[:, 'WarfarinPredicted'] *= 7
                 dftemp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPCGage_" + str(j) + ".csv",";")
                 mae_current = score_mae(dftemp["WarfarinDose"], dftemp["WarfarinPredicted"])
                 #pw20_current = score_pw20(dftemp["WarfarinDose"], dftemp["WarfarinPredicted"])
                 pw20_current = PercIn20(dftemp["WarfarinDose"], dftemp["WarfarinPredicted"])
                 r2_current = score_r2(dftemp["WarfarinDose"], dftemp["WarfarinPredicted"])
                 results.append({'Iteration':m, 'Imputation':j, 'IWPC': dftemp, 'MAE': mae_current, 'PW20': pw20_current, 'R2': r2_current})
                 imp_data = mice.MICEData(dfimp)

        mean_MAE = 0
        mean_PW20 = 0
        mean_R2 = 0
        n = len(results)
        max_MAE = 0
        max_PW20 = 0

        for k in range(len(results)):

            print(results[k]['Iteration'], results[k]['Imputation'],  results[k]['MAE'],  results[k]['PW20'], results[k]['R2'])
            if results[k]['MAE'] > max_MAE:
                max_MAE = results[k]['MAE']
            if results[k]['PW20'] > max_PW20:
                max_PW20 = results[k]['PW20']
            mean_MAE += results[k]['MAE']
            mean_PW20 +=results[k]['PW20']
            mean_R2 += results[k]['R2']

        print('Running GAGE Clinical Dosing Algorithm on IWPC Data Set')
        print('Mean MAE', mean_MAE/n, 'Mean PW20' ,mean_PW20/n, 'Mean_R2', mean_R2/n)
        print('Max MAE  ', max_MAE, 'Max PW20 ', max_PW20)

    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)


if __name__ == "__main__":
    main()