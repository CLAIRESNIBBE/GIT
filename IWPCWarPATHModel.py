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

def hivstatus(comorbidities, positive):
    comorblist = list(comorbidities.split(";"))
    hivloc = False
    for x in comorblist:
        if x.strip() == 'HIV':
           hivloc = hivloc or True
    if positive == True:
      if hivloc == True:
        return 1
      else:
        return 0
    else:
       if hivloc == False:
          return 1
       else:
          return 0

def targetINRThree(targetINR):
    INRlist = list(targetINR.split("-"))
    INR3 = False
    for x in INRlist:
        if x.strip()[0] == '3':
            INR3 = INR3 or True
    if INR3 == True:
        return 1
    else:
        return 0

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
        dfmod = df.iloc[:, [0, 8, 9, 10, 12, 35, 38]]
        print(dfmod)
        factory = {'Intercept': 20.2832, 'Age': -0.0656, 'Weight': 0.2178, 'TargetINR': 0.73190, 'HIVPositive': 8.7973,
                   'HIVUnknown': 3.4054}
        dfmod.rename(columns={'Height (cm)': 'Height', 'Weight (kg)': 'Weight'}, inplace=True)
        dfmod["TargetINRThree"] = dfmod.apply(lambda x: targetINRThree(x["Target INR"]), axis=1)
        dfmod["Comorbidities"] = dfmod["Comorbidities"].astype("str")
        dfmod['HIVPositive'] = dfmod.apply(lambda x: hivstatus(x["Comorbidities"], True), axis=1)
        dfmod['HIVUnknown'] = dfmod.apply(lambda x: hivstatus(x["Comorbidities"], False), axis=1)
        dfmod["AgeLower"] = dfmod.Age.str.split('-', expand=True)[0]
        dfmod['AgeLower'] = dfmod['AgeLower'].str.replace('+', '')
        dfmod["AgeinDecades"] = dfmod["AgeLower"].astype("float") * 0.1
        dfmod["AgeInYears"] = dfmod["AgeLower"].astype("float")
        dfmod.drop(['AgeLower'], axis=1, inplace=True)
        dfmod.drop(['Age'], axis=1, inplace = True)
        dfmod["Height"] = dfmod["Height"].str.replace(",", ".")
        dfmod["Weight"] = dfmod["Weight"].str.replace(",", ".")
        dfmod["Weight"] = dfmod["Weight"].astype("float")
        dfmod["Height"] = dfmod["Height"].astype("float")
        dfmod.rename(columns={'Therapeutic Dose of Warfarin': 'WarfarinDoseTrue'}, inplace=True)
        dfmod["WarfarinDoseTrue"] = dfmod["WarfarinDoseTrue"].str.replace(",", ".")
        dfmod["WarfarinDoseTrue"] = dfmod["WarfarinDoseTrue"].astype("float")
        dfmod.rename(columns={'PharmGKB Subject ID': 'PatientID'}, inplace=True)
        print(dfmod)
        dfimp = dfmod.iloc[:, [1, 2, 9]]
        dfmod.drop(['Height'], axis = 1, inplace = True)
        dfmod.drop(['Weight'], axis = 1, inplace = True)
        dfmod.drop(['Comorbidities'], axis = 1, inplace = True)
        print("Columns to be imputed")
        print(dfimp)
        imp_data = mice.MICEData(dfimp)
        imp_data.update_all(100)
        results = []
        for m in range(100):
            for j in range(50):
                 x = imp_data.next_sample()
                 imp_data.data.to_csv('data%02d.csv' % j)
                 dftempimp = x
                 dftempimp.drop(['AgeInYears'],axis = 1, inplace = True)
                 dftemp = pd.concat([dftempimp, dfmod], axis = 1)
                 dftemp['WarfarinDosePred'] = np.round(factory['Intercept'] + factory['Age'] * dftemp['AgeInYears'] + factory['Weight'] * dftemp['Weight'] + factory['TargetINR'] * dftemp['TargetINRThree'] + factory[
                                              'HIVPositive'] * dftemp['HIVPositive'] + factory['HIVUnknown'] * dftemp['HIVUnknown'],2)
                 np.exp(
                     factory["Intercept"] + factory["AgeInYears"] * dftemp["AgeInYears"] +
                     factory["BSA"] * dftemp["BSA"] + factory["Target INR"] * dftemp["Target_INR2"] +
                     factory["Amiodarone"] * dftemp["Amiodarone"] +
                     factory["Indication"] * dftemp["Indication"] + factory["Smoker"] * dftemp["Smoker"])
                 dftemp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPCWarPath_" + str(j) + ".csv",";")
                 mae_current = score_mae(dftemp["WarfarinDoseTrue"], dftemp["WarfarinDosePred"])
                 #pw20_current = score_pw20(dftemp["WarfarinDoseTrue"], dftemp["WarfarinDosePred"])
                 pw20_current = PercIn20(dftemp["WarfarinDoseTrue"], dftemp["WarfarinDosePred"])
                 r2_current = score_r2(dftemp["WarfarinDoseTrue"], dftemp["WarfarinDosePred"])
                 results.append({'Iteration':m, 'Imputation':j, 'IWPC': dftemp, 'MAE': mae_current, 'PW20': pw20_current, 'R2': r2_current})
                 imp_data = mice.MICEData(dfimp)

        mean_MAE = 0
        mean_PW20 = 0
        mean_R2 = 0
        n = len(results)

        for k in range(len(results)):

            print(results[k]['Iteration'], results[k]['Imputation'],  results[k]['MAE'],  results[k]['PW20'], results[k]['R2'])
            mean_MAE += results[k]['MAE']
            mean_PW20 +=results[k]['PW20']
            mean_R2 += results[k]['R2']

        print('Running WARPATH Clinical Dosing Algorithm on IWPC Data Set')
        print('Mean MAE', mean_MAE/n, 'Mean PW20' ,mean_PW20/n, 'Mean_R2', mean_R2/n)


    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)


if __name__ == "__main__":
    main()