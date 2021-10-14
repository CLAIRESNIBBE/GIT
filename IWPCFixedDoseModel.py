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
        dfmod = df.iloc[:, [0,  38]]
        dfmod.rename(columns={'Therapeutic Dose of Warfarin': 'WarfarinDose'}, inplace=True)
        dfmod["WarfarinDose"] = dfmod["WarfarinDose"].str.replace(",", ".")
        dfmod["WarfarinDose"] = dfmod["WarfarinDose"].astype("float")
        dfmod.rename(columns={'PharmGKB Subject ID': 'PatientID'}, inplace=True)
        print(dfmod)
        results = []
        factory = {"Intercept": 35}
        for m in range(100):
            for j in range(50):
                 dftemp = dfmod

                 dftemp["WarfarinPredicted"] = factory["Intercept"]
                 dftemp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPCFixed_" + str(j) + ".csv",";")
                 mae_current = score_mae(dftemp["WarfarinDose"], dftemp["WarfarinPredicted"])
                 pw20_current = score_pw20(dftemp["WarfarinDose"], dftemp["WarfarinPredicted"])
                 r2_current = score_r2(dftemp["WarfarinDose"], dftemp["WarfarinPredicted"])
                 #print("MAE and PW20 and r2 for imputation " + str(j),mae_current, pw20_current, r2_current)
                 #iwpc_formatted = format_summary(dftemp5)
                 #dftemp5.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Clinic.csv", ";ag")
                 results.append({'Iteration':m, 'Imputation':j, 'IWPC': dftemp, 'MAE': mae_current, 'PW20': pw20_current, 'R2': r2_current})

        mean_MAE = 0
        mean_PW20 = 0
        mean_R2 = 0
        n = len(results)

        for k in range(len(results)):
            #print(results[k]['Iteration'], results[k] ['Imputation'], results[k]['IWPC'], results[k]['MAE'], results[k]['PW20'], results[k]['R2'])
            print(results[k]['Iteration'], results[k]['Imputation'],  results[k]['MAE'],  results[k]['PW20'], results[k]['R2'])
            mean_MAE += results[k]['MAE']
            mean_PW20 +=results[k]['PW20']
            mean_R2 += results[k]['R2']

        print('Running the Fixed Dose Clinical Dosing Algorithm on IWPC Data Set')
        print('Mean MAE', mean_MAE/n, 'Mean PW20' ,mean_PW20/n, 'Mean_R2', mean_R2/n)


    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)


if __name__ == "__main__":
    main()