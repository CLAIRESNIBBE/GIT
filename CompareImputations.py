import csv
import pandas as pd
import numpy as np
import os
import time
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
#from numpy.testing import assert_equal, assert_allclose, dec
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
           #print(metrics[i]['model'], metrics[i]['metric'], metrics[i]['value'])
           container.append(metrics[i]['value'])
    return container

def collect_Results(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model']== model):
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
    filelist = [f for f in os.listdir(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ImpCompare") if f.endswith(".csv")]
    for f in filelist:
        os.remove(os.path.join(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ImpCompare", f))

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    filesImp1 = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Imputations"):
        for file in files:
          if file.endswith('.csv') :
            filesImp1.append(file)

    filesImp2 = []
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL"):
      for file in files:
         if file.endswith('.csv') :
           filesImp2.append(file)

    for i in range(len(filesImp1)):
      file = filesImp1[i]
      compfile = filesImp2[i]
      dfcomp = pd.read_csv(root + '\\' + compfile, ";")
      df = pd.read_csv('C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\Imputations\\' + file, ";")
      print("Comparing " + file + ' with ' + compfile)
      if True:
        df["Inducer"] = df.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
        df["Amiodarone"] = df.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
        df.drop(["Inducer_status", "Amiodarone_status"], axis=1, inplace=True)
        df["AgeLower"] = df["Age_years"].str.split('_', expand=True)[0]
        df['AgeLower'] = df['AgeLower'].str.replace('+', '')
        df["AgeDecades"] = df["AgeLower"].astype("float") * 0.1
        df["AgeYears"] = df["AgeLower"].astype("float")
        df['AgeYears'] = np.where((df['AgeYears'] <= 18), 18, df['AgeYears'])
        df.drop(['AgeLower'], axis=1, inplace=True)
        df.drop(['Age_years'], axis=1, inplace=True)
        df["Weight_kg"] = df["Weight_kg"].astype("float")
        df["Height_cm"] = df["Height_cm"].astype("float")
        df["Target_INR"] = df["Target_INR"].astype("float")
        df["Dose_mg_week"] = df["Dose_mg_week"].astype("float")
        df["HeightComp"] = dfcomp["Height_cm"]
        df["WeightComp"] = dfcomp["Weight_kg"]
        df["AmiodaroneComp"] = dfcomp["Amiodarone"]
        df["InducerComp"]  = dfcomp["Inducer"]
        df["AgeComp"] = dfcomp["AgeDecades"]
        df["AgeMatch?"] = np.where(df["AgeDecades"] == df['AgeComp'],0, df['AgeDecades'] - df['AgeComp'])
        df["HeightMatch?"] = np.where(df['Height_cm'] == df['HeightComp'], 0, df['Height_cm'] - df['HeightComp'])
        df["WeightMatch?"] = np.where(df['Weight_kg'] == df['WeightComp'], 0, df['Weight_kg'] - df['WeightComp'])
        df["InducerMatch?"]= np.where(df['Inducer'] == df['InducerComp'], 0, df['Inducer'] - df['InducerComp'])
        df["AmiodaroneMatch?"] = np.where (df['Amiodarone'] == df['AmiodaroneComp'], 0, df['Amiodarone']-df['AmiodaroneComp'])
        suffix = str(i).zfill(3)
        df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ImpCompare\IWPCIMPCOMPARE_"  + str(i) + ".csv", ";")










if __name__ == "__main__":
        main()