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
        #print('*Debug MAPE**', (predval[i]-trueval[i])/trueval[i])
        #print('*Debug MAPE***',np.log((predval[i]-trueval[i])/trueval[i]))
    return topsum/len(trueval)*100
    #exponent=(topsum/len(trueval))
    #print('*Debug MAPE*', exponent, len(trueval), topsum)
    #return np.exp(exponent) * 100

def MLAR(trueval, predval):
    # mean log of accuracy ratio
    sum = 0
    for i in range(len(trueval)):
        sum += np.log(predval[i]/trueval[i])
    return sum/len(trueval)

def MALAR(trueval, predval):
    # mean absolute log of absolute ratio
    sum = 0
    for i in range(len(trueval)):
        sum += abs(np.log(predval[i]/trueval[i]))
    return sum/len(trueval)

def RSquared(trueval,predval):
    true_mean = np.mean(trueval)
    topsum = 0
    lowersum  = 0
    for i in range(len(trueval)):
       topsum += np.square((predval[i] - true_mean))
       lowersum += np.square(trueval[i] - true_mean)
    num = topsum
    den = lowersum
    ratio = num/den * 100
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


def ConvertYesNo(variable):
    if variable == "Yes":
        return 1
    elif variable == "No":
        return 0

def PercIn20(true, predicted):
    #sum(as.numeric(c(abs(data$Dose_mg_week - rounded) < 0.2 * data$Dose_mg_week)))*100 / nrow(data)
    patients_in_20 = 0
    for i in range(len(true)):
      if abs(true[i] - predicted[i]) < 0.2 * true[i]:
        patients_in_20 += 1
    return float(100 * patients_in_20 / len(true))

def INRThree(targetINR):
    if (targetINR >= 2.5).all() & (targetINR <= 3.5).all():
        return 1
    else:
        return 0


def DoseRound(predicted):
    return round(predicted/2.5, 0) * 2.5

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
        dfmod["AgeYears"] = dfmod.apply(lambda x:dfmod["AgeLower"].astype("float")
        dfmod.drop(['AgeLower'], axis=1, inplace=True)
        dfmod.drop(['Age_years'], axis = 1, inplace = True)
        dfmod["Weight_kg"] = dfmod["Weight_kg"].astype("float")
        dfmod["Height_cm"] = dfmod["Height_cm"].astype("float")
        dfmod["INR_Three"] = dfmod.apply(lambda x: INRThree(dfmod["Target_INR"]), axis =1)
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

        models = []
        models.append({'model':'IWPC','MAE':0,'PW20':0,'R2':0, 'MAPE':0, 'MLAR':0, 'MALAR':0})
        models.append({'model':'WarPATH','MAE':0, 'PW20':0, 'R2':0, 'MAPE':0, 'MLAR':0,'MALAR':0})
        models.append({'model': 'Gage', 'MAE': 0, 'PW20':0, 'R2':0, 'MAPE':0, 'MLAR':0,'MALAR':0})
        models.append({'model': 'Fixed', 'MAE': 0, 'PW20':0,'R2':0, 'MAPE':0, 'MLAR':0,'MALAR':0})

        results = []
        imp_data = mice.MICEData(dfmod)
        imp_data.update_all(1)
        results = []
        for m in range(100):
            x = imp_data.next_sample()
            imp_data.data.to_csv('data%02d.csv' % m)
            x.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imputedx" + str(m) + ".csv", ";")
            dfmodnew = x
            dftemp = pd.concat([dfmodnew, dfgen], axis=1)
            dftemp['BSA'] = dftemp.apply(lambda x: BSA(x["Height_cm"],x["Weight_kg"]), axis=1)
            dftemp['IWPC_Pred'] = round(np.square(factor_IWPC["Intercept"] + factor_IWPC["Age"] * dftemp["AgeDecades"]+factor_IWPC["Height"]*dftemp["Height_cm"]+factor_IWPC["Weight"]*dftemp["Weight_kg"]+factor_IWPC["Amiodarone"]*dftemp["Amiodarone"]+factor_IWPC["Inducer"]*dftemp["Inducer"]),0)
            dftemp['IWPC_PredRound']= dftemp.apply(lambda x: DoseRound(x['IWPC_Pred']), axis = 1)
            dfKey = dftemp[['IWPC_PredRound', 'IWPC_Pred','Dose_mg_week']]
            dfKey.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_PredActual_" + str(m) + ".csv", ";")
            results.append({'Imp':m,'model': 'IWPC','predactual':dfKey, 'MAE': 0  ,'PW20': 0, 'R2':0,'MAPE':0, 'MLAR' :0, 'MALAR' :0})
            dftemp["Gage_Pred"] = round(7*np.exp(
                     factor_Gage["Intercept"] + factor_Gage["Age"] * dftemp["AgeYears"] +
                     factor_Gage["BSA"]*dftemp["BSA"] + factor_Gage["TargetINR"] * dftemp["Target_INR"] +
                     factor_Gage["Amiodarone"] * dftemp["Amiodarone"] +
                     factor_Gage["Indication"] * dftemp["Indicationflag"] + factor_Gage["Smoker"]*dftemp["Smoker"]),0)
            dftemp["Gage_PredRound"] = dftemp.apply(lambda x: DoseRound(x["Gage_Pred"]), axis = 1)
            dfKey = dftemp[["Gage_PredRound","Gage_Pred","Dose_mg_week"]]
            results.append({'Imp': m, 'model': 'Gage', 'predactual': dfKey, 'MAE' : 0, 'PW20':0, 'R2':0, 'MAPE':0,'MLAR' :0, 'MALAR' :0})
            dftemp["WarPATH_Pred"] = round(factor_WarPath['Intercept']+factor_WarPath['Age']*dftemp['AgeYears']+factor_WarPath['Weight']*dftemp['Weight_kg']+factor_WarPath['TargetINR']*dftemp['INR_Three'],0)
            dftemp["WarPATH_PredRound"] = dftemp.apply(lambda x: DoseRound(x["WarPATH_Pred"]), axis = 1)
            dfKey = dftemp[["WarPATH_PredRound", "WarPATH_Pred","Dose_mg_week"]]
            results.append({'Imp': m,'model':'WarPATH','predactual' :dfKey,'MAE' : 0, 'PW20':0,'R2':0, 'MAPE':0, 'MLAR' :0, 'MALAR' :0})
            dftemp["Fixed_Pred"] = factor_Fixed["Intercept"]
            dftemp["Fixed_PredRound"] = dftemp.apply(lambda x: DoseRound(x["Fixed_Pred"]), axis = 1)
            dfKey = dftemp[["Fixed_PredRound","Fixed_Pred","Dose_mg_week"]]
            results.append({'Imp': m, 'model':'Fixed','predactual' :dfKey, 'MAE' : 0, 'PW20':0, 'R2':0, 'MAPE':0, 'MLAR' :0, 'MALAR' :0})
            dftemp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imputed_" + str(m) + ".csv", ";")
            imp_data = mice.MICEData(dfmod)

        for k in range(len(results)):
            print("K is", k)
            dfKey = results[k]['predactual']
            model = results[k]['model']
            imputation = results[k]['Imp']
            print(dfKey, model, imputation)
            doseTrue = dfKey["Dose_mg_week"]
            dosePred = dfKey[model+'_PredRound']
            dosePred2 = dfKey[model+'_Pred']
            results[k]['MAE'] = round(score_mae(doseTrue, dosePred),3)
            results[k]['PW20'] = round(PercIn20(doseTrue, dosePred),3)
            results[k]['R2'] = round(RSquared(doseTrue, dosePred2),3)
            results[k]['MAPE'] = round((MAPE(doseTrue, dosePred)),3)
            results[k]['MLAR'] =round(MLAR(doseTrue, dosePred),3)
            results[k]['MALAR'] = round(MALAR(doseTrue, dosePred), 3)
            print('**',dfKey,model,imputation,results[k]['MAE'],results[k]['PW20'],results[k]['R2'],results[k]['MAPE'],results[k]['MLAR'],results[k]['MALAR'])

        for k in range(len(models)):
            fieldname = models[k]['model']
            for m in range(len(results)):
                if results[m]['model']==fieldname:
                    models[k]['MAE']+= results[m]['MAE']
                    models[k]['PW20']+= results[m]['PW20']
                    models[k]['R2']+= results[m]['R2']
                    models[k]['MAPE'] += results[m]['MAPE']
                    models[k]['MLAR'] += results[m]['MLAR']
                    models[k]['MALAR'] += results[m]['MALAR']
            print('***', m, fieldname, results[m]['MAE'],results[m]['PW20'],results[m]['R2'], results[m]['MAPE'], results[m]['MLAR'], results[m]['MALAR'])

        for k in range(len(models)):
            fieldname = models[k]['model']
            mae_value = round(models[k]['MAE']/100,1)
            pw20_value = round(models[k]['PW20']/100,1)
            R2_value = round(models[k]['R2']/100,1)
            MAPE_value = round(models[k]['MAPE']/100,1)
            MLAR_value = round(models[k]['MLAR']/100,1)
            MALAR_value = round(models[k]['MALAR']/100,1)
            print(fieldname,'MAE:', mae_value, 'PW20:', pw20_value, 'R2:', R2_value, 'MAPE:', MAPE_value,'MLAR:',MLAR_value,'MALAR:',MALAR_value)

    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)

if __name__ == "__main__":
    main()