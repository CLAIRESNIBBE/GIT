# Collect 100 MAEs for War-PATH, 100 MAEs for IWPC,100 MAEs for Gage and 100 MAEs for Fixed.
#        100 PW20's for each of War-PATH, IWPC, Gage and Fixed.
#        100 R2's for each of War-PATH, IWPC, Gage and Fixed.
#        100 MAPEs (use MALAR) for War-Path, 100 MALAR for IWPC, 100 MALAR for Gage and 100 MALAR for Fixed
#        100 Bias (MLAR) for War-Path, 100 Bias for IWPC, 100 Bias for Gage and 100 Bias for Fixed

metrics = []
# listmetrics = ['MAE', 'PW20', 'R2', 'MALAR', 'MLAR']
# listmodels = ['WarPATH', 'IWPC', 'Gage', 'Fixed']
for m in range(len(listmodels)):
    modelinlist = listmodels[m]
    for l in range(len(metric_columns)):
        metric = metric_columns[l]
        for j in range(len(results)):
            model = results[j]['model']
            metric_value = results[j][metric]
            if model == modelinlist:
                metrics.append({'model': model, 'metric': metric, 'value': metric_value})

for j in range(len(metrics)):
    for keys in metrics[j].items():
        print(keys)

datar = np.random.randint(10, size=(100, 5))
# metric_columns = ['MAE', 'PW20', 'R2', 'MALAR', 'MLAR']
df_WARPATH = pd.DataFrame(data=datar, columns=metric_columns)
df_IWPC = pd.DataFrame(data=datar, columns=metric_columns)
df_GAGE = pd.DataFrame(data=datar, columns=metric_columns)
df_FIXED = pd.DataFrame(data=datar, columns=metric_columns)
for i in range(len(metric_columns)):
    current_metric = metric_columns[i]
    df_WARPATH[current_metric] = np.array(collect_Metrics(metrics, 'WarPATH', current_metric))
    df_IWPC[current_metric] = np.array(collect_Metrics(metrics, 'IWPC', current_metric))
    df_GAGE[current_metric] = np.array(collect_Metrics(metrics, 'Gage', current_metric))
    df_FIXED[current_metric] = np.array(collect_Metrics(metrics, 'Fixed', current_metric))

print(df_WARPATH)
df_WARPATH.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarPATH_metrics" + ".csv", ";")
print(df_IWPC)
df_IWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_metrics" + ".csv", ";")
print(df_GAGE)
df_GAGE.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Gage_metrics" + ".csv", ";")
print(df_FIXED)
df_FIXED.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Fixed_metrics" + ".csv", ";")

if False:
    print('Confidence Intervals for the 4 Models')
    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        df_WARPATH['Mean' + current_metric] = np.mean(df_WARPATH[current_metric])
        meanvalue = np.mean(df_WARPATH[current_metric])
        confintcore = confintlimit95(df_WARPATH[current_metric])
        print('WARPATH ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)

    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        meanvalue = np.mean(df_IWPC[current_metric])
        confintcore = confintlimit95(df_IWPC[current_metric])
        print('IWPC ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)

    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        meanvalue = np.mean(df_GAGE[current_metric])
        confintcore = confintlimit95(df_GAGE[current_metric])
        print('GAGE ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)

    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        meanvalue = np.mean(df_FIXED[current_metric])
        confintcore = confintlimit95(df_FIXED[current_metric])
        print('Fixed Dose ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)
        print('Confidence Intervals for the 4 Models')
    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        df_WARPATH['Mean' + current_metric] = np.mean(df_WARPATH[current_metric])
        meanvalue = np.mean(df_WARPATH[current_metric])
        confintcore = confintlimit95(df_WARPATH[current_metric])
        print('WARPATH ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)

    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        meanvalue = np.mean(df_IWPC[current_metric])
        confintcore = confintlimit95(df_IWPC[current_metric])
        print('IWPC ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)

    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        meanvalue = np.mean(df_GAGE[current_metric])
        confintcore = confintlimit95(df_GAGE[current_metric])
        print('GAGE ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)

    for i in range(len(metric_columns)):
        current_metric = metric_columns[i]
        meanvalue = np.mean(df_FIXED[current_metric])
        confintcore = confintlimit95(df_FIXED[current_metric])
        print('Fixed Dose ', current_metric, meanvalue, meanvalue - confintcore, meanvalue + confintcore)
