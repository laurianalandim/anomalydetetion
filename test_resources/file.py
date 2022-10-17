from datetime import datetime, timedelta, timezone

def getConfusionMatrix(faults, predicted):
    import datetime
    
    truePositive = []
    trueNegative = []
    falsePositive = []
    falseNegative = []
    predicted = predicted.assign(date_time = lambda x: df['timestamp'])    
    
    for i, act in faults.iterrows(): 
        minTime = datetime.datetime.strptime(act.time_preliminary,'%Y-%m-%d %H:%M:%S+08:00')
        maxTime = minTime + timedelta(minutes=5)

        truePositive.append(predicted.loc[(predicted.date_time  >= minTime) & (predicted.date_time <= maxTime) & (predicted.outliers == 1) & (predicted.target == act.ground_truth)])
        trueNegative.append(predicted.loc[predicted.outliers == 0])
        falsePositive.append(predicted.loc[(predicted.date_time < minTime) | (predicted.date_time > maxTime) & (predicted.outliers == 1)])
        falseNegative.append(predicted.loc[(predicted.date_time < minTime) | (predicted.date_time > maxTime) & (predicted.outliers == 0)])

        del maxTime, minTime
    
    else: #remove all empty row
        truePositive = list(filter(lambda dfTP: not dfTP.empty, truePositive))        
        trueNegative = list(filter(lambda dfTN: not dfTN.empty, trueNegative))
        falsePositive = list(filter(lambda dfFP: not dfFP.empty, falsePositive))
        falseNegative = list(filter(lambda dfFN: not dfFN.empty, falseNegative))

    return