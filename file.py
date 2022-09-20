from datetime import datetime, timedelta, timezone

def getConfusionMatrix(faults, predicted):
    
    truePositive = []
    trueNegative = []
    falsePositive = []
    falseNegative = []
    predicted = predicted.assign(date_time = list(map( lambda x: datetime.fromtimestamp(x/1000), predicted.timestamp)))
    
    for i, act in faults.iterrows():        
        minTime = datetime.strptime(act.time_preliminary,'%Y-%m-%d %H:%M:%S+08:00')
        maxTime = minTime + timedelta(minutes=5)

        truePositive.append(predicted.loc[(predicted.date_time > minTime) & (predicted.date_time < maxTime) & (predicted.predition == 1) & (predicted.source == act.ground_truth)])
        trueNegative.append(predicted.loc[(predicted.date_time > minTime) & (predicted.date_time < maxTime) & (predicted.predition == 0) & (predicted.source == act.ground_truth)])
        falsePositive.append(predicted.loc[(predicted.date_time < minTime) | (predicted.date_time > maxTime) & (predicted.predition == 1)])
        falseNegative.append(predicted.loc[(predicted.date_time < minTime) | (predicted.date_time > maxTime) & (predicted.predition == 0)])

        del maxTime, minTime
    
    else: #remove all empty row
        truePositive = list(filter(lambda dfTP: not dfTP.empty, truePositive))        
        trueNegative = list(filter(lambda dfTN: not dfTN.empty, trueNegative))
        falsePositive = list(filter(lambda dfFP: not dfFP.empty, falsePositive))
        falseNegative = list(filter(lambda dfFN: not dfFN.empty, falseNegative))

        count = 0
        for i, trueP in falsePositive:
            count = count + len(trueP)
        print("True Positive: ", count)


        # print("\nTrue Positive: ", len(truePositive))
        # print("True Negative: ", len(trueNegative))
        # print("False Positive: ", len(falsePositive))
        # print("False Negative: ", len(falseNegative))
        # print((falseNegative))

    return 