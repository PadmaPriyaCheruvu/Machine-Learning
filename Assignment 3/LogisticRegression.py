import math,os

hamPath = 'ham'
bKey = 'b_IA_s'
spamPath = 'spam'

val = {hamPath: 1.0, spamPath: 0.0}
totalWordCount = {hamPath: 0.0, spamPath: 0.0}

def trainLogisticRegression(data, vocabulary, itr, learn_rate, lmbda):
    wght = weight_init(vocabulary)

    for i in range(0, itr):
        errorTotal = {}

        for classType in data:
            for dataItem in data[classType]:
                parameters = getParams(dataItem)
                error_class = val[classType] - probCalculate(parameters, wght)

                if error_class != 0:
                    for param in parameters.keys():
                        if(param in errorTotal.keys()):
                            errorTotal[param] = errorTotal[param] +  (
                                parameters[param] * error_class)
                        else:
                            errorTotal[param] = (
                                parameters[param] * error_class)
        for wt in wght.keys():
            if wt in errorTotal:
                wght[wt] = wght[wt] + \
                    (learn_rate * errorTotal[wt]) - (learn_rate * lmbda * wght[wt])
    return wght


def testLogisticRegression(testDataLoc, wght):
    accuracy = {1: 0.0, 0: 0.0}

    finalHamPath = os.getcwd() + '\\' + testDataLoc + '\\' + hamPath

    for name_file in os.listdir(finalHamPath):
        parameters = pullParams(finalHamPath, name_file)
        classWeightedTot = weighted_sum(parameters, wght)
        if(classWeightedTot >= 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0

    finalSpamPath = os.getcwd() + '\\' + testDataLoc + '\\' + spamPath

    for name_file in os.listdir(finalSpamPath):
        parameters = pullParams(finalSpamPath, name_file)
        classWeightedTot = weighted_sum(parameters, wght)
        if(classWeightedTot < 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    return (accuracy[1] * 100) / sum(accuracy.values())

def getWord(filePath):
    wrd = []
    try:
        with open(filePath) as fi:
            wrd = [word for line in fi for word in line.split()]
    except OSError as err:
        print(err.message)
    finally:
        return wrd


def getData(dataPath):
    data = {hamPath: [], spamPath: []}
    fileLocation = os.getcwd() + '\\' + dataPath + '\\' + hamPath

    for name_file in os.listdir(fileLocation):
        wrd = getWord(fileLocation + '\\' + name_file)

        if len(wrd) > 0:
            data[hamPath].append(wrd)
        totalWordCount[hamPath] += 1.0

    fileLocation = os.getcwd() + '\\' + dataPath + '\\' + spamPath

    for name_file in os.listdir(fileLocation):
        wrd = getWord(fileLocation + '\\' + name_file)

        if len(wrd) > 0:
            data[spamPath].append(wrd)
        totalWordCount[spamPath] += 1.0
    return data

def weight_init(vocabulary):
    wght = {bKey: 0.0}
    for word in vocabulary:
        wght[word] = 0.0
    return wght


def weighted_sum(parameters, wght):
    wghtdSum = 0.0
    for param, value in parameters.items():
        if param in wght:
            wghtdSum = value * wght[param] + wghtdSum  
    return wghtdSum
    
def createVocab(data, skip_word_list):
    vocabulary = []
    for classType in data.keys():
        for dataItem in data[classType]:
            for word in dataItem:
                if word not in vocabulary and word.lower() not in skip_word_list:
                    vocabulary.append(word)
    return vocabulary


def getParams(document):
    parameters = {bKey: 1.0}
    for word in document:
        parameters[word] = document.count(word)
    return parameters
    
def getStopWords(name_file):
    fileLocation = os.getcwd() + '\\' + name_file
    wrd = getWord(fileLocation)
    return wrd

def probCalculate(parameters, wght):
    wghtdSum = weighted_sum(parameters, wght)
    try:
        expVal = 1.0 * math.exp(wghtdSum) 
    except OverflowError:
        return 1
    return round((expVal) / (1.0 + expVal), 4)

def pullParams(dir_file, name_file):   
    parameters = {bKey: 1.0}
    wrd = getWord(dir_file + '\\' + name_file)
    for w in wrd:
        parameters[w] = wrd.count(w)
    return parameters

lRate = 0.01
lamb = 10 ** -2   
n_iterations = 200

train_data = getData('train')

vocabulary = createVocab(train_data, [])
wght = trainLogisticRegression(train_data, vocabulary, n_iterations, lRate, lamb)

print(f"LOGISTIC REGRESSION \n Lambda= {lamb} \n Learning rate = {lRate} \n Number of Iterations = {n_iterations} \n  Accuracy before stop words removal : ",
      testLogisticRegression('test', wght))

stop_wrd = getStopWords('stopWords.txt')

restrictVocabulary = createVocab(train_data, stop_wrd)
wght = trainLogisticRegression(train_data, restrictVocabulary, n_iterations, lRate, lamb)

print(f"LOGISTIC REGRESSION \n Lambda= {lamb} \n Learning rate = {lRate} \n Number of Iterations = {n_iterations} \n  Accuracy after stop words removal : ",
      testLogisticRegression('test', wght))
