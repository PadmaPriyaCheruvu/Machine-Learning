import math, os

def naiveBayes_train(Dir, loc, stop, stopWord):
    
    cond_Ham = {}
    finalHam = {}
    cond_Spam = {}
    finalSpam = {}
    prior = {}
    docs = {}
    vocabulary = []
    n = 0
    
    
    for (rt, directory, file) in os.walk(loc):
        root_split = rt.split('\\')
        if(len(root_split) <= 1):
            continue
        if root_split[1] == 'ham':
            docs[root_split[1]] = len(file)
        elif root_split[1] == 'spam':
            docs[root_split[1]] = len(file)
        for f in file:
            n = n+1
            f1 = open(os.path.join(rt, f), "r", encoding='latin-1')
            flLine = f1.readlines()
            for ln in flLine:
                words = ln.split()
                if(stop):
                    for word in stopWord:
                        if word in words:
                            words.remove(word)
                if(root_split[1] == 'ham'):
                    for wrd in words:
                        if(wrd not in finalHam.keys()):
                            finalHam[wrd] = 1
                        else:
                            finalHam[wrd] += 1
                elif(root_split[1] == 'spam'):
                    for wrd in words:
                        if(wrd not in finalSpam.keys()):
                            finalSpam[wrd] = 1
                        else:
                            finalSpam[wrd] += 1
                wrdsUnique = set(words)
                for uw in wrdsUnique:
                    if uw not in vocabulary:
                        vocabulary.append(uw)
            f1.close()
    for d in Dir:
        prior[d] = docs[d]/n
        if(d == 'ham'):
            n_count = 0
            for ham in finalHam.values():
                n_count += ham
            for j in vocabulary:
                if(j in finalHam.keys()):
                    frequency = finalHam[j]
                else:
                    frequency = 0
                cond_Ham[j] = (frequency+1)/(n_count+len(finalHam.keys()))
        elif(d == 'spam'):
            n_spam = 0
            for s in finalSpam.values():
                n_spam += s
            for j in vocabulary:
                if(j in finalSpam.keys()):
                    freq = finalSpam[j]
                else:
                    freq = 0
                cond_Spam[j] = (freq+1)/(n_spam+len(finalSpam.keys()))
    return vocabulary, prior, cond_Ham, cond_Spam

def naiveBayes(Dir, vocabulary, prior, cond_Ham, cond_Spam, loc, stop, stopWord):
    actual = {}
    score = {}
    classf = {}
    for (rt, directory, file) in os.walk(loc):
        root_split = rt.split('\\')
        for f in file:
            f1 = open(os.path.join(rt, f), "r", encoding='latin-1')
            flLine = f1.readlines()
            for ln in flLine:
                words = ln.split()
                if(stop):
                    for word in stopWord:
                        if word in words:
                            words.remove(word)
            for d in Dir:
                score[d] = math.log2(prior[d])
                if(d == 'ham'):
                    for wrd in words:
                        if wrd in vocabulary:
                            score[d] += math.log2(cond_Ham[wrd])
                elif(d == 'spam'):
                    for wrd in words:
                        if wrd in vocabulary:
                            score[d] += math.log2(cond_Spam[wrd])
            probMax = max(score.values())
            classf[f] = [k for k, v in score.items() if v == probMax][0]
            if root_split[1] == 'ham':
                actual[f] = root_split[1]
            elif root_split[1] == 'spam':
                actual[f] = root_split[1]
    return classf, actual

Dir = ['ham', 'spam']
vocabulary, prior, conditional_ham, conditional_spam = naiveBayes_train(Dir, 'train', False, [])
classf, actual = naiveBayes(
    Dir, vocabulary, prior, conditional_ham, conditional_spam, 'test', False, [])
sumCorrect = 0
for k in classf.keys():
    if(classf[k] == actual[k]):
        sumCorrect += 1
acc = (sumCorrect/len(classf.keys()))*100
print('\n NAIVE BAYES CLASSIFIER \n Accuracy with stop words =', acc)
f = open(os.path.join(os.getcwd(), 'stopWord.txt'), "r")
stopWords = f.read().splitlines()
vocabulary, prior, conditional_ham, conditional_spam = naiveBayes_train(Dir, 'train', True, stopWords)
classf, actual = naiveBayes(
    Dir, vocabulary, prior, conditional_ham, conditional_spam, 'test', True, stopWords)
sumCorrect = 0
for k in classf.keys():
    if(classf[k] == actual[k]):
        sumCorrect += 1
acc = (sumCorrect/len(classf.keys()))*100
print('NAIVE BAYES CLASSIFIER \n Accuracy without stop words =', acc)
