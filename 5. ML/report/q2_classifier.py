import argparse
import pandas as pd
import sys
import os
import math


#function to parse the command line arguments
def argumentParsing():
    parser = argparse.ArgumentParser(add_help = False, description = "Usage: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>")
    parser.add_argument("-f1", metavar = "train_dataset")
    parser.add_argument("-f2", metavar = "test_dataset")
    parser.add_argument("-o", metavar = "output_file")
    args = parser.parse_args()
    return args.f1, args.f2, args.o

if __name__ == '__main__':
    train_dataset, test_dataset, output_file = argumentParsing()
    if not train_dataset:
        print "train_dataset not entered"
        print "Usage: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>"
        sys.exit(0)

    if not test_dataset:
        print "test_dataset not entered"
        print "Usage: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>"
        sys.exit(0)

    if not output_file:
        print "output_file not entered"
        print "Usage: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>"
        sys.exit(0)

    #reading from training set
    trainingSamples = pd.read_csv(train_dataset, header = None)
    df = pd.DataFrame(data = trainingSamples)
    #creating data frame
    dataframetemp = df[0].str.split(' ')
    ids = list()
    classification = list()
    text = list()

    dataframe = pd.DataFrame(index = range(len(dataframetemp)), columns = ['id', 'classification', 'text'])

    for i in range(len(dataframetemp)):
        # dataframe['id'][i] = dataframe[i][0]
        dataframe['id'][i] = dataframetemp[i][0]
        dataframe['classification'][i] = dataframetemp[i][1]
        dataframe['text'][i] = dataframetemp[i][2:]

    #dict to store frequencies
    freqTbl = {}
    countHam = 0
    countSpam = 0
    #initializing dictionaries by adding frequency of each word in ham or spam column
    for i in range(0,len(dataframe['text'])):
        data = dataframe['text'][i]
        for j in range(0, len(data), 2):
            if freqTbl.get(data[j]) == None:
                freqTbl[data[j]] = (0,0)
            if dataframe['classification'][i] == 'ham':
                freqTbl[data[j]] = (freqTbl[data[j]][0], freqTbl[data[j]][1]+int(data[j+1]))
            else:
                freqTbl[data[j]] = (freqTbl[data[j]][0]+int(data[j+1]), freqTbl[data[j]][1])

    #getting the total count of ham and spam classifications
    for i in dataframe['classification']:
        if i == 'spam':
            countSpam += 1
        else:
            countHam += 1


    #this block calculates the Conditional Probability
    total = countHam + countSpam
    prior={}
    conditionalProb = {}
    probSpam = float(countSpam)/total
    probHam = float(countHam)/total
    V = len(dataframe)
    for k,v in freqTbl.iteritems():
        tmp = v[1]/float(total)
        prior[k] = (1-tmp, tmp)
        conditionalProb[k] = ((v[0] + 1)/(float(countSpam) + V*V), (v[1] + 1)/(float(countHam) + V*V))

    #Loading test set    
    testingSamples = pd.read_csv(test_dataset, header = None)
    df = pd.DataFrame(data = testingSamples)
    dataframetemp = df[0].str.split(' ')

    idsTest = list()
    classificationTest = list()
    textTest = list()

    dataframeTest = pd.DataFrame(index = range(len(dataframetemp)), columns = ['id', 'classification', 'text'])

    #init test set dataframe
    for i in range(len(dataframetemp)):
        # dataframe['id'][i] = dataframe[i][0]
        dataframeTest['id'][i] = dataframetemp[i][0]
        dataframeTest['classification'][i] = dataframetemp[i][1]
        dataframeTest['text'][i] = dataframetemp[i][2:]

    label = list()
    correct = 0
    tempResS = 0
    tempResH = 0

    '''
    the following block of code iterates over each entry in the test set. For each entry it takes a word, its corresponding
    count and its previously calculated conditionalProb. Then the multinomial naive bayes formula is appplied to calculate
    P(spam/ham | word). As the conditionalProb are very small, python gave error during taking power of that small number, so
    we had to take the log of the entire term, and then solve.
    '''

    for i in range(len(dataframeTest['text'])):
        tempResS = 1
        tempResH = 1

        id = dataframeTest['id'][i]
        for j in range(0, len(dataframeTest['text'][i]), 2):
            word = dataframeTest['text'][i][j]
            probS = conditionalProb[word][0]
            probH = conditionalProb[word][1]

            power = dataframeTest['text'][i][j+1]
            
            if probS != 0:
                term1 = math.log(probS)*int(power)
            else:
                term1 = 0
            if probH != 0:
                term2 = math.log(probH)*int(power)
            else:
                term2 = 0

            temp1 = tempResS + term1
            tempResS = temp1
            temp2 = tempResH + term2
            tempResH = temp2

        temp11 = tempResS + math.log(probSpam)
        temp22 = tempResH + math.log(probHam)
        tempResS = temp11
        tempResH = temp22

        if tempResS > tempResH:
            label.append([id, 'spam'])
            if dataframeTest['classification'][i] == 'spam':
                correct += 1
        else:
            label.append([id, 'ham'])
            if dataframeTest['classification'][i] == 'ham':
                correct += 1


    acc = float(correct) / len(dataframeTest.index)
    #writing to csv file
    dfOut = pd.DataFrame(label)
    dfOut.to_csv(output_file, index=False, header = False, sep =' ')
    print acc
