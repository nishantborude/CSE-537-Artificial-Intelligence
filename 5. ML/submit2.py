import random
import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd
from math import log
from scipy.stats import chisquare

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain, Ytrain, Xtest = [],[],[]
	with open(ftrain, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtrain.append(rw)

	with open(ftest, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	with open(ftrain_label, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = int(row[0])
	        Ytrain.append(rw)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest


num_feats = 274


def getEntropy(positive, negative, total):
    posRatio = (float(positive) / float(total))
    negRatio = (float(negative) / float(total))
    entropy = 0.0
    if posRatio != 0:
        entropy = -posRatio * log(posRatio, 2)
    if negRatio != 0:
        entropy -= negRatio * log (negRatio, 2)
    return entropy
 
#featureValSet = [1, 2, 3, 4, 5]

def calculateOccurences(Data, featureNum):
    count = 0
    # count of pos and neg for 5 vals
    pos = [0] * 5
    neg = [0] * 5
    # Using pandas

    for i in range(0, len(pos)):
        yPos = Data['Label'] == 1
        yNeg = Data['Label'] == 0
        xPos = Data[featureNum] == i+1
        pos[i] = Data[xPos & yPos].shape[0]
        neg[i] = Data[yNeg & xPos].shape[0]
    return pos,neg
    '''
    # Serial code where data is in array
    for i in range(0, len(Xdata)):
        # print Xdata[i][featureNum]-1
        if Ydata[i] == 1:
            pos[Xdata[i][featureNum]-1] += 1
        else:
            neg[Xdata[i][featureNum]-1] += 1
    return pos,neg
    '''

def getChiSquareVal(p, n, N, Data, selectedFeature):
    p_prime = [0] * 5
    n_prime = [0] * 5
    p_obs = [0] * 5
    n_obs = [0] * 5
    for i in range(0, 5):
        yPos = Data['Label'] == 1
        yNeg = Data['Label'] == 0
        xPos = Data[selectedFeature] == i+1
        p_obs[i] = float(Data[xPos & yPos].shape[0])
        n_obs[i] = float(Data[xPos & yNeg].shape[0])
        p_prime[i] = float(p) * float(Data[yPos].shape[0]) / float(N)
        n_prime[i] = float(n) * float(Data[yNeg].shape[0]) / float(N)
         
    expected = p_prime + n_prime
    observed = p_obs + n_obs
    #print 'E',  expected
    #print 'O', observed
    return chisquare(observed, expected)

def getMaxInfoGainFeature(remainingFeatures, Data):

    positive = Data[Data['Label'] == 1].shape[0]
    negative = Data[Data['Label'] == 0].shape[0]

    #positive = Ydata.count(1)
    #negative = Ydata.count(0)
    total = positive + negative
    
    rootEntropy = getEntropy(positive, negative, total)
    
    #print "ROOT: p=", positive, " n=", negative, " e=", rootEntropy
    if rootEntropy == 0:
        if total == positive:
            return 'T'
        else:
            return 'F'

    featureEntropy = []
    for feature in remainingFeatures:
        pos, neg = calculateOccurences(Data, feature)
        infoGainSum = 0
        #print "For Feature ", feature
        #print "p=", pos, " n=", neg
        
        inforGainSum = 0
        for i in range(len(pos)):
            cTotal = pos[i] + neg[i]
            if cTotal > 0:
                cEntropy = getEntropy(pos[i], neg[i], cTotal)
                infoGainSum += (float(cTotal)/float(total)) * float(cEntropy)
        infoGain = rootEntropy - infoGainSum
        featureEntropy.append(infoGain)  
    #print "Feature Entropy", featureEntropy
    maxGain = featureEntropy.index(max(featureEntropy))
    #print "Max: ", max(featureEntropy)
    cVal = getChiSquareVal(positive, negative, total, Data, remainingFeatures[maxGain])
    #print "CVAL: ", cVal.pvalue
    if cVal.pvalue > pval:
        yPos = Data['Label'] == 1
        yNeg = Data['Label'] == 0
        xPos = Data[remainingFeatures[maxGain]] == i+1
        pos[i] = Data[xPos & yPos].shape[0]
        neg[i] = Data[yNeg & xPos].shape[0]
        if pos[i] >= neg[i]:
            return 'T'
        else:
            return 'F'
    return remainingFeatures[maxGain]


#A random tree construction for illustration, do not use this in your code!
def create_random_tree(depth):
    if(depth >= 7):
        if(random.randint(0,1)==0):
            return TreeNode('T',[])
        else:
            return TreeNode('F',[])

    feat = random.randint(0,273)
    root = TreeNode(data=str(feat))

    for i in range(5):
        root.nodes[i] = create_random_tree(depth+1)
    return root
'''
def createId3TreeBWise(Data, fSet):
    if len(fSet) == 0:
        return -1;
    nextRoot = getMaxInfoGainFeature(fSet, Data)
    print "Next Root", nextRoot 
    root = TreeNode(nextRoot)
    
    nodesQ = Queue()
    nodesQ.push(root)

    while nodesQ.empty() == False:
        cNode = nodesQ.pop()
        rootFeature = cNode.data

        if nextRoot == 'T' or nextRoot == 'F':
            continue
        fSet.remove(rootFeature)    
        for i in range(0, 5):
            featureWise = Data[rootFeature] == i+1
            nData = Data[featureWise]
            if nData.shape[0] > 0:
                root.nodes[i] = createId3TreeRec(nData, fSet)
            else:
                root.nodes[i] = -1
    return root
'''


def createId3TreeRec(Data, fSet):
    if len(fSet) == 0:
        return -1;
    nextRoot = getMaxInfoGainFeature(fSet, Data)
    print "Next Root", nextRoot 
    root = TreeNode(nextRoot)
    print "Data ", Data.shape[0]
    if nextRoot == 'T' or nextRoot == 'F':
        return root
    fSet.remove(nextRoot)    
    for i in range(0, 5):
        featureWise = Data[nextRoot] == i+1
        nData = Data[featureWise]
        if nData.shape[0] > 0:
            root.nodes[i] = createId3TreeRec(nData, fSet)
        else:
            root.nodes[i] = -1
    return root

def create_id3_tree(Xdata, Ydata):
    print "XDATA"
    print len(Xdata), len(Xdata[0])
    print "YDATA"
    print len(Ydata)
    featureSet = list(range(0,num_feats))
    mergedData = pd.concat([Xdata, Ydata], axis = 1)
    return createId3TreeRec(mergedData, featureSet)



def BFS(root):
    
    queue = list()
    
    queue.append(root)
    
    while len(queue)>0:
        n = len(queue)
        r = list()
        for i in range(n):
            node = queue[0]
            queue.remove(node)
            if node is not None:
                r.append(node.data)
                for children in node.nodes:
                    if children!=-1:
                        queue.append(children)
        
        print (r)

'''
def BFS(node):
    nNode = [node]
    while len(nNode) > 0:
        nodeV = nNode[0]
        nNode = nNode[1:]
        if nodeV != '-1':
            print nodeV.data, '[ '
            for child in nodeV.nodes:
                if child != -1:
                    print child.data
                    nNode.append(child)
            print ']'
'''
parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']



Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

print("Training...")
# s = create_random_tree(4)
XTrainDF = pd.DataFrame(Xtrain)
YTrainDF = pd.DataFrame(Ytrain, columns=['Label'])
s = create_id3_tree(XTrainDF, YTrainDF)
BFS(s)
s.save_tree(tree_name)
print("Testing...")
Ypredict = []
#generate random labels
for i in range(0,len(Xtest)):
	Ypredict.append([np.random.randint(0,2)])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")








