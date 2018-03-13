import random
import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd
from math import log
from scipy.stats import chisquare
import sys
sys.setrecursionlimit(10000)

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
countG = 0
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data
        global countG
        countG += 1


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

# 
# Computes the entropy for given positive negative and total
#
def getEntropy(positive, negative, total):
    posRatio = (float(positive) / float(total))
    negRatio = (float(negative) / float(total))
    entropy = 0.0
    if posRatio != 0:
        entropy = -posRatio * log(posRatio, 2)
    if negRatio != 0:
        entropy -= negRatio * log(negRatio, 2)
    return entropy
 
#
# Calculates number of positive and negative label occurence 
# in given data
#
def calculateOccurences(Data, featureNum):
    count = 0
    # count of pos and neg for 5 vals
    pos = [0] * 5
    neg = [0] * 5
    
    for i in range(0, len(pos)):
        yPos = Data['Label'] == 1
        yNeg = Data['Label'] == 0
        xPos = Data[featureNum] == i+1
        pos[i] = Data[xPos & yPos].shape[0]
        neg[i] = Data[yNeg & xPos].shape[0]
    return pos,neg
    

#
# returns Chi-Square Value for given data
#
def getChiSquareVal(p, n, N, Data, selectedFeature):
    p_prime = []
    p_obs = [] 
    for i in range(0, 5):
        yPos = Data['Label'] == 1
        yNeg = Data['Label'] == 0
        xPos = Data[selectedFeature] == i+1
        tmpP = float(Data[xPos & yPos].shape[0])
        tmpN = float(Data[xPos & yNeg].shape[0])
        L = Data[xPos].shape[0]
        if tmpP + tmpN > 0:
            p_obs.append(tmpP)
            p_obs.append(tmpN)
            p_prime.append(float(L) * float(Data[yPos].shape[0]) / float(N))
            p_prime.append(float(L) * float(Data[yNeg].shape[0]) / float(N))

    expected = p_prime 
    observed = p_obs 
    return chisquare(observed, expected)

#
# Returns attribute with maxInfoGain
#
def getMaxInfoGainFeature(remainingFeatures, Data):

    positive = Data[Data['Label'] == 1].shape[0]
    negative = Data[Data['Label'] == 0].shape[0]
    total = positive + negative
    
    rootEntropy = getEntropy(positive, negative, total)
    
    if rootEntropy == 0:
        # If root entropy is 0
        # print "WARNING: ROOT ENTROPY 0"
        if total == positive:
            return 'T'
        else:
            return 'F'

    featureEntropy = []
    for feature in remainingFeatures:
        pos, neg = calculateOccurences(Data, feature)
        infoGainSum = 0
        
        for i in range(len(pos)):
            cTotal = pos[i] + neg[i]
            if cTotal > 0:
                cEntropy = getEntropy(pos[i], neg[i], cTotal)
                infoGainSum += (float(cTotal)/float(total)) * float(cEntropy)
        infoGain = rootEntropy - infoGainSum
        featureEntropy.append(infoGain)
        
    maxGain = featureEntropy.index(max(featureEntropy))
    selectedFeature = remainingFeatures[maxGain]
    # if InfoGain is 0 return current optimal attribute
    if max(featureEntropy) == 0:
        return getMaxPossibleVal(Data)
    
    cVal = getChiSquareVal(positive, negative, total, Data, selectedFeature)
    
    
    if cVal.pvalue > pval:
        yPos = Data['Label'] == 1
        yNeg = Data['Label'] == 0
        pos = Data[yPos].shape[0]
        neg = Data[yNeg].shape[0]
        if pos > neg:
            return 'T'
        else:
            return 'F'
    del remainingFeatures[maxGain]
    return selectedFeature


# A random tree construction for illustration, do not use this in your code!
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

#
# Returns Optimal Value in current Data
#
def getMaxPossibleVal(Data):
    yPos = Data['Label'] == 1
    yNeg = Data['Label'] == 0
    pos = Data[yPos].shape[0]
    neg = Data[yNeg].shape[0]
    if pos > neg:
        return 'T'
    else:
        return 'F'

#
# Returns Optimal TreeNode in current Data
#
def getMaxPossibleNode(Data):
    yPos = Data['Label'] == 1
    yNeg = Data['Label'] == 0
    pos = Data[yPos].shape[0]
    neg = Data[yNeg].shape[0]
    if pos > neg:
        return TreeNode('T')
    else:
        return TreeNode('F')
#
# Creates Id3Tree recursively
#   
def createId3TreeRec(Data, fSet):
    positive = Data[Data['Label'] == 1].shape[0]
    negative = Data[Data['Label'] == 0].shape[0]
    total = Data['Label'].shape[0]
    
    # Initial conditions
    # Return if -
    #           1. All positive
    #           2. All negative
    #           3. No attribute remaining
    if positive == total:
        return TreeNode('T')
    elif negative == total:
        return TreeNode('F')
    elif len(fSet) == 0:
        return getMaxPossibleNode(Data)
    
    # Get optimal attribute
    nextRoot = getMaxInfoGainFeature(fSet, Data)
    
    # If leaf node then exit
    if nextRoot == 'T' or nextRoot == 'F':
        return TreeNode(nextRoot)
    
    root = TreeNode(nextRoot+1)
   
    for i in range(0, 5):
        featureWise = Data[nextRoot] == i+1
        nData = Data[featureWise]
        
        if nData.shape[0] > 0:
            root.nodes[i] = createId3TreeRec(nData, fSet)
        else:
            root.nodes[i] = getMaxPossibleNode(Data)
    return root

#
# Creates and returns ID3 Tree
#
def create_id3_tree(Xdata, Ydata):
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



parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = float(args['p'])
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)



# In[162]:


#print("Training...")
# s = create_random_tree(4)
XTrainDF = pd.DataFrame(Xtrain)
YTrainDF = pd.DataFrame(Ytrain, columns=['Label'])
s = create_id3_tree(XTrainDF, YTrainDF)

print 'Nodes', BFS(s)
# In[163]:


#print 'COUNTG', countG
# In[83]:


#s.data, s.nodes


# In[166]:


tree_name = "tree.pkl"
s.save_tree(tree_name)
def evaluate_datapoint(root,datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)

#Ytest_predict_name = "output1.csv"
Ypredict = []
#root = pkl.load(open('tree.pkl','r'))
root = s
for i in range(0,len(Xtest)):
    Ypredict.append([evaluate_datapoint(root,Xtest[i])])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

#print("Output files generated")


