
# coding: utf-8

# In[1]:


import random
import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd
from math import log
from scipy.stats import chisquare


# In[137]:



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


# In[161]:



def getEntropy(positive, negative, total):
    posRatio = (float(positive) / float(total))
    negRatio = (float(negative) / float(total))
    entropy = 0.0
    if posRatio != 0:
        entropy = -posRatio * log(posRatio, 2)
    if negRatio != 0:
        entropy -= negRatio * log(negRatio, 2)
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

    expected = p_prime #p_prime + n_prime
    observed = p_obs #+ n_obs
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
        print "WARNING: ROOT ENTROPY 0"
        #remainingFeatures.remove()
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
        
        for i in range(len(pos)):
            cTotal = pos[i] + neg[i]
            if cTotal > 0:
                cEntropy = getEntropy(pos[i], neg[i], cTotal)
                #infoGainSum += float(cEntropy)
                infoGainSum += (float(cTotal)/float(total)) * float(cEntropy)
        infoGain = rootEntropy - infoGainSum
        featureEntropy.append(infoGain)
        #featureEntropy.append(infoGainSum)
    #print "Feature Entropy", featureEntropy
    #maxGain = featureEntropy.index(max(featureEntropy))
    maxGain = featureEntropy.index(max(featureEntropy))
    selectedFeature = remainingFeatures[maxGain]
    #print "Max: ", max(featureEntropy)
    # if InfoGain is 0
    if max(featureEntropy) == 0:
        return getMaxPossibleVal(Data)
    
    cVal = getChiSquareVal(positive, negative, total, Data, selectedFeature)
    #print "CVAL: ", cVal.pvalue
    
    if cVal.pvalue > pval:
        yPos = Data['Label'] == 1
        yNeg = Data['Label'] == 0
        #xPos = Data[selectedFeature] == i+1
        #pos[i] = Data[xPos & yPos].shape[0]
        #neg[i] = Data[yNeg & xPos].shape[0]
        pos = Data[yPos].shape[0]
        neg = Data[yNeg].shape[0]
        if pos > neg:
            return 'T'
        else:
            return 'F'
    del remainingFeatures[maxGain]
    return selectedFeature


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
def getMaxPossibleVal(Data):
    yPos = Data['Label'] == 1
    yNeg = Data['Label'] == 0
    pos = Data[yPos].shape[0]
    neg = Data[yNeg].shape[0]
    if pos > neg:
        return 'T'
    else:
        return 'F'


def getMaxPossibleNode(Data):
    yPos = Data['Label'] == 1
    yNeg = Data['Label'] == 0
    pos = Data[yPos].shape[0]
    neg = Data[yNeg].shape[0]
    if pos > neg:
        return TreeNode('T')
    else:
        return TreeNode('F')
    
def createId3TreeRec(Data, fSet):
    positive = Data[Data['Label'] == 1].shape[0]
    negative = Data[Data['Label'] == 0].shape[0]
    total = Data['Label'].shape[0]
    
    if positive == total:
        return TreeNode('T')
    elif negative == total:
        return TreeNode('F')
    elif len(fSet) == 0:
        return getMaxPossibleNode(Data)
    
    nextRoot = getMaxInfoGainFeature(fSet, Data)
    #print "Next Root", nextRoot 
    #print "Data ", Data.shape[0]
    #print Data[:5]
    if nextRoot == 'T' or nextRoot == 'F':
        return TreeNode(nextRoot)
    root = TreeNode(nextRoot+1)
    #fSet.remove(nextRoot)    
    for i in range(0, 5):
        featureWise = Data[nextRoot] == i+1
        nData = Data[featureWise]
        #nData1 = nData.drop(columns=[featureWise])
        if nData.shape[0] > 0:
            root.nodes[i] = createId3TreeRec(nData, fSet)
        else:
            root.nodes[i] = getMaxPossibleNode(Data)
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
    count = 0
    queue = list()
    
    queue.append(root)
    
    while len(queue)>0:
        n = len(queue)
        r = list()
        for i in range(n):
            node = queue[0]
            count += 1
            queue.remove(node)
            if node is not None:
                r.append(node.data)
                for children in node.nodes:
                    if children !=-1 :
                        queue.append(children)
        
        print (r)
    return count

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


# In[142]:


#featureSet = list(range(0,num_feats))
#print featureSet


# In[143]:


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
'''
Xtrain_name = "train.csv"
Ytrain_name = "train_label.csv"
Xtest_name = "test.csv"
Ytest_predict_name = "output.csv"
pval = 0.05
Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)



# In[162]:


print("Training...")
# s = create_random_tree(4)
XTrainDF = pd.DataFrame(Xtrain)
YTrainDF = pd.DataFrame(Ytrain, columns=['Label'])
s = create_id3_tree(XTrainDF, YTrainDF)




# In[163]:


BFS(s)


# In[83]:


s.data, s.nodes


# In[166]:


tree_name = "tree.pkl"
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
Ytest_predict_name


# In[170]:


Ytest_predict_name
Xtest


# In[169]:


def evaluate_datapoint(root,datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)

Ytest_predict_name = "output.csv"
Ytree = []
#root = pkl.load(open('tree.pkl','r'))
root = s
for i in range(0,len(Xtest)):
    Ytree.append(evaluate_datapoint(root,Xtest[i]))

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")

