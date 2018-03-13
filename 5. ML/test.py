import random
import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd
from math import log
from scipy.stats import chisquare

class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)


def evaluate_datapoint(root,datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)

Ytest_predict_name = "output.csv"
Ypredict = []
root = pkl.load(open('tree.pkl','r'))
#root = s
for i in range(0,len(Xtest)):
    Ypredict.append(evaluate_datapoint(root,Xtest[i]))

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")
