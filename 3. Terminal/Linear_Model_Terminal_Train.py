import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
import sklearn.preprocessing as skprp
import sklearn.metrics as skmetric
import matplotlib.pyplot as mtpt
column_remain=[2,10,19,25,26,27,29,31,32,33,36,37,44,45,49,56,57,58,63,64,65,66]
null_set=[4,9,21,22,25,32,36,38,40,41,46,47,48,50,51]

def Card_Func(words,column):
#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
            # initialize words_proc
    words_proc = []
    #column is the index list that is related with card
    for index,i in enumerate(column):
        methodToCall = getattr(wp, 'wordProc_'+str(i))
        words_proc.append(methodToCall(words[i])) 
    return words_proc
from imp import reload
import csv
from operator import itemgetter 

# read and prepare data
'''
d=pd.read_csv('/home/brent/brent.zhang@utexas.edu/CapitalOne_Team/CW_Zhang/train_all.csv',header=None)
data=d.dropna(axis=1)
target=data[0]
nm_data=data.drop(0,axis=1)
nm_data.fillna(0)
#descr=data.describe()
del d
test=pd.read_csv('/home/brent/brent.zhang@utexas.edu/CapitalOne_Team/CW_Zhang/test_partial.csv',header=None)
testdata=test.dropna(axis=1)
testy=testdata[0]
testx=testdata.drop(0,axis=1)
testx.fillna(0)
del test
'''

#develop a model class
from sklearn.externals import joblib
class Models:
    def __init__(self,mod = None):
        # model
        self.mod = mod
        self.load= None
    def fit(self,x,y):
        self.mod.fit(x,y)
    def predict(self,testx):
        self.ypred=self.mod.predict(testx)
        return self.ypred
    def perform(self): 
        # performance
        self.fpr,self.tpr,self.thresholds=skmetric.roc_curve(testy,self.ypred)
        self.auc=skmetric.roc_auc_score(testy,self.ypred)
        self.fig=mtpt.plot(self.fpr,self.tpr)
        mtpt.show()
        return self.auc
    def savemod(self,path):
        joblib.dump(self.mod, path)
    def loadmod(self,path,modify= True):
        self.load=joblib.load(path)
        if modify:
            self.mod = self.load
        return self.load



reg_list=[]
for i in range(1,2):
    current_csv='terminal_word2_'+str(i)+'.csv'
    current_card=pd.read_csv(current_csv, header=None)
    print "finish reading the data"
    # get the target column and x columns 
    target=current_card[0]
    current_card.drop(0,axis=1,inplace=True)
    # train the current data using linear reg
    # fit and save
    regressor = LinearRegression()
    s= Models(regressor)
    s.fit(current_card,target)

    regressor.fit(current_card,target)
    path='D:/CapitalOne/Terminal/terminal/linear2_'+str(i)+'.model'
    s.savemod(path)
    reg_list.append(regressor)
    print 'finish training'+str(i)



