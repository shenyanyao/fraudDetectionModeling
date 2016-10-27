import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as skprp
import sklearn.metrics as skmetric
import matplotlib.pyplot as mtpt
import wordprocess_2nd as wp
from sklearn import (metrics, cross_validation, linear_model, preprocessing)   #Additional scklearn functions
from sklearn.metrics import roc_auc_score
import csv
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
    def predict_proba(self,testx):
        self.ypred=self.mod.predict_proba(testx)
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


# load and estimate
z = Models()
path='D:/CapitalOne/Terminal/terminal/linear2_1.model'
z.loadmod(path, True)

datapath    = 'D:/CapitalOne/data/'
test_data_filename = datapath + 'validation_part_02_of_02.txt'

i=1
target_list = []
y_pred = []
with open(test_data_filename,'r') as f:
    for line in f:
        words           = line.split('|')
        # fine this money is in which interval
        use_model   = z
        new_line    =Card_Func(words,column_remain)
        # change the | by , and can use it as a list
        new_line    = ','.join(new_line)
        new_line    = new_line.split(',')
        target_list.append(int( new_line[0] ))
        x_test      = new_line[1:]
        test_array  = np.asarray(x_test)
        test_array  = test_array.astype('float64')
        y           = use_model.predict(test_array)
        y           = y[0]
        y_pred.append(y)
        #print y
        if i % 100000 == 0:
            print i
        i = i+1

#print(y_pred)

y_test = target_list
#print y_test

d={'y_pred':y_pred,'y_test':y_test}
compare=pd.DataFrame(d)
# auc result

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)
print auc

import matplotlib.pyplot as plt

plt.plot(fpr,tpr)
plt.show()

with open('result_validation2.csv', 'w') as fp:
    a       = csv.writer(fp, delimiter=',')
    data    = zip(y_test, y_pred)
    a.writerow(["True", "Predict"])
    a.writerows(data)
    fp.close()