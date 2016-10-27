import pandas as pd
import wordprocess_2nd as wp
from sklearn import (metrics, cross_validation, linear_model, preprocessing)   #Additional scklearn functions
from sklearn.metrics import roc_auc_score
# the feature related with card, 2nd is target
column_remain=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,23,27,28,30,31,33,36,38,39,40,41,42,43,46,47,50,51,52,53,54,55,59,68]
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
import pick_amount as pk
import csv
from operator import itemgetter 
interval =[[0.0, 1.0], [1.0, 2.1200000000000001], [2.1200000000000001, 6.3700000000000001], [6.3700000000000001, 10.01], [10.01, 16.010000000000002], [16.010000000000002, 23.91], [23.91, 35.0], [35.0, 52.920000000000002], [52.920000000000002, 100.0], [100.0, 15178142.5]]


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

import pandas as pd
from sklearn.linear_model import LinearRegression

reg_list=[]
for i in range(1,11):
    s= Models()
    path='D:/CapitalOne/2. Card/card/linear_'+str(i)+'.model'
    s.loadmod(path, True)
    reg_list.append(s)

#this store all the target line by line
import numpy as np

target_list=[]
# this is pred y line by line
y_pred=[]

test_data_filename = 'D:/CapitalOne/data/hashed/test_data.txt'
# test_data_filename = 'D:/CapitalOne/data/validation_part_02_of_02.txt'


#test_data = pd.read_csv(test_data_filename, header = None, delimiter = '|',na_values =['null','none'],   nrows = 1)
#print test_data

i=1
with open(test_data_filename,'r') as f:
    for line in f:
        words           = line.split('|')
        current_money   = float(words[18])
        # fine this money is in which interval
        for index,inter in enumerate(interval):
            start   = inter[0]
            end     = inter[-1]
            if end > current_money >= start:
                break
        use_model   =reg_list[index]
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
        if i % 100000 == 0:
            print i
        i = i+1
        if i > 1000000:
            break

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

with open('result_test_linear_n.csv', 'w') as fp:
    a       = csv.writer(fp, delimiter=',')
    data    = zip(y_test, y_pred)
    a.writerow(["True", "Predict"])
    a.writerows(data)
    fp.close()



