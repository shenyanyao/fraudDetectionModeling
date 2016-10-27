import numpy as np
import pandas as pd
import xgboost as xgb 
from xgboost.sklearn import XGBClassifier
from sklearn import (metrics, cross_validation, linear_model, preprocessing)   #Additional scklearn functions
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import csv
from random import randint
import sys
import math
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

l = list([])
for i in range(1,69+1):
	s = 'X'+str(i)
	l.append(s)


def string_to_int(string):
    if pd.isnull(s):
        return float(0)
    string = str(string)
    a = [ord(i) for i in string]
    return float(sum(a))
def time_to_int(s):
    hour  = float(s[0:2])
    minute = float(s[3:5])
    milliseconds   = float(s[6:])
    value = hour*3600+minute*60+milliseconds
    return value
def date_to_int(s):
    year  = float(s[0:4])
    month = float(s[5:7])
    day   = float(s[8:10])
    value = (year - 2005)*365 + (month-1)*365/12 + day
    return value
def YN_to_int(s):
    if s=='Y':
        return 1
    if s=='N':
        return -1
    if pd.isnull(s):
        return 0
def YN_to_int2(s):
    if s=='Y':
        return 1
    if s=='N':
        return 0
    if pd.isnull(s):
        return 0
def num_to_int(s):
    if pd.isnull(s):
        return -1
    return np.int32(s)

def preprocess_df(data_frame):
    # 19,25,26,27,31,32,33,36,37,39,49,56,57,63
    data = pd.DataFrame()
    
    
    data['Money'] = data_frame['X24'].apply(lambda x: float(x)) # good
    data['Money_Daily'] = data_frame['X31'].apply(lambda x: float(x))
    data['Money_Ratio'] = data['Money']/data['Money_Daily']
    data['Money_RMB'] = data_frame['X39'].apply(lambda x: float(x))
    data['Money_Balance'] = data_frame['X8'].apply(lambda x: float(x)) 
    data['Money_Available'] = data_frame['X5'].apply(lambda x: float(x)) 
    data['Address_Match'] = data_frame['X14'].apply(lambda x: string_to_int(x)) 
    data['Auth_Number_Cash'] = data_frame['X16'].apply(lambda x: float(x)) 
    data['Money_required'] = data_frame['X19'].apply(lambda x: float(x))
    data['Auth_TypeCD'] = data_frame['X29'].apply(lambda x: string_to_int(x)) 
    data['Card_Activate'] = data_frame['X52'].apply(lambda x: string_to_int(x)) 
    data['Card_Current_Indi'] = data_frame['X55'].apply(lambda x: string_to_int(x)) 

    data['Auth_Type'] = data_frame['X20'].apply( lambda x: num_to_int(x) )
    
    data['Auth_Request_CD'] = data_frame['X26'].apply(lambda x: num_to_int(np.int32(x)))
    data['Auth_Date'] = data_frame['X27'].apply(lambda x: date_to_int(x))
    data['Auth_Time'] = data_frame['X28'].apply(lambda x: time_to_int(x))
    
    data['CVV_Valid'] = data_frame['X32'].apply(lambda x: string_to_int(x))
    data['CVV_Duration_Time'] = data_frame['X33'].apply(lambda x: np.int32(x))
    data['CVV_Mismatch_CD'] = data_frame['X34'].apply(lambda x: string_to_int(x))
    data['CVI2_Auth_CD'] = data_frame['X35'].apply(lambda x: string_to_int(x)) 
    data['CVV_Valid2'] = data_frame['X36'].apply(lambda x: string_to_int(x))
    data['Terminal_Security'] = data_frame['X37'].apply(lambda x: string_to_int(x))
    data['Present'] = data_frame['X38'].apply(lambda x: 0 if pd.isnull(x) else np.int32(x))
    data['CVV_Mismatch_CD'] = data_frame['X40'].apply(lambda x: string_to_int(x))
    
    data['PIN_Needed'] = data_frame['X50'].apply(lambda x: YN_to_int(x))
    data['POS_Method'] = data_frame['X57'].apply(lambda x: string_to_int(x))
    data['Multi_Auth'] = data_frame['X58'].apply(lambda x: string_to_int(x))
    data['Terminal_Ability'] = data_frame['X64'].apply(lambda x: num_to_int(x))
    
    data['label'] = data_frame['X3'].apply(lambda x: YN_to_int2(x))
    return data


datapath = 'D:/CapitalOne/data/hashed/' 
fraud_filename 		= datapath+'train_fraud.txt'

test_filename 		= datapath+'test_data.txt'

fraud_df 			= pd.read_csv(fraud_filename, delimiter = '|',header = None, names = l, na_values =['null','none'], dtype = str, nrows = None)
print 'fraud data is read'

test_df 			= pd.read_csv(test_filename, delimiter = '|',header = None, names = l, na_values =['null','none'], dtype = str, nrows = 200000)
print 'test data is read'

train_fraud = preprocess_df(fraud_df)

test = preprocess_df(test_df)
del fraud_df
del test_df

train_nofraud_all = pd.DataFrame()
for i in range(9,10):
	nofraud_filename 	= datapath+'train_nofraud'+str(i)+'.txt'
	nofraud_df 			= pd.read_csv(nofraud_filename, delimiter = '|',header = None, names = l, na_values =['null','none'], dtype = str, skiprows = 800000*(i-1), nrows = 2000000)
	print 'no fraud data '+str(i)+'th is read'
	train_nofraud = preprocess_df(nofraud_df)
	train_nofraud_all = train_nofraud_all.append(train_nofraud)
	del nofraud_df


print 'finish preprocessing'
train = train_fraud.append(train_nofraud_all)
train.to_csv('simple_test.csv')
haha
l_test = test['label']
l_train = train['label']
w = l_train.apply(lambda x: x+0.1)
#print l_test.sum(axis=0)

test = test.drop('label', 1)
train = train.drop('label',1)




#print test
X_train       	= train
X_test       	= test
del train
del test
print 'finish postprocessing'



'''
seed1       = randint(0,100)
xgb1        = XGBClassifier(
 learning_rate = 0.1, n_estimators = 100, max_depth = 6, min_child_weight = 1,
 gamma=0,subsample = 0.9, colsample_bytree = 0.6, objective = 'binary:logistic', nthread = 16,
 scale_pos_weight = 1, seed = seed1
 )
[preds1, preds1_train, auc1]   = modelfit(1, xgb1, X_train, l_train, X_test, useTrainCV = False)
'''
D_train = xgb.DMatrix(X_train.values, label = l_train.values ,weight=w)
param = [('max_depth', 6), ('objective', 'binary:logistic'), ('bst:eta', 0.1),  
	('eval_metric', 'auc'), ('subsample', 0.9), ('colsample_bytree', 0.6), ('nthread', 16),('early_stopping_rounds',50)]
#watchlist  = [(D_test,'eval'), (D_train,'train')]
num_round = 500
evals_result = {}

bst = xgb.train( param, D_train, num_round, verbose_eval = True)
bst.save_model('XGB_9.model')


D_test = xgb.DMatrix(X_test.values)
preds1 = bst.predict(D_test)
print preds1


#print preds1_train
#print auc1
print sum(l_test.values)



fpr, tpr, thresholds = metrics.roc_curve(l_test.values, preds1)
auc = metrics.auc(fpr, tpr)
print auc

import matplotlib.pyplot as plt

plt.plot(fpr,tpr)
plt.show()
#print ConfusionMatrix(l_test.values, preds1)
#print tpr