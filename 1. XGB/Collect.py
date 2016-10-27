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
def postprocess_df(df):
    ll = list(df.columns.values)
    '''
    for str_i in ll:
        if str_i == 'ACTION' or str_i == 'id' or str_i == 'label':
            continue
        dict_str = 'dict_'+str_i
        temp_dict = dict( df[str_i].value_counts() )
        df[dict_str] = df[str_i].apply( lambda x: temp_dict[x] )
    '''
    return df
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
test_filename 		= 'D:/CapitalOne/data/validation_part_02_of_02.txt'
test_df 			= pd.read_csv(test_filename, delimiter = '|',header = None, names = l, na_values =['null','none'], dtype = str, nrows = None)
print "finish reading test data"
test = preprocess_df(test_df)
print "finish preprocess test data"
l_test = test['label']
test = test.drop('label', 1)
X_test       	= test
D_test = xgb.DMatrix(X_test.values)



modelfile1 = 'XGB_model/XGB_1.model'
modelfile2 = 'XGB_model/XGB_2.model'
modelfile3 = 'XGB_model/XGB_3.model'
modelfile4 = 'XGB_model/XGB_4.model'
modelfile5 = 'XGB_model/XGB_5.model'
modelfile6 = 'XGB_model/XGB_6.model'
modelfile7 = 'XGB_model/XGB_7.model'
modelfile8 = 'XGB_model/XGB_8.model'
modelfile9 = 'XGB_model/XGB_9.model'
bst1 = xgb.Booster({'nthread':16}, model_file = modelfile1)
bst2 = xgb.Booster({'nthread':16}, model_file = modelfile2)
bst3 = xgb.Booster({'nthread':16}, model_file = modelfile3)
bst4 = xgb.Booster({'nthread':16}, model_file = modelfile4)
bst5 = xgb.Booster({'nthread':16}, model_file = modelfile5)
bst6 = xgb.Booster({'nthread':16}, model_file = modelfile6)
bst7 = xgb.Booster({'nthread':16}, model_file = modelfile7)
bst8 = xgb.Booster({'nthread':16}, model_file = modelfile8)
bst9 = xgb.Booster({'nthread':16}, model_file = modelfile9)


preds1 =  bst1.predict(D_test)
preds2 =  bst2.predict(D_test)
preds3 =  bst3.predict(D_test)
preds4 =  bst4.predict(D_test)
preds5 =  bst5.predict(D_test)
preds6 =  bst6.predict(D_test)
preds7 =  bst7.predict(D_test)
preds8 =  bst8.predict(D_test)
preds9 =  bst9.predict(D_test)



preds = (preds1 + preds2+ preds3+ preds4+ preds5+ preds6+ preds7+ preds8 +preds9)/9
fpr, tpr, thresholds = metrics.roc_curve(l_test.values, preds)
auc = metrics.auc(fpr, tpr)
print auc



import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.show()

with open('result_validation_2_xgb.csv', 'w') as fp:
    a       = csv.writer(fp, delimiter=',')
    data    = zip(l_test.values, preds)
    a.writerow(["True", "Predict"])
    a.writerows(data)
    fp.close()