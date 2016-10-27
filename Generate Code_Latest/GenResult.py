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
from imp import reload
import pick_amount as pk
import csv
from operator import itemgetter 
from sklearn.linear_model import LinearRegression
import wordprocess_2nd as wp
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
def Card_Func(words,column):
#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
            # initialize words_proc
    words_proc = []
    #column is the index list that is related with card
    for index,i in enumerate(column):
        methodToCall = getattr(wp, 'wordProc_'+str(i))
        words_proc.append(methodToCall(words[i])) 
    return words_proc

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

def XGB_Model(test_filename):
	test_df 			= pd.read_csv(test_filename, delimiter = '|',header = None, names = l, na_values =['null','none'], dtype = str, nrows = None)
	print "finish reading test data"
	test = preprocess_df(test_df)
	print "finish preprocess test data"
	l_test = test['label']
	test = test.drop('label', 1)
	X_test       	= test
	D_test = xgb.DMatrix(X_test.values)



	modelfile1 = 'models/XGB_1.model'
	modelfile2 = 'models/XGB_2.model'
	modelfile3 = 'models/XGB_3.model'
	modelfile4 = 'models/XGB_4.model'
	modelfile5 = 'models/XGB_5.model'
	modelfile6 = 'models/XGB_6.model'
	modelfile7 = 'models/XGB_7.model'
	modelfile8 = 'models/XGB_8.model'
	modelfile9 = 'models/XGB_9.model'
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


	return pd.DataFrame(preds) 
def LinearModel_Card(test_data_filename):
	column_remain=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,23,27,28,30,31,33,36,38,39,40,41,42,43,46,47,50,51,52,53,54,55,59,68]
	interval =[[0.0, 1.0], [1.0, 2.1200000000000001], [2.1200000000000001, 6.3700000000000001], [6.3700000000000001, 10.01], [10.01, 16.010000000000002], [16.010000000000002, 23.91], [23.91, 35.0], [35.0, 52.920000000000002], [52.920000000000002, 100.0], [100.0, 15178142.5]]
	reg_list=[]
	for i in range(1,11):
	    s= Models()
	    path='models/linear_'+str(i)+'.model'
	    s.loadmod(path, True)
	    reg_list.append(s)

	#this store all the target line by line

	target_list=[]
	# this is pred y line by line
	y_pred=[]
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
	#print(y_pred)
	y_test = target_list
	return pd.DataFrame(y_pred)

	
def LinearModel_Terminal(test_data_filename):
	column_remain=[2,10,19,25,26,27,29,31,32,33,36,37,44,45,49,56,57,58,63,64,65,66]
	z = Models()
	path='models/linear2_1.model'
	z.loadmod(path, True)
	
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
	return pd.DataFrame(y_pred) 

# please edit the filename here
# this prediction may take a while
filename = 'validation_test.txt' 
l = list([])
for i in range(1,69+1):
	s = 'X'+str(i)
	l.append(s)
f1_df = pd.read_csv(filename, delimiter = '|',header = None, names = l, na_values =['null','none'], dtype = str, nrows = None)
ID = f1_df['X1']

pred1 = XGB_Model(filename)
pred2 = LinearModel_Card(filename)
pred3 = LinearModel_Terminal(filename)

pred1 = ( pred1[0] - pred1[0].min() ) / (pred1[0].max() - pred1[0].min())
pred2 = ( pred2[0] - pred2[0].min() ) / (pred2[0].max() - pred2[0].min())
pred3 = ( pred3[0] - pred3[0].min() ) / (pred3[0].max() - pred3[0].min())
P_FRAUD =  0.4*pred1 + 0.3*pred2+ 0.3*pred3
#print P_FRAUD
DECLINE = P_FRAUD.apply(lambda x: 'Y' if float(x)>0.49 else 'N' )

with open('output.csv', 'w') as fp:
    a       = csv.writer(fp, delimiter=',')
    data    = zip(ID.values, P_FRAUD.values, DECLINE.values)
    a.writerow(["AUTH_ID","P_FRAUD","DECLINE"])
    a.writerows(data)
    fp.close()