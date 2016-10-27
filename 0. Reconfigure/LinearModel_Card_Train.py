import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression,LogisticRegression
import random

def wordProc_X4(s):
    # ACCT_ACTVN_DT: the date when the first plastic on 
    # this account was activated.
    if pd.isnull(s):
        return 0
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
# the feature related with card, 2nd is target
column_remain=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,23,27,28,30,31,33,36,38,39,40,41,42,43,46,47,50,51,52,53,54,55,59,68]
def preprocess_Card(data_df):
	target 	= data_df['X3'].apply(lambda x: YN_to_int2(x))
	train 	= pd.DataFrame()
	train['X4_0'] = data_df['X4'].apply(lambda x: wordProc_X4(x))
	train['X5_0'] = data_df['X5'].apply(lambda x: float(x))
	train['X6_0'] = data_df['X6'].apply(lambda x: float(x))
	train['X7_0'] = data_df['X7'].apply(lambda x: float(x))
	train['X8_0'] = data_df['X8'].apply(lambda x: float(x))
	train['X9_0'] = data_df['X9'].apply(lambda x: 1 if x=='Y' else 0)
	train['X9_1'] = data_df['X9'].apply(lambda x: 1 if x=='N' else 0)
	train['X10_0'] = data_df['X10'].apply(lambda x: wordProc_X4(x))
	train['X11_0'] = data_df['X11'].apply(lambda x: string_to_int(x))

	train['X12_0'] = data_df['X12'].apply(lambda x: 1 if x=='B' else 0)
	train['X12_1'] = data_df['X12'].apply(lambda x: 1 if x=='E' else 0)


	train['X14_0'] = data_df['X14'].apply(lambda x: 1 if x=='5A' else 0)
	train['X14_1'] = data_df['X14'].apply(lambda x: 1 if x=='5X' else 0)
	train['X14_2'] = data_df['X14'].apply(lambda x: 1 if x=='5N' else 0)
	train['X14_3'] = data_df['X14'].apply(lambda x: 1 if x=='5W' else 0)
	train['X14_4'] = data_df['X14'].apply(lambda x: 1 if x=='5Z' else 0)
	train['X14_5'] = data_df['X14'].apply(lambda x: 1 if x=='5Y' else 0)

	train['X13_0'] = data_df['X13'].apply(lambda x:  1 if x=='X' else 0)
	train['X13_1'] = data_df['X13'].apply(lambda x:  1 if x=='4' else 0)

	train['X15_0'] = data_df['X15'].apply(lambda x: float(x))

	train['X19_0'] = data_df['X19'].apply(lambda x: float(x))
	train['X23_0'] = data_df['X23'].apply(lambda x: 1 if x == 'B' else 0)
	
	train['X30_0'] = data_df['X30'].apply(lambda x: 1 if x=='8' else 0)
	train['X30_1'] = data_df['X30'].apply(lambda x: 1 if x=='2' else 0)
	train['X30_2'] = data_df['X30'].apply(lambda x: 1 if x=='1' else 0)
	
	#train['X27_0'] = data_df['X27'].apply(lambda x: wordProc_X4(x))
	train['X28_0'] = data_df['X28'].apply(lambda x: time_to_int(x))
	train['X33_0'] = data_df['X33'].apply(lambda x: np.int32(x))
	train['X36_0'] = data_df['X36'].apply(lambda x: string_to_int(x)) # seems very important!
	train['X38_0'] = data_df['X38'].apply(lambda x: 0 if pd.isnull(x) else np.int32(x))
	train['X40_0'] = data_df['X40'].apply(lambda x: string_to_int(x))
	train['X50_0'] = data_df['X50'].apply(lambda x: YN_to_int(x))
	
	train['X31_0'] = data_df['X31'].apply(lambda x: float(x))
	train['X39_0'] = data_df['X39'].apply(lambda x: float(x))
	train['X41_0'] = data_df['X41'].apply(lambda x: float(x))
	train['X42_0'] = data_df['X42'].apply(lambda x: float(x))
	train['X43_0'] = data_df['X43'].apply(lambda x: float(x))

	train['X46_0'] = data_df['X46'].apply(lambda x: 1 if x=='840' else 0)
	train['X46_1'] = data_df['X46'].apply(lambda x: 1 if x=='250' else 0)

	train['X47_0'] = data_df['X47'].apply(lambda x: float(x))
	
	train['X51_0'] = data_df['X51'].apply(lambda x: wordProc_X4(x))
	train['X52_0'] = data_df['X52'].apply(lambda x: YN_to_int2(x))


	train['X53_0'] = data_df['X53'].apply(lambda x: wordProc_X4(x))


	train['X54_0'] = data_df['X54'].apply(lambda x: float(x))

	train['X55_0'] = data_df['X55'].apply(lambda x: 1 if x=='M' else 0)
	train['X55_1'] = data_df['X55'].apply(lambda x: 1 if x=='C' else 0)

	train['X59_0'] = data_df['X59'].apply(lambda x: YN_to_int2(x))
	train['X68_0'] = data_df['X68'].apply(lambda x: 1 if x=='8' else 0)
	train['X68_1'] = data_df['X68'].apply(lambda x: 1 if x=='2' else 0)
	train['X68_2'] = data_df['X68'].apply(lambda x: 1 if x=='1' else 0)

	return train,target



l = list([])
for i in range(1,69+1):
	s = 'X'+str(i)
	l.append(s)

for i in range(9,10):
    #current_card='card_'+str(i)
    filename_csv = 'D:/CapitalOne/data/money/train_'+str(i)+'.csv'
    n = sum(1 for line in open(filename_csv)) 
    print 'number of rows: '+str(n)
    s = 200000
    SEED = 18
    random.seed(SEED)
    skip = sorted(random.sample(xrange(n),n-s))
    data_pd = pd.read_csv(filename_csv,delimiter = '|',names = l, na_values =['null','none'], dtype = str, skiprows = skip)
    print "finish reading the training data"
    # get the target column and x columns
    train, target = preprocess_Card(data_pd)

    # train the current data using reg
    # fit and save
    scaler=StandardScaler()
    train_norm = scaler.fit_transform(train)
    train_mean = scaler.mean_
    train_std  = scaler.std_
    stat = pd.DataFrame()
    stat['mean'] = pd.Series(train_mean)
    stat['std'] = pd.Series(train_std)
    stat.to_csv('D:/CapitalOne/2. Card/card/stat_test'+str(i)+'.csv')
    print "finish normalizing the data"

    regressor = LogisticRegression(C=1)
    # regressor = LinearRegression()
    regressor.fit(train_norm,target)

    filename_csv_test = 'D:/CapitalOne/data/money/test_'+str(i)+'.csv'
    data_test_pd = pd.read_csv(filename_csv_test,delimiter = '|',names = l, na_values =['null','none'], dtype = str)
    print "finish reading the test data"
    test, target_test = preprocess_Card(data_test_pd)
    print "finish preprocessing the test data"
    test_norm = (test - train_mean) / train_std
    print "finish normalizing the test data"

    y_pred = regressor.predict_proba(test_norm)[:,1]
    y_test = target_test.values
    #y_pred = regressor.predict(train_norm) # This is for Linear regression
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    print auc
    
    path='D:/CapitalOne/2. Card/card/logistic_vtest_'+str(i)+'.model'
    joblib.dump(regressor, path)
    print 'finish training '+str(i)
#this store all the target line by line




