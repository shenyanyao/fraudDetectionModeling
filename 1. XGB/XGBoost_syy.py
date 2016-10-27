import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import (metrics, cross_validation, linear_model, preprocessing)   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import csv
from random import randint
import sys
import math
#sys.path.append('xgboost/wrapper/')


# XGBoost learning framework:


# loading data
print "loading data"
train_fraud = pd.read_csv('D:/CapitalOne/data/f1_data_clean.csv')
train_nfraud = pd.read_csv('D:/CapitalOne/data/ft1_data_clean.csv')
test = pd.read_csv('D:/CapitalOne/data/ftest_data_clean.csv')
train = train_fraud.append(train_nfraud)
l_test = test['label']
l_train = train['label']
test.drop('label', 1)
train.drop('label',1)
'''
test        = pd.read_csv('data/test.csv', index_col=0)
train       = pd.read_csv('data/train.csv') 
y           = train['ACTION']
train       = train.drop(['ACTION'], axis=1)
ID          = pd.read_csv('data/test.csv')['id']

X_all       = pd.concat([test, train], ignore_index=True)
test_rows   = len(test)
X_all       = X_all.drop(['ROLE_CODE'], axis=1)
'''

def preprocess_df(df):
    ll = list(df.columns.values)
    for str_i in ll:
        if str_i == 'ACTION' or str_i == 'id':
            continue
        dict_str = 'dict_'+str_i
        temp_dict = dict( df[str_i].value_counts() )
        df[dict_str] = df[str_i].apply( lambda x: temp_dict[x] )
    return df

X_train       = preprocess(train)
X_test       = preprocess(test)




# function modelfit: return test data predictions, training data predictions
# id:               number the input classifier.
# xgb_classifier:   the input classifier.
# dtrain:           training data (pandas dataframe)
# y:                training label (pandas dataframe)
# dtest:            test data (pandas dataframe)

def modelfit(id, xgb_classifier, dtrain, y , dtest,
            useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    print "fitting model "+str(id)
    if useTrainCV:
        xgb_param   = xgb_classifier.get_xgb_params()
        xgtrain     = xgb.DMatrix(dtrain.values, label=y.values)
        cvresult    = xgb.cv(   xgb_param, 
                                xgtrain, 
                                num_boost_round=xgb_classifier.get_params()['n_estimators'], 
                                nfold=cv_folds,
                                metrics=['auc'], 
                                early_stopping_rounds=early_stopping_rounds,
                                )
        xgb_classifier.set_params( n_estimators = cvresult.shape[0] )
    print "n_estimators : " + str(cvresult.shape[0])
    #Fit the algorithm on the data
    xgb_classifier.fit( dtrain, y, eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions  = xgb_classifier.predict(dtrain)
    dtrain_predprob     = xgb_classifier.predict_proba(dtrain)[:,1]
        
    #Print model report:
    accuracy            = metrics.accuracy_score(y.values, dtrain_predictions)
    auc                 = metrics.roc_auc_score(y, dtrain_predprob)
    print "\nModel Report"
    print "Accuracy : %.4g" % accuracy
    print "AUC Score (Train): %f" % auc
   
    dtest_predprob      = xgb_classifier.predict_proba(dtest)[:,1] 
    return dtest_predprob, dtrain_predprob, auc






seed1       = randint(0,100)
xgb1        = XGBClassifier(
 learning_rate = 0.1, n_estimators = 10, max_depth = 6, min_child_weight = 1,
 gamma=0,subsample = 0.9, colsample_bytree = 0.6, objective = 'binary:logistic', nthread = 4,
 scale_pos_weight = 1, seed = seed1
 )
[preds1, preds1_train, auc1]   = modelfit(1, xgb1, X_train, l_train, X_test)



auc = gl.evaluation.auc(preds1, l_test)
print auc
'''
roc = gl.evaluation.roc_curve(label_test, predictions)
fpr = roc['fpr']
tpr = roc['tpr']
precision = 1-fpr
recall = tpr

import matplotlib.pyplot as plt
def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision, recall, 'Precision recall curve (all)')
'''

