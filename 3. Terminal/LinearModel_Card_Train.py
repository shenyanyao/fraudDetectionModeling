import pandas as pd
import wordprocess_2nd as wp
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

def Card_Func_df(words_df,column):
#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
            # initialize words_proc
    words_proc = []
    #column is the index list that is related with card
    for i in column:
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
from sklearn.linear_model import LinearRegression,LogisticRegression
reg_list=[]
for i in range(4,11):
    current_card='card_'+str(i)
    current_csv='card_word_'+str(i)+'.csv'
    current_card=pd.read_csv(current_csv,header=None)
    print "finish reading the data"
    # get the target column and x columns 
    target=current_card[0]
    current_card.drop(0,axis=1,inplace=True)
    # train the current data using linear reg
    # fit and save
    regressor = LogisticRegression()
    s= Models(regressor)
    s.fit(current_card,target)

    regressor.fit(current_card,target)
    path='D:/CapitalOne/Card/card/logistic_'+str(i)+'.model'
    s.savemod(path)
    reg_list.append(regressor)
    print 'finish training'+str(i)
#this store all the target line by line




