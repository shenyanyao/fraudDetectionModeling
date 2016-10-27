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
import matplotlib.pyplot as plt
from sklearn.externals import joblib
path = 'D:/CapitalOne/1. XGB/'
modelfile1 = path + 'XGB_model/XGB_1.model'
modelfile2 = path + 'XGB_model/XGB_2.model'
modelfile3 = path + 'XGB_model/XGB_3.model'
modelfile4 = path + 'XGB_model/XGB_4.model'
modelfile5 = path + 'XGB_model/XGB_5.model'
modelfile6 = path + 'XGB_model/XGB_6.model'
modelfile7 = path + 'XGB_model/XGB_7.model'
modelfile8 = path + 'XGB_model/XGB_8.model'
modelfile9 = path + 'XGB_model/XGB_9.model'
bst1 = xgb.Booster({'nthread':16}, model_file = modelfile1)
bst2 = xgb.Booster({'nthread':16}, model_file = modelfile2)
bst3 = xgb.Booster({'nthread':16}, model_file = modelfile3)
bst4 = xgb.Booster({'nthread':16}, model_file = modelfile4)
bst5 = xgb.Booster({'nthread':16}, model_file = modelfile5)
bst6 = xgb.Booster({'nthread':16}, model_file = modelfile6)
bst7 = xgb.Booster({'nthread':16}, model_file = modelfile7)
bst8 = xgb.Booster({'nthread':16}, model_file = modelfile8)
bst9 = xgb.Booster({'nthread':16}, model_file = modelfile9)
'''
t1 = pd.Series(bst1.get_fscore())
print t1
xgb.plot_tree(bst1, num_trees=2)
#xgb.to_graphviz(bst1, num_trees=1)
plt.show()

featp = t1.sort_values()[-15:].plot(kind='bar', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.show()
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
'''
i = 1
path = 'D:/CapitalOne/2. Card/card/logistic_v1_'+str(i)+'.model'
LR = joblib.load(path)
print LR.coef_