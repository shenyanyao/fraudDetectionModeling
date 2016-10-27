import pandas as pd
import wordprocess_2nd as wp
# the feature related with card, 2nd is target
column_remain=[2, 10,19,25,26,27,29,31,32,33,36,37,44,45,49,56,57,58,63,64,65,66]

def Card_Func(words,column):
#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
            # initialize words_proc
    words_proc = []
    #column is the index list that is related with card
    for index,i in enumerate(column):
        methodToCall = getattr(wp, 'wordProc_'+str(i))
        words_proc.append(methodToCall(words[i])) 
    return words_proc
'''
from imp import reload
import pick_amount as pk
reload(pk)
#get the column of money
datapath = 'D:/CapitalOne/data/hashed/' 
card_money=pk.card_money([datapath+'train_fraud.txt',datapath+'train_nofraud1.txt',datapath+'train_nofraud2.txt',datapath+'train_nofraud3.txt',datapath+'train_nofraud4.txt',
    datapath+'train_nofraud5.txt',datapath+'train_nofraud6.txt',datapath+'train_nofraud7.txt',datapath+'train_nofraud8.txt',datapath+'train_nofraud9.txt'])
reload(pk)
interval=pk.sort_amount(card_money)

print(interval,len(card_money))
'''

import csv
from operator import itemgetter 
for i in range(1,2):
    write_csv='terminal_word2_'+str(i)+'.csv'
    f_out = open(write_csv,'w')
    with open('D:/CapitalOne/data/hashed/train_nofraud1.txt','r') as f:
        for line in f:
            words = line.split('|')
            new_line=Card_Func(words,column_remain)
            f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_fraud.txt','r') as f:
        for line in f:
            words = line.split('|')
            new_line=Card_Func(words,column_remain)
            f_out.write(','.join(new_line)+'\n')
    f_out.close()
             
