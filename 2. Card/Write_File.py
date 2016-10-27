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
interval =[[0.0, 1.0], [1.0, 2.1200000000000001], [2.1200000000000001, 6.3700000000000001], [6.3700000000000001, 10.01], [10.01, 16.010000000000002], [16.010000000000002, 23.91], [23.91, 35.0], [35.0, 52.920000000000002], [52.920000000000002, 100.0], [100.0, 15178142.5]]

len_card_money = 69859341

import csv
from operator import itemgetter 
for i in range(2,3):
    write_csv='card_word_'+str(i)+'.csv'
    start=interval[i-1][0]
    end=interval[i-1][-1]
    f_out = open(write_csv,'w')
    with open('D:/CapitalOne/data/hashed/train_nofraud2.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_nofraud3.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_nofraud4.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_nofraud5.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')   
    with open('D:/CapitalOne/data/hashed/train_nofraud6.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_nofraud7.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_nofraud8.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_nofraud9.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')           
    with open('D:/CapitalOne/data/hashed/train_nofraud1.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    with open('D:/CapitalOne/data/hashed/train_fraud.txt','r') as f:
        for line in f:
            words = line.split('|')
            if float(words[18])>start and float(words[18])<end:
                new_line=Card_Func(words,column_remain)
                f_out.write(','.join(new_line)+'\n')
    f_out.close()
             
