def myFunc2():
    import csv
    pay_amount=[]
    # wirte only the transcation amount in this csv
    with open("amount.csv", "w") as f_out:
#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
        for i in range(1,10):
        # iterate nine train set
            temp='/Users/weiyansun/Downloads/Capital/training/training_part_0'+str(i)+'_of_10.txt'
            with open(temp,'r',encoding='utf-8') as f:
                for line in f:
            # get vector 'words' from line (69 elements) 
                    words = line.split('|')
            # get the money amount and add to list
                    pay_amount.append(float(words[18]))
            # write the line to the file, end with '\n'(enter)
        writer = csv.writer(f_out)
        writer.writerow(pay_amount)
    return pay_amount
def sort_amount(amount):
    import numpy as np
    import math
    sort_amount=np.sort(amount)
    lengh=len(sort_amount)
    divide_num=math.floor(lengh/10)
    divide_nums=[[sort_amount[i*divide_num],sort_amount[(i+1)*divide_num]]
    for i in range(0,9)]
    # make max plus a large num in order to avoid test data has large num
    divide_nums.append([sort_amount[9*divide_num],sort_amount.max()+10000000])
    return divide_nums
def card_money(direction_list):
    import csv
    pay_amount=[]
    # wirte only the transcation amount in this csv
    with open("amount.csv", "w") as f_out:
#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
        for direction in direction_list:
            with open(direction,'r') as f:
                for line in f:
            # get vector 'words' from line (69 elements) 
                    words = line.split('|')
            # get the money amount and add to list
                    pay_amount.append(float(words[18]))
            # write the line to the file, end with '\n'(enter)
        writer = csv.writer(f_out)
        writer.writerow(pay_amount)
    return pay_amount

#pay_amount=myFunc2()
