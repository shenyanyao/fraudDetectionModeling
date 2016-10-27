import pandas as pd
# the feature related with card, 2nd is target
# column_remain=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,23,27,28,30,31,33,36,38,39,40,41,42,43,46,47,50,51,52,53,54,55,59,68]

interval =[[-1.0, 1.0], [1.0, 2.1200000000000001], [2.1200000000000001, 6.3700000000000001], [6.3700000000000001, 10.01], [10.01, 16.010000000000002], [16.010000000000002, 23.91], [23.91, 35.0], [35.0, 52.920000000000002], [52.920000000000002, 100.0], [100.0, 15178142.5]]

for i in range(0,10):
    write_csv='D:/CapitalOne/data/money/train_'+str(i)+'.csv'
    start, end   =   interval[i]
    f_out   = open(write_csv,'w')
    path    = 'D:/CapitalOne/data/hashed/'
    for j in range(1,10):
        file_read = path + 'train_nofraud'+str(j)+'.txt'
        with open(file_read , 'r') as f:
            for line in f:
                elements = line.split('|')
                elements_th = float(elements[18])
                elements_rule = int(elements[0])
                R = elements_rule %2 
                if i < 2 and elements_th == 1:
                    #print elements[0]
                    if (R == 0 and i == 0) or (R == 1 and i == 1) :
                        f_out.write(line)
                        continue
                else:
                    if elements_th > start and elements_th <= end:
                        f_out.write(line)  
        print '    finish reading: '+str(j)  
    with open('D:/CapitalOne/data/hashed/train_fraud.txt','r') as f:
        for line in f:
            elements = line.split('|')
            elements_th = float(elements[18])
            elements_rule = int(elements[0])
            R = elements_rule %2
            if i < 2 and elements_th == 1:
                if (R == 0 and i == 0) or (R == 1 and i == 1) :
                    f_out.write(line)
                    continue
            else:
                if elements_th > start and elements_th <= end:
                    f_out.write(line) 
    print '    finish reading: fraud'
    f_out.close()
    print 'finish writing: '+write_csv
             
