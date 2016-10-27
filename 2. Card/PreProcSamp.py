import wordprocess_2nd as wp
# this function calculate the elements for ath column
def myFunc2(a):
    # a is in the range [0, ..., 68]
    S = set()
    num_S = 0
    with open('sample.txt','r') as f:
        for line in f:
            words = line.split('|')
            if num_S <= 10000:
                # !!!!!!
                S.add(words[a])
                num_S = num_S + 1
            else:
                return S
    return S


def myFunc1():
    line_num = 0
    f_out = open('preprocess_sample.csv','w')

#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
    with open('sample.txt','r') as f:
        for line in f:
            line_num = line_num + 1
            # get vector 'words' from line (69 elements) 
            words = line.split('|')
            # initialize words_proc
            words_proc = ['null']*69
            # process for words[0]
            for i in range(0,69):
                methodToCall = getattr(wp, 'wordProc_'+str(i))
                words_proc[i] = methodToCall(words[i])
            # write the line to the file, end with '\n'(enter)
            f_out.write(','.join(words_proc)+'\n')
            # the following is just a test
    f_out.close()

def Card_Func(line,column):
#    with open('/Users/apple/Google Drive/CapitalOne_Team/Sample_200MB/sample.txt','r') as f:
            # initialize words_proc
    words_proc = []
    #column is the index list that is related with card
    for i in column:
        methodToCall = getattr(wp, 'wordProc_'+str(i))
        words_proc.append(methodToCall(words[i])) 
    return words_proc
#myFunc1()
#S = myFunc2(38)
#print (S)
#print wp.wordProc_66('001')