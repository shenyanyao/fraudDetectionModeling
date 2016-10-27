import wordprocess_2nd as wp
#import graphlab
# this function calculate the elements for ath column
def myFunc2(a):
    # a is in the range [0, ..., 68]
    S = set()
    num_S = 0
    with open('/media/brent/handsome/Capitalonedata/training/training_part_01_of_10.txt','r') as f:
        for line in f:
            words = line.split('|')
            if num_S <= 100:
                S.add(words[a])
                num_S = num_S + 1
            else:
                return S
    return S
    

def myFunc_termina():
    line_num = 0
    f_out = open('/home/brent/brent.zhang@utexas.edu/CapitalOne_Team/CW_Zhang/train_all.csv','a')
    termina_set=[2,10,19,25,26,27,29,31,32,33,36,37,44,45,49,56,57,58,63,64,65,66]
    null_set=[4,9,21,22,25,32,36,38,40,41,46,47,48,50,51]
    with open('/home/brent/brent.zhang@utexas.edu/CapitalOne_Team/Hashed Data/train_nofraud1.txt','r') as f:
    #with open('/home/brent/brent.zhang@utexas.edu/CapitalOne_Team/Hashed Data/test_data.txt','r') as f:
        for line in f:
            termina_out=''
            line_num = line_num + 1
            # get vector 'words' from line (69 elements) 
            words = line.split('|')
            # initialize words_proc
            # words_proc = ['null']*69
            # process for words[0]
            for i in range(0,69):
                if i in termina_set: #and i not in null_set:
                    methodToCall = getattr(wp, 'wordProc_'+str(i))
                    temp = methodToCall(words[i])
                    termina_out += temp+','
            # write the line to the file, end with '\n'(enter)
            # f_out.write(','.join(words_proc)+'\n')
            # the following is just a test
            f_out.write(termina_out+'\n')
            if line_num>300000:
                break
    f_out.close()
    print line_num

print "hello world!"
myFunc_termina()

termina_set=[10,19,25,26,27,29,31,32,33,36,37,44,45,49,56,57,58,63,64,65,66]
null_set=[4,9,21,22,25,32,36,38,40,41,46,47,48,50,51]
# S = myFunc2(27)
# print S

# print wp.wordProc_66('001')