from random import randint
def myFunc1():
    line_num = 0
    line_samp_num = 0
    f_out = open('test_data.txt','a')
    f_out_fraud = open('train_fraud.txt','a')
    f_o1 = open('train_nofraud1.txt','a')
    f_o2 = open('train_nofraud2.txt','a')
    f_o3 = open('train_nofraud3.txt','a')
    f_o4 = open('train_nofraud4.txt','a')
    f_o5 = open('train_nofraud5.txt','a')
    f_o6 = open('train_nofraud6.txt','a')
    f_o7 = open('train_nofraud7.txt','a')
    f_o8 = open('train_nofraud8.txt','a')
    f_o9 = open('train_nofraud9.txt','a')
    p_hash = 32452843  
    a_hash = 1327131
    b_hash = 567128
    for n in ['10']:
		filename = 'D:/CapitalOne/data/training_part_'+n+'_of_10.txt'
		with open(filename,'r') as f:
			for line in f:
				line_num = line_num + 1
				t = randint(0,9)
				if t%10 == 0:
					f_out.write(line)
					continue
				words = line.split('|')
				if words[2] == 'Y':
					f_out_fraud.write(line)
					continue
				else:
					if t%10 == 1:
						f_o1.write(line)
						continue
					if t%10 == 2:
						f_o2.write(line)
						continue
					if t%10 == 3:
						f_o3.write(line)
						continue
					if t%10 == 4:
						f_o4.write(line)
						continue
					if t%10 == 5:
						f_o5.write(line)
						continue
					if t%10 == 6:
						f_o6.write(line)
						continue
					if t%10 == 7:
						f_o7.write(line)
						continue
					if t%10 == 8:
						f_o8.write(line)
						continue
					if t%10 == 9:
						f_o9.write(line)
						continue
	
    f_out_fraud.close()
    f_out.close()
    f_o1.close()
    f_o2.close()
    f_o3.close()
    f_o4.close()
    f_o5.close()
    f_o6.close()
    f_o7.close()
    f_o8.close()
    f_o9.close()

def myFunc2():
	f_out1 = open('test_data_fraud.txt','a')
	f_out2 = open('test_data_nofraud.txt','a')
	filename = 'D:/CapitalOne/data/hashed/test_data.txt'
	with open(filename,'r') as f:
		for line in f:
			words = line.split('|')
			if words[2] == 'Y':
				f_out1.write(line)
			else:
				f_out2.write(line)
	f_out1.close()
	f_out2.close()

myFunc2()
    