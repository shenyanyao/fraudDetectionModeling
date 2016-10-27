def wordProc_0(s):
    # do preprocessing for str
    # AUTH_ID: meaningless
    list = [s]
    return ','.join(list)
def wordProc_1(s):
    # ACCT_ID_TOKEN: a number that is uniquely associated 
    # with single card holder account
    list = [s]
    return ','.join(list)
def wordProc_2(s):
    # FRD_IND: "Y" if the transaction was fraud, "N" if 
    # it was non-fraud
    switcher = {'Y':'1','N':'0'}
    list = [switcher.get(s,'-1')]
    return ','.join(list)
def wordProc_3(s):
    # ACCT_ACTVN_DT: the date when the first plastic on 
    # this account was activated.
    if s == 'null':
        return '0'
    year  = float(s[0:4])
    month = float(s[5:7])
    day   = float(s[8:10])
    value = (year - 2011)*365 + (month-1)*365/12 + day
    list = [str(value)]
    return ','.join(list)
def wordProc_4(s):
    # ACCT_AVL_CASH_BEFORE_AMT: the cash available money 
    # on the account
    list = [s]
    return ','.join(list)
def wordProc_5(s):
    # ACCT_AVL_MONEY_BEFORE_AMT: the available money on 
    # the account
    list = [s]
    return ','.join(list)
def wordProc_6(s):
    # ACCT_CL_AMT: current credit limit on the account
    list = [s]
    return ','.join(list)
def wordProc_7(s):
    # ACCT_CURR_BAL: the current balance of the acount 
    # at the end of posting
    list = [s]
    return ','.join(list)
def wordProc_8(s):
    # ACCT_MULTICARD_IND: indicates if the account is a 
    # multi-card account
    switcher = {'null':'0|0','Y':'1|0','N':'0|1'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_9(s):
    # ACCT_OPEN_DT: the date the account was opend in
    # the system of record
    if s == 'null':
        return '0'
    year  = float(s[0:4])
    month = float(s[5:7])
    day   = float(s[8:10])
    value = (year - 2011)*365 + (month-1)*365/12 + day
    list = [str(value)]
    return ','.join(list)
def wordProc_10(s):
    # ACCT_PROD_CD: the product code is defined by capital
    # one and is independent of any system. in populating
    # this element and the description, the product cross
    # reference table must be used
    switcher = {'C23':0,'C14':1,'001':2,'C01':3,'010':4,'011':5,'012':6,
                '021':7,'015':8,'023':9,'022':10,'014':11,'040':12,'029':13,
                '053':14}
    #{'029', 'C23', '015', '052', '030', '053', '050', '022', '002', '021', '014', 
    #'020', '023', '009', '260', '255', '051', 'C01', 'C21', '259', '258', '253', 
    #'013', '040', '251', '001', '252', 'L01', '026', 'L05', '010', '012', '057', 
    #'056', 'C14', '048', 'L06', 'L04', '024', '256', '025', '254', '019', '058', '011', 'L03'}
    list = ['0']*16
    list[switcher.get(s,15)] = '1'
    list = list[0:-2]
    #list = [switcher.get(s,'-1')]
    return ','.join(list)
def wordProc_11(s):
    # ACCT_TYPE_CD: code identifying the account type
    switcher = {'B':'1|0','E':'0|1','null':'0|0'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_12(s):
    # ADR_VFCN_FRMT_CD: a code that specifies the address
    # verification system format used for billing or ship-
    # ping address verification
    switcher = {'X':'1|0','null':'0|0','4':'0|1'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_13(s):
    # ADR_VFCN_RESPNS_CD: a code that describes the outcome
    # of the address verification
    switcher = {'5A':'1|0|0|0|0|0','5N':'0|1|0|0|0|0','5Y':'0|0|1|0|0|0',
    '5Z':'0|0|0|1|0|0','5X':'0|0|0|0|1|0','5W':'0|0|0|0|0|1','null':'0|0|0|0|0|0'}
    #{'5X', '5A', '5N', 'null', '5Z', '5Y', '5W'}
    list = switcher.get(s,'-1|-1|-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_14(s):
    # APPRD_AUTHZN_CNT: total number of authorizations
    list = [s]
    return ','.join(list)
def wordProc_15(s):
    # APPRD_CASH_AUTHZN_CNT: total number of cash authorizations
    list = [s]
    return ','.join(list)
def wordProc_16(s):
    # identifies the result of the authorization request cryptogram (arqc) validation. 
    list = []
    return ','.join(list)
def wordProc_17(s):
    # status on the account at the disposition of this transaction.
    switcher = {'A':'1|0','D':'0|0','R':'0|1'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_18(s):
    # the transaction amount is amount of funds requested by the cardholder in the authorization request.
    list = [s]
    return ','.join(list)
def wordProc_19(s):
    # category type of the authorization
    switcher = {'9':'1|0|0|0|0|0|0|0|0|0','8':'0|1|0|0|0|0|0|0|0|0','7':'0|0|1|0|0|0|0|0|0|0',
    '6':'0|0|0|1|0|0|0|0|0|0','5':'0|0|0|0|1|0|0|0|0|0','4':'0|0|0|0|0|1|0|0|0|0','3':'0|0|0|0|0|0|1|0|0|0',
    '2':'0|0|0|0|0|0|0|1|0|0','1':'0|0|0|0|0|0|0|0|1|0','0':'0|0|0|0|0|0|0|0|0|1','null':'0|0|0|0|0|0|0|0|0|0'}
    #{'2', '3',  , 'null', '1', '0', '5',}
    list = switcher.get(s,'-1|-1|-1|-1|-1|-1|-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_20(s):
    # specifies the requested custom payment service processing on input
    switcher = {'U':'1|0|0|0|0|0|0|0|0|0|0|0|0','A':'0|1|0|0|0|0|0|0|0|0|0|0|0','C':'0|0|1|0|0|0|0|0|0|0|0|0|0',
    'E':'0|0|0|1|0|0|0|0|0|0|0|0|0','I':'0|0|0|0|1|0|0|0|0|0|0|0|0',
    'K':'0|0|0|0|0|1|0|0|0|0|0|0|0','N':'0|0|0|0|0|0|1|0|0|0|0|0|0','P':'0|0|0|0|0|0|0|1|0|0|0|0|0','R':'0|0|0|0|0|0|0|0|1|0|0|0|0',
    'S':'0|0|0|0|0|0|0|0|0|1|0|0|0','T':'0|0|0|0|0|0|0|0|0|0|1|0|0','V':'0|0|0|0|0|0|0|0|0|0|0|1|0','W':'0|0|0|0|0|0|0|0|0|0|0|0|1',
    'null':'0|0|0|0|0|0|0|0|0|0|0|0|0'}
    list = switcher.get(s,'-1|-1|-1|-1|-1|-1|-1|-1|-1|-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_21(s):
    # an option set is a group of related terms/settings that can be applied to an account. 
    #the option set identifier is a unique identifier for a specific set of values for the 
    #terms/settings in the option set that have been assigned to an account.
    switcher = {'2':'1','1':'0'}
    list = [switcher.get(s,'-1')]
    return ','.join(list)
def wordProc_22(s):
    # a code that identifies the original sender of the transaction message
    switcher = {'B':'1','I':'0'}
    list = [switcher.get(s,'-1')]
    return ','.join(list)
def wordProc_23(s):
    # authorization outstanding amount
    list = [s]
    return ','.join(list)
def wordProc_24(s):
    # this is the amount of unmatched approved cash authorization requests as of the previous posting 
    list = [s]
    return ','.join(list)
def wordProc_25(s):
    # a code that identifies how the authorization  request is processed
    switcher = {'5100':'1','1000':'0'}
    list = [switcher.get(s,'-1')]
    return ','.join(list)
def wordProc_26(s):
    # for all message types except advice records, this is the date of the authorization request.
    if s == 'null':
        return '0'
    year  = float(s[0:4])
    month = float(s[5:7])
    day   = float(s[8:10])
    value = (year - 2011)*365 + (month-1)*365/12 + day
    list = [str(value)]
    return ','.join(list)
def wordProc_27(s):
    a = s.split(':')
    b = float(a[0])*3600 + float(a[1])*60 + float(a[2])
    return str(b)
def wordProc_28(s):
    # a code that is received in the authorization message which describes the specific type of authorization request.
    switcher = {'50':'1|0|0','00':'0|0|0','17':'0|1|0','11':'0|0|1'}
    list = switcher.get(s,'-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_29(s):
    # a one digit number that identifies the capability of the authorization terminal, if one was used, to capture pins.
    switcher = {'8':'1|0|0','2':'0|1|0','1':'0|0|1','null':'0|0|0'}
    list = switcher.get(s,'-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_30(s):
    # the average daily authorization amount on the plastic since the day of first use.
    # it is calculated using the number of days since the first use.
    list = [s]
    return ','.join(list)
def wordProc_31(s):
    # indicates if the card verification value was validated and the result of the validation.
    switcher = {'M':'1|0','N':'0|0','null':'0|1'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_32(s):
    # duration in days since when the plastic was issued.
    list = [s]
    return ','.join(list)
def wordProc_33(s):
    # CARD_VFCN_MSMT_REAS_CD: 
    # the cvi (cvv/cvc) value did not match, 
    #this field contains the reason the value did not match
    # verification system format used for billing or ship-
    # ping address verification
    switcher = {'F':'1|0|0|0','S':'0|1|0|0','N':'0|0|1|0','A':'0|0|0|1','null':'0|0|0|0'}
    #{'N', 'S', 'null', 'F', 'A'}
    list = switcher.get(s,'-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_34(s):
    # CARD_VFCN_PRESNC_CD: 0-1
    # a code that uniquely identifies the presence of the
    # cvi2 value on the authorization
    # and the result of the validation.
    switcher = {'9':'1|0|0|0','2':'0|1|0|0','1':'0|0|1|0','0':'0|0|0|1','null':'0|0|0|0'}
    #{'2', 'null', '1', '9', '0'}
    list = switcher.get(s,'-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_35(s):
    switcher = {'X':'1|0|0','M':'0|1|0','N':'0|0|1','null':'0|0|0'}
    #{'M', 'null', 'X', 'N'}
    list = switcher.get(s,'-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_36(s):
    switcher = {'M':'1|0|0|0','S':'0|1|0|0','N':'0|0|1|0','P':'0|0|0|1','null':'0|0|0|0'}
    #{'M', 'S', 'null', 'N', 'P'}
    list = switcher.get(s,'-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_37(s):
    #CDHLDR_PRES_CD: card holder presence
    switcher = {'2':0,'4':1,'3':2,'null':3,'1':4,'0':5,'5':6}
    #{'2', '4', '3', 'null', '1', '0', '5'}
    list = ['0']*8
    list[switcher.get(s,7)] = '1'
    if list[7]=='1':
        list=['-1']*6
    else:
        list = list[0:-2]
    return ','.join(list)
    
def wordProc_38(s):
    #this is the conversion rate for a currency type 
    #compared to the us dollars.the rate 
    #at which the transaction currency was 
    #converted into the client's currency (prior to markup).
    list = [s]
    return ','.join(list)
def wordProc_39(s):
    #indicates the level of security used 
    #for the electronic transmission of the authorization.
    #{'2', '4', '3', '6', '1', '0', '5', '*2', '9', '7'}
    switcher = {'0':0,'1':1,'2':2,'*2':3,'3':4,'4':5,'5':6,'6':7,'7':8,'9':9}
    list = ['0']*11
    list[switcher.get(s,10)] = '1'
    if list[10]=='1':
        list=['-1']*9
    else:
        list = list[0:-2]
    return ','.join(list)
def wordProc_40(s):
    #HOME_PHN_NUM_CHNG_DUR
    list = [s]
    return ','.join(list)
def wordProc_41(s):
    #HOTEL_STAY_CAR_RENTL_DUR
    list = [s]
    return ','.join(list)
def wordProc_42(s):
    #LAST_ADR_CHNG_DUR
    list = [s]
    return ','.join(list)
def wordProc_43(s):
    #LAST_PLSTC_RQST_REAS_CD
    #reason
    #{'C', 'T', 'N', 'S', 'null', 'O', 'A', 'E', 'R', 'D'}
    switcher = {'C':0,'T':1,'N':2,'S':3,'null':4,'0':5,'A':6,'E':7,'R':8,'D':9}
    list = ['0']*11
    list[switcher.get(s,10)] = '1'
    if list[10]=='1':
        list=['-1']*9
    else:
        list = list[0:-2]
    return ','.join(list)
def wordProc_44(s):
    list = [s]
    return ','.join(list)
def wordProc_45(s):
    #MRCH_CNTRY_CD
    #merchant country!!!
    switcher = {'840':'1'}
    list = [switcher.get(s,'-1')]
    return ','.join(list)
    #{'276', '643', '076', '398', '188', '246', '630', '840', '792', '344', '764',
    # '233', '480', '724', '380', '484', '312', '280', '702', '356', '850', '376',
    # '060', '616', '608', '578', '156', '533', '410', '458', '752', '250'}
def wordProc_46(s):
    list = [s]
    return ','.join(list)
def wordProc_47(s):
    #PHN_CHNG_SNC_APPN_IND  
    #an indicator that specifies if the phone number 
    #has ever been changed on the account, 
    #since the account application was processed.
    switcher = {'5':'1','8':'0'}
    list = [switcher.get(s,'-1')]
    return ','.join(list)
def wordProc_48(s):
    #MRCH_CNTRY_CD
    #merchant country!!!
    list = []
    return ','.join(list)
def wordProc_49(s):
    switcher = {'N':'1|0','Y':'0|1','null':'0|0'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
# SUN from 50 - end
def wordProc_50(s):
    # date when the newly issued plastic was activated.
    if s == 'null':
        return '0'
    year  = float(s[0:4])
    month = float(s[5:7])
    day   = float(s[8:10])
    value = (year - 2011)*365 + (month-1)*365/12 + day
    list = [str(value)]
    return ','.join(list)
def wordProc_51(s):
    # an indicator that specifies if the card that was mailed out requires activation.
    switcher = {'N':'1|0','Y':'0|1','null':'0|0'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_52(s):
    # ACCT_OPEN_DT: the date the account was opend in
    # the date and time when the plastic was first used.
    if s == 'null':
        return '0'
    year  = float(s[0:4])
    month = float(s[5:7])
    day   = float(s[8:10])
    value = (year - 2011)*365 + (month-1)*365/12 + day
    list = [str(value)]
    return ','.join(list)
def wordProc_53(s):
    # duration in days since when the plastic was issued.
    list = [s]
    return ','.join(list)
def wordProc_54(s):
    # this indicates whether the card used by the card holder is the current card or a previous card on file.
    switcher = {'C':'1|0','M':'0|0','P':'0|1'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_55(s):
    # Date and time when newly issued card was requested.
    if s == 'null':
        return '0'
    year  = float(s[0:4])
    month = float(s[5:7])
    day   = float(s[8:10])
    value = (year - 2011)*365 + (month-1)*365/12 + day
    list = [str(value)]
    return ','.join(list)
def wordProc_56(s):
    # describes the condition under which the authorization took place. au01/ha01-pos-cond-code
    #{'02', '71',   '05',}
    switcher = {'71':'1|0|0|0|0|0|0|0','52':'0|1|0|0|0|0|0|0','51':'0|0|1|0|0|0|0|0',
    '08':'0|0|0|1|0|0|0|0','06':'0|0|0|0|1|0|0|0','05':'0|0|0|0|0|1|0|0',
    '02':'0|0|0|0|0|0|1|0','01':'0|0|0|0|0|0|0|1','00':'0|0|0|0|0|0|0|0'}
    list = switcher.get(s,'-1|-1|-1|-1|-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_57(s):
    # this is a two digit code that identifies the actual method 
    #used at the point of service to enter the cardholder account number.
    #{'90', '02', '82', '81', '91', '00', '01'}
    switcher = {'90':'1|0|0|0|0|0','02':'0|1|0|0|0|0','82':'0|0|1|0|0|0',
    '81':'0|0|0|1|0|0','91':'0|0|0|0|1|0','00':'0|0|0|0|0|1','01':'0|0|0|0|0|0'}
    list = switcher.get(s,'-1|-1|-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_58(s):
    # indicates if the authorization occurs on a regular basis. 
    switcher = {'N':'1|0','Y':'0|1','null':'0|0'}
    list = switcher.get(s,'-1|-1')
    return ','.join(list.split('|'))
def wordProc_59(s):
    # iindicates if the reversal was processed.
    switcher = {'Y':'1','null':'0'}
    list = [switcher.get(s,'-1')]
    return ','.join(list)
def wordProc_60(s):
    # it identifies the country of residence associated with the sender's primary address. 
    list = []
    return ','.join(list)
def wordProc_61(s):
    # code which reflects the currency code of the original transaction 
    #{'608', '356', '643', '156', '985', '756', '978', '986', '398', '826', '376', '188', '484', '410', '840', '036', '949', '344', '764'}
    list = []
    return ','.join(list)
def wordProc_62(s):
    #one of the fields used in currency conversion, if required.
    #it is an exponential value used in the conversion. 
    list = []
    return ','.join(list)
def wordProc_63(s):
    # this is a two digit code that identifies the actual method 
    #used at the point of service to enter the cardholder account number.
    #{'2', 'null', '1', '0'}
    switcher = {'1':'1|0|0','null':'0|0|0','0':'0|1|0','2':'0|0|1'}
    list = switcher.get(s,'-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_64(s):
    # identifies the capability of the terminal to 
    #electronically read account number and expiration dates from cards.
    #{'2', '4', '6', '8', '1', '0', '5', '9', '7'}
    switcher = {'2':0,'4':1,'6':2,'8':3,'1':4,'0':5,'5':6,'9':7,'7':8}
    list = ['0']*10
    list[switcher.get(s,9)] = '1'
    if list[9]=='1':
        list=['-1']*8
    else:
        list = list[0:-2]
    return ','.join(list)
def wordProc_65(s):
    # identifies the basic category of the electronic terminal (if any) used at the point 
    #of service and is used to identify atm transactions. the data in this field is provided 
    #by the acquiring center for the transaction. the terminal classification identifies the
    # basic category of the electronic terminal used at the point of service.
    #{'2', '4', '3', '1', '0', '5', '7'}
    switcher = {'2':'1|0|0|0|0|0','4':'0|1|0|0|0|0','3':'0|0|1|0|0|0',
    '1':'0|0|0|1|0|0','0':'0|0|0|0|1|0','5':'0|0|0|0|0|1','7':'0|0|0|0|0|0'}
    list = switcher.get(s,'-1|-1|-1|-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_66(s):
    # terminal identifier of the terminal used within the store where the transaction occurred. 
    switcher = {'001':'1|0|0','00361922':'0|1|0','10010008':'0|0|1','null':'0|0|0'}
    list = switcher.get(s,'-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_67(s):
    # identifies the capability of the authorization terminal, if one was used, to capture pins
    switcher = {'8':'1|0|0','2':'0|1|0','1':'0|0|1','0':'0|0|0'}
    list = switcher.get(s,'-1|-1|-1')
    return ','.join(list.split('|'))
def wordProc_68(s):
    # Approximate distance of customer's home from merchant. Set to 7000 for non-USA purchases, 0 if same zip-5 code.
    temp=s.split('\n')
    list=[temp[0]]
    return ','.join(list)
# this function calculate the elements for ath column
