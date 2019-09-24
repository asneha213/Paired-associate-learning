import pdb
import numpy as np
import pickle

DATA = 'as9.txt'

with open('as9.txt', 'r') as f:
    data = f.read()

data = data.split('\n')

new_data = {}

pair_num = 0
for item in data:
    if 'Subject:' in item:
        sub_num = item.split(':')[-1].strip(' ')
        if sub_num not in new_data:
            new_data[sub_num] = {}

    elif 'Trial:' in item:
        trial_num = item[-1]
        if trial_num not in new_data[sub_num]:
            new_data[sub_num][trial_num] = {}

    elif 'Test ' in item:
        test_num = item[5]

    elif 'PAIR:' in item:
        pair_num = item.split(':')[-1].strip(' ')
        if pair_num not in new_data[sub_num][trial_num]:
            new_data[sub_num][trial_num][pair_num] = {}
    
    elif 'PART:' in item:
        part_num = item[-1]

    elif 'CODE:' in item:
        code_num = item[-1]

    elif 'GIVEN:' in item:
        if part_num == '0':
            new_data[sub_num][trial_num][pair_num]['WORD1'] = item.split(':')[-1].strip(' ')
        if part_num == '1':
            new_data[sub_num][trial_num][pair_num]['WORD2'] = item.split(':')[-1].strip(' ')

    elif 'NEEDED:' in item:
        if part_num == '0':
            new_data[sub_num][trial_num][pair_num]['WORD2'] = item.split(':')[-1].strip(' ')
        if part_num == '1':
            new_data[sub_num][trial_num][pair_num]['WORD1'] = item.split(':')[-1].strip(' ')

    elif 'TIME:' in item:
        if test_num == '1':
            if code_num == '1':
                new_data[sub_num][trial_num][pair_num]['F1'] = item.split(':')[-1].strip(' ')
            if code_num == '2':
                new_data[sub_num][trial_num][pair_num]['F1'] = item.split(':')[-1].strip(' ')
            if code_num == '3':
                new_data[sub_num][trial_num][pair_num]['B1'] = item.split(':')[-1].strip(' ')
            if code_num == '4':
                new_data[sub_num][trial_num][pair_num]['B1'] = item.split(':')[-1].strip(' ')

        if test_num == '2':
            if code_num == '1':
                new_data[sub_num][trial_num][pair_num]['F2'] = item.split(':')[-1].strip(' ')
            if code_num == '2':
                new_data[sub_num][trial_num][pair_num]['B2'] = item.split(':')[-1].strip(' ')
            if code_num == '3':
                new_data[sub_num][trial_num][pair_num]['F2'] = item.split(':')[-1].strip(' ')
            if code_num == '4':
                new_data[sub_num][trial_num][pair_num]['B2'] = item.split(':')[-1].strip(' ')


    elif 'xREPS:' in item:
        new_data[sub_num][trial_num][pair_num]['xREPS'] = item[-1]

    elif 'ANSWER:' in item:
        if test_num == '1':
            new_data[sub_num][trial_num][pair_num]['ANSWER1'] = item[-1]
        if test_num == '2':
            new_data[sub_num][trial_num][pair_num]['ANSWER2'] = item[-1]


val_sub = []
s0 = []
s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
s6 = []
s7 = []
for sub in new_data:
    total = 0
    total_0 = 0
    num_1 =0
    num_2 =0
    num_3 =0
    num_4 =0
    num_5 =0
    num_6 =0
    tr_c = []

    for i in new_data[sub]:
        trc = 0
        trt = 0
        for j in new_data[sub][i]:
            db = new_data[sub][i][j]
            if db['xREPS'] == '1':
                total_0 += 1
                if ('F1' in db and 'B2' in db) or  ('B1' in db and 'F2' in db):
                    #if db['ANSWER2'] == 'N' or db['ANSWER1'] == 'N':
                    #    continue
                    total += 1
                    trt += 1
                    if db['ANSWER1'] == 'C':
                        num_5 += 1
                    if db['ANSWER1'] == 'C' and db['ANSWER2'] == 'C':
                        #print(db)
                        num_1 += 1
                        trc += 1

                    if db['ANSWER1'] != 'C' and db['ANSWER2'] == 'C':
                        num_2 += 1

                    if db['ANSWER1'] == 'C' and db['ANSWER2'] != 'C':
                        num_3 += 1

                    if db['ANSWER1'] != 'C' and db['ANSWER2'] != 'C':
                        num_4 += 1
        tr_c.append(float(trc)/trt)
    '''
    for i in new_data[sub]:
        trc = 0
        trt = 0
        for j in new_data[sub][i]:
            db = new_data[sub][i][j]
            if db['xREPS'] == '3':
                if db['ANSWER2'] == 'N':
                    print(db)
                    #continue
                total_0 += 1
                if 'F1' in db :
                    total += 1
                    trt += 1
                    if db['ANSWER2'] == 'C':
                        num_1 += 1
                        trc += 1
        print(trc, trt)
        #tr_c.append(float(trc)/trt)
    '''

    s0.append(total)
    s1.append(num_1)
    s2.append(num_2)
    s3.append(num_3)
    s4.append(num_4)
    s5.append(num_5)
    s6.append(num_6)
    s7.append(np.mean(tr_c))

print(np.sum(s1)/np.sum(s0))
print(np.sum(s2)/np.sum(s0))
print(np.sum(s3)/np.sum(s0))
print(np.sum(s4)/np.sum(s0))
print(np.mean(s7))


pdb.set_trace()
