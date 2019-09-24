import numpy as np
np.set_printoptions(precision=4)
import pickle
import pdb
import math


def get_data_stats():
    data = pickle.load(open('data.pkl','rb'))

    stats = []
    avg_stats = []

    for sub in data:
        
        total = {'1':0 , '3': 0, '5': 0}
        correct_first = {'1':0 , '3': 0, '5': 0}
        incorrect_second_correct_first_i = {'1':0 , '3': 0, '5': 0}
        incorrect_second_correct_first_r = {'1':0 , '3': 0, '5': 0}
        correct_second_incorrect_first_i = {'1':0 , '3': 0, '5': 0}
        correct_second_incorrect_first_r = {'1':0 , '3': 0, '5': 0}
        
        for trial in data[sub]:
            for pair in data[sub][trial]:

                datum = data[sub][trial][pair]
                reps = datum['xREPS']
                total[reps] += 1

                if datum['ANSWER1'] == 'C':
                    correct_first[reps] += 1

                if ('B1' in datum and 'B2' in datum) or ('F1' in datum and 'F2' in datum):
                    identical = 1
                else:
                    identical = 0
                
                if datum['ANSWER1'] == 'C' and datum['ANSWER2'] != 'C':
                    if identical:
                        incorrect_second_correct_first_i[reps] += 1
                    else:
                        incorrect_second_correct_first_r[reps] += 1

                if datum['ANSWER1'] != 'C' and datum['ANSWER2'] == 'C':
                    if identical:
                        correct_second_incorrect_first_i[reps] += 1
                    else:
                        correct_second_incorrect_first_r[reps] += 1
        
        stats.append(list(correct_first.values()) + list(incorrect_second_correct_first_i.values()) + list(correct_second_incorrect_first_i.values()) + list(incorrect_second_correct_first_r.values()) + list(correct_second_incorrect_first_r.values()))
        avg_stats.append([sum(list(correct_first.values())), sum(list(incorrect_second_correct_first_i.values())), sum(list(correct_second_incorrect_first_i.values())), sum(list(incorrect_second_correct_first_r.values())), sum(list(correct_second_incorrect_first_r.values()))])


    avg_stats = np.array(avg_stats).astype(float)
    incorrect_2_correct_1_i = avg_stats[:,1]/avg_stats[:,0]
    correct_2_incorrect_1_i = avg_stats[:,2]/(72-avg_stats[:,0])
    incorrect_2_correct_1_r = avg_stats[:,3]/avg_stats[:,0]
    correct_2_incorrect_1_r = avg_stats[:,4]/(72-avg_stats[:,0])

    avg_stats[:,1] = avg_stats[:,1]/avg_stats[:,0]
    avg_stats[:,2] = avg_stats[:,2]/(72-avg_stats[:,0])
    avg_stats[:,3] = avg_stats[:,3]/avg_stats[:,0]
    avg_stats[:,4] = avg_stats[:,4]/(72-avg_stats[:,0])
    avg_stats[:,0] = avg_stats[:,0]/72

    difference_i = incorrect_2_correct_1_i - correct_2_incorrect_1_i
    difference_r = incorrect_2_correct_1_r - correct_2_incorrect_1_r

    t_r = np.mean(difference_r)/(np.std(difference_r)/math.sqrt(15))
    t_i = np.mean(difference_i)/(np.std(difference_i)/math.sqrt(15))
    print(t_i, t_r)
    return avg_stats


if __name__ == "__main__" :
    stats = get_data_stats()
    print(stats)



