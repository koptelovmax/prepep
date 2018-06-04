import numpy as np
import sys

exp_id = sys.argv[1]
test_set_size = 90 #90/900/9000
train_set_size = 812
num_samplings = 200
total_num = 152636
num_features = []
#%%
test_indexes = np.zeros(test_set_size,np.int)
for i in range(test_set_size):
    number = np.random.randint(0,total_num+1,1)
    while number in test_indexes:
        number = np.random.randint(0,total_num+1,1)
    test_indexes[i] = number
#print np.size(np.unique(test_indexes))
#%%
f_num = open('num_infreq.csv', 'r')
for line in f_num:
    try:
        num_features.append(line)
    except ValueError:
        print "Invalid input:", line
f_num.close()
#%%
f = open('infreq_hless.gsp', 'r')
f_test = open('test_data//infreq_test_'+str(exp_id)+'.gsp', 'w')
f_num_test = open('num_features//num_infreq_test_'+str(exp_id)+'.csv', 'w')

for line in f:
    try:
        if ('t #' or 't#') in line:
            data = line.split('#')
            index = np.int(data[1])
            if index in test_indexes:
                f_test.write(line)
                while not (line == '\n'):
                    line = f.next()
                    f_test.write(line)
                f_num_test.write(num_features[index])
    except ValueError:
        print "Invalid input:", line
f.close()
f_test.close()
f_num_test.close()
#%%
for j in range(num_samplings):
    f = open('infreq_hless.gsp', 'r')
    f_training = open('train_data//infreq_training_'+str(exp_id)+'_'+str(j+1)+'.gsp', 'w')
    f_num_training = open('num_features//num_infreq_training_'+str(exp_id)+'_'+str(j+1)+'.csv', 'w')
    
    train_indexes = np.zeros(train_set_size,np.int)
    for i in range(train_set_size):
        number = np.random.randint(0,total_num+1,1)
        while (number in test_indexes) or (number in train_indexes):
            number = np.random.randint(0,total_num+1,1)
        train_indexes[i] = number
    
    for line in f:
        try:
            if ('t #' or 't#') in line:
                data = line.split('#')
                index = np.int(data[1])
                if index in train_indexes:
                    f_training.write(line)
                    while not (line == '\n'):
                        line = f.next()
                        f_training.write(line)
                    f_num_training.write(num_features[index])
        except ValueError:
            print "Invalid input:", line
    f.close()
    f_training.close()
    f_num_training.close()
#%%