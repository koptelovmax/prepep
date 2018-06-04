import numpy as np

k = 10 #number of folds
num_to_gen = 90 #dataset for test
total_num = 902
num_features = []

f_num = open('num_freq.csv', 'r')
#%%
sample_indexes = np.zeros((k,num_to_gen),np.int)
for j in range(k):
    for i in range(num_to_gen):
        number = np.random.randint(0,total_num+1,1)
        while number in sample_indexes:
            number = np.random.randint(0,total_num+1,1)
        sample_indexes[j,i] = number
np.size(np.unique(sample_indexes))
#%%
for line in f_num:
    try:
        num_features.append(line)
    except ValueError:
        print "Invalid input:", line
#%%
for j in range(k):
    f = open('freq_hless.gsp', 'r')
    f_training = open("train_data//freq_training_"+str(j+1)+".gsp", "w")
    f_test = open("test_data//freq_test_"+str(j+1)+".gsp", "w")
    f_num_training = open("num_features//num_freq_training_"+str(j+1)+".csv", "w")
    f_num_test = open("num_features//num_freq_test_"+str(j+1)+".csv", "w")
    for line in f:
        try:
            if ('t #' or 't#') in line:
                data = line.split('#')
                index = np.int(data[1])
                if (index+1) not in sample_indexes[j]:
                    f_training.write(line)
                    while not (line == '\n'):
                        line = f.next()
                        f_training.write(line)
                    f_num_training.write(num_features[index])
                else:
                    f_test.write(line)
                    while not (line == '\n'):
                        line = f.next()
                        f_test.write(line)
                    f_num_test.write(num_features[index])
        except ValueError:
            print "Invalid input:", line
    f.close()
    f_training.close()
    f_test.close()
    f_num_training.close()
    f_num_test.close()

f_num.close()
#%%
