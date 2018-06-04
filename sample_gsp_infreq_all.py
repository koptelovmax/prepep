import numpy as np

train_set_size = 902
num_samplings = 200
total_num = 152636
#num_features = []

#f_num = open('num_infreq.csv', 'r')
#for line in f_num:
#    try:
#        num_features.append(line)
#    except ValueError:
#        print "Invalid input:", line
#f_num.close()
#%%
for j in range(num_samplings):
    f = open('infreq_hless.gsp', 'r')
    f_training = open('classifier//infreq_train_'+str(j+1)+'.gsp', 'w')
    #f_num_training = open('classifier//num_infreq_training_'+str(j+1)+'.csv', 'w')
    
    train_indexes = np.zeros(train_set_size,np.int)
    for i in range(train_set_size):
        number = np.random.randint(0,total_num+1,1)
        while (number in train_indexes):
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
                    #f_num_training.write(num_features[index])
        except ValueError:
            print "Invalid input:", line
    f.close()
    f_training.close()
    #f_num_training.close()
#%%
