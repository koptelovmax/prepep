#%%
import numpy as np
#%%
name = 'freq' #'infreq'
f = open(name+'.gsp', 'r')
f_out = open(name+'_hless.gsp', 'w')
count = 0

for line in f:
    try:
        if '#' in line:
            count+=1
            f_out.write(line)
            line = f.next()
            h_list = []
            while 'v' in line:
                data = line.split(' ')
                if np.int(data[2]) != 1:
                    f_out.write(line)
                else:
                    h_list.append(data[1])
                line = f.next()
            while 'e' in line:
                data = line.split(' ')
                if (data[1] not in h_list) and (data[2] not in h_list):
                    f_out.write(line)                
                line = f.next()
            f_out.write('\n')
    except ValueError:
        print "Invalid input:", line
        
f.close()
f_out.close()
#%%
